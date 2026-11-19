"""Nominal-policy rollout collection and PNCBF certificate regression."""
from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .constraint_model import ARENA_PLANES, PNCBFCertificate
from .schedule import PNCBFDiscountSchedule
from .state import PNCBFObservationModel
from .targets import pncbf_value_targets


@dataclass
class PNCBFTrainConfig:
    hidden: list = field(default_factory=lambda: [256, 256, 256])
    rollout_steps: int = 64
    updates: int = 10_000
    gradient_steps: int = 8
    batch_size: int = 16384
    replay_rollouts: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-1
    discount_schedule: PNCBFDiscountSchedule = field(default_factory=PNCBFDiscountSchedule)
    target_tau: float = 0.005
    grad_norm_clip: float = 10.0
    checkpoint_interval: int = 2500
    # Policy iteration: earlier certificate filtering the nominal policy (set by scripts/train.py).
    policy_filter: dict | None = None

    @classmethod
    def from_dict(cls, values: dict) -> "PNCBFTrainConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown PNCBF config keys: {sorted(unknown)}")
        values = dict(values)
        if isinstance(values.get("discount_schedule"), dict):
            values["discount_schedule"] = PNCBFDiscountSchedule(**values["discount_schedule"])
        cfg = cls(**values)
        counts = ("rollout_steps", "updates", "gradient_steps", "batch_size", "replay_rollouts", "checkpoint_interval")
        if any(getattr(cfg, key) < 1 for key in counts):
            raise ValueError("PNCBF count and interval settings must be positive")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


class PNCBFTrainer:
    """Fit the certificate V of a frozen nominal policy from its rollouts.

    Targets are built per obstacle identity, then mapped to observation slots.
    """

    def __init__(self, *, env, policy, obs_layout, command_scale, step_dt: float, config: PNCBFTrainConfig,
                 output_dir: str, run_info: dict) -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = policy
        self.device = torch.device(env.device)
        self.codec = PNCBFObservationModel(obs_layout)
        self.command_scale = [float(v) for v in command_scale]
        self.step_dt = float(step_dt)
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.run_info = dict(run_info)
        self.model = PNCBFCertificate(obs_layout, hidden=config.hidden, device=self.device)
        self.target = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.replay = deque(maxlen=config.replay_rollouts)
        self.obs = None
        self.frames = 0
        self.optimizer_steps = 0
        self.total_optimizer_steps = config.updates * config.gradient_steps
        self.discount_lambda = config.discount_schedule.value(0, self.total_optimizer_steps, self.step_dt)
        self.bootstrap_weight = config.discount_schedule.bootstrap_weight(0, self.total_optimizer_steps)

    @property
    def gamma(self) -> float:
        return math.exp(-self.discount_lambda * self.step_dt)

    def _slot_identity(self) -> torch.Tensor:
        """Allocated obstacle shown by each slot of the latest observation (-1: padding)."""
        return self.base_env._obstacle_observation_identity.detach().to("cpu", torch.int16)

    def collect(self, steps: int | None = None) -> None:
        steps = self.cfg.rollout_steps if steps is None else steps
        if self.obs is None:
            self.obs, _ = self.env.reset()
            self.policy.reset(None)
        states, constraints, dones, terminal_constraints, identities = [], [], [], [], []
        for _ in range(steps):
            states.append(self.codec.decode(self.obs).detach().cpu())
            identities.append(self._slot_identity())
            # True constraints behind the current observation.
            constraints.append(self.base_env.pncbf_constraints().detach().cpu())
            with torch.inference_mode():
                action = self.policy.act(self.obs)
            self.obs, _reward, terminated, truncated, _info = self.env.step(action)
            self.frames += int(self.obs.shape[0])
            done = torch.logical_or(terminated, truncated).reshape(-1)
            dones.append(done.detach().cpu())
            terminal = torch.zeros_like(constraints[-1])
            if done.any():
                # step() already reset these rows: take the endpoint from the pre-reset snapshot.
                endpoint = self.base_env.get_last_step_snapshot()["pncbf_constraints"].detach().cpu()
                endpoint = torch.where(torch.isfinite(endpoint), endpoint, constraints[-1])
                terminal = torch.where(dones[-1][:, None], endpoint, terminal)
                self.policy.reset(done.nonzero(as_tuple=False).reshape(-1))
            terminal_constraints.append(terminal)

        states.append(self.codec.decode(self.obs).detach().cpu())
        constraints.append(self.base_env.pncbf_constraints().detach().cpu())
        identities.append(self._slot_identity())
        self.replay.append({
            "state": torch.stack(states),
            "constraint": torch.stack(constraints),
            "done": torch.stack(dones),
            "terminal_constraint": torch.stack(terminal_constraints),
            "identity": torch.stack(identities),
        })

    @staticmethod
    def _to_slots(per_identity: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """[..., obstacles + 6] by identity -> [..., slots + 6] in observation order."""
        slots = torch.gather(per_identity[..., :-ARENA_PLANES], -1, identity.long().clamp_min(0))
        return torch.cat((slots, per_identity[..., -ARENA_PLANES:]), dim=-1)

    def _training_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_states, all_targets, all_constraints, all_masks = [], [], [], []
        with torch.no_grad():
            for rollout in self.replay:
                state = rollout["state"].to(self.device)
                constraint = rollout["constraint"].to(self.device)
                done = rollout["done"].to(self.device)
                identity = rollout["identity"].to(self.device).long()
                terminal = rollout["terminal_constraint"].to(self.device)
                # Bootstrap V(x_T) from the target network, moved from slots to identities.
                values = self.target(state[-1])
                bootstrap = constraint.new_full(constraint.shape[1:], -1.0)
                final = identity[-1]
                rows = (final >= 0).nonzero(as_tuple=True)
                bootstrap[rows[0], final[rows]] = values[:, :-ARENA_PLANES][rows]
                bootstrap[:, -ARENA_PLANES:] = values[:, -ARENA_PLANES:]
                targets = pncbf_value_targets(
                    constraint, bootstrap, done,
                    discount_lambda=self.discount_lambda, step_dt=self.step_dt, terminal_constraints=terminal,
                )
                if self.bootstrap_weight < 1.0:
                    # Upstream tgt_rhs: lhs + w * (target - lhs), lhs taking V(x_T) = h(x_T).
                    in_rollout = pncbf_value_targets(
                        constraint, constraint[-1], done,
                        discount_lambda=self.discount_lambda, step_dt=self.step_dt, terminal_constraints=terminal,
                    )
                    targets = in_rollout + self.bootstrap_weight * (targets - in_rollout)
                targets = self._to_slots(targets, identity[:-1])
                constraint = self._to_slots(constraint[:-1], identity[:-1])
                mask = torch.cat((identity[:-1] >= 0, torch.ones_like(targets[..., :ARENA_PLANES], dtype=torch.bool)), -1)
                all_states.append(state[:-1].reshape(-1, self.codec.state_dim))
                all_targets.append(targets.reshape(-1, targets.shape[-1]))
                all_constraints.append(constraint.reshape(-1, constraint.shape[-1]))
                all_masks.append(mask.reshape(-1, mask.shape[-1]))
        return torch.cat(all_states), torch.cat(all_targets), torch.cat(all_constraints), torch.cat(all_masks)

    def update(self) -> dict[str, float]:
        # Discount and bootstrap weight stay fixed for this update.
        schedule_step = self.optimizer_steps
        self.discount_lambda = self.cfg.discount_schedule.value(schedule_step, self.total_optimizer_steps, self.step_dt)
        self.bootstrap_weight = self.cfg.discount_schedule.bootstrap_weight(schedule_step, self.total_optimizer_steps)
        state, target, constraint, mask = self._training_data()
        losses = []
        for _ in range(self.cfg.gradient_steps):
            index = torch.randint(state.shape[0], (min(self.cfg.batch_size, state.shape[0]),), device=self.device)
            prediction = self.model(state[index])
            weight = mask[index].to(prediction.dtype)
            loss = ((prediction - target[index]).square() * weight).sum() / weight.sum().clamp_min(1.0)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm_clip)
            self.optimizer.step()
            self.optimizer_steps += 1
            losses.append(float(loss.item()))

        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.model.parameters()):
                target_param.lerp_(param, self.cfg.target_tau)
        metrics = {
            "loss": sum(losses) / len(losses),
            "grad_norm": float(grad_norm),
            "replay_samples": float(state.shape[0]),
            "gamma": self.gamma,
            "discount_lambda": self.discount_lambda,
            "bootstrap_weight": self.bootstrap_weight,
            "optimizer_steps": self.optimizer_steps,
            "rollout_done_fraction": float(self.replay[-1]["done"].float().mean()),
        }
        # In-sample diagnostics on the last minibatch (no extra sampling).
        with torch.no_grad():
            prediction = self.model(state[index])
            heads = prediction.shape[1]
            for name, columns in (("obstacle", slice(0, heads - ARENA_PLANES)), ("arena", slice(heads - ARENA_PLANES, heads))):
                keep = mask[index][:, columns]
                predicted = prediction[:, columns][keep]
                expected = target[index][:, columns][keep]
                metrics.update({
                    f"train/{name}/mse": float((predicted - expected).square().mean()),
                    f"train/{name}/target_positive_fraction": float((expected > 0).float().mean()),
                    f"train/{name}/predicted_safe_fraction": float((predicted < 0).float().mean()),
                    f"train/{name}/below_geometry_fraction": float((predicted < constraint[index][:, columns][keep]).float().mean()),
                })
        return metrics

    def checkpoint(self, update: int, name: str | None = None) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / (name or f"certificate_{update}.pt")
        torch.save(
            {
                "format_version": 3,
                "update": int(update),
                "frames": self.frames,
                "optimizer_steps": self.optimizer_steps,
                "discount_lambda": self.discount_lambda,
                "model": self.model.state_dict(),
                "target_model": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_cfg": self.model.config_dict(),
                "train_cfg": self.cfg.to_dict(),
                "state_spec": self.codec.metadata(),
                "command_scale": self.command_scale,
                "step_dt": self.step_dt,
                "run_info": self.run_info,
            },
            path,
        )
        return path

    def run(self, *, timesteps: int | None = None) -> Path:
        remaining = self.cfg.updates * self.cfg.rollout_steps if timesteps is None else timesteps
        if remaining < 1:
            raise ValueError("PNCBF timesteps must be positive")
        updates = (remaining + self.cfg.rollout_steps - 1) // self.cfg.rollout_steps
        self.cfg.updates = updates
        self.total_optimizer_steps = updates * self.cfg.gradient_steps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with (self.output_dir / "metrics.jsonl").open("a", buffering=1) as stream:
            for update in range(1, updates + 1):
                steps = min(self.cfg.rollout_steps, remaining)
                self.collect(steps)
                remaining -= steps
                metrics = self.update()
                checkpoint = None
                if update % self.cfg.checkpoint_interval == 0:
                    checkpoint = str(self.checkpoint(update))
                if update == updates:
                    checkpoint = str(self.checkpoint(update, name="certificate_final.pt"))
                record = {"update": update, "frames": self.frames, "elapsed_seconds": time.monotonic() - started,
                          "task": self.run_info.get("task"), "seed": self.run_info.get("certificate_seed"),
                          "checkpoint": checkpoint, **metrics}
                stream.write(json.dumps(record) + "\n")
                print(f"[PNCBF] update={update}/{updates} frames={self.frames} loss={metrics['loss']:.6f} "
                      f"samples={int(metrics['replay_samples'])} discount_lambda={self.discount_lambda:.6f}", flush=True)
        return self.output_dir / "checkpoints" / "certificate_final.pt"
