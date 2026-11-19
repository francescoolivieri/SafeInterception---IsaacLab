"""DG-PPO agent on skrl's Agent lifecycle: split rollout collection and the update.

The first half of the environments acts deterministically (the policy mode) and trains
the constraint critic V_h on the policy's own trajectories; the second half samples
actions and provides the PPO data, the task critic V_l and the CBF advantage.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import torch
from skrl.agents.torch import Agent
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from target_interception.nn.graph import graph_data_select
from target_interception.nn.networks import build_dgppo_critics, build_dgppo_policy

from .algo_utils import (
    apply_optimizer_update,
    compute_cbf_advantages,
    compute_policy_surrogate,
    constraint_targets,
    reward_targets,
    scheduled_cbf_scale,
    value_l2_loss,
)
from .config import DGPPOAgentCfg
from .distributed import broadcast_modules, is_distributed

# Episode outcomes logged as rates over the last finished episodes.
OUTCOMES = {
    "capture_rate": ("capture",),
    "collision_rate": ("collision",),
    "timeout_rate": ("timeout",),
    "out_of_bounds_rate": ("boundary", "low_altitude"),
}


class Rollout:
    """One rollout split, stored [T, E, ...], plus the observation after the last step."""

    def __init__(self, T: int, E: int, obs_dim: int, action_dim: int, device) -> None:
        self.observations = torch.zeros(T, E, obs_dim, device=device)
        self.actions = torch.zeros(T, E, action_dim, device=device)
        self.log_prob = torch.zeros(T, E, device=device)
        self.rewards = torch.zeros(T, E, device=device)
        self.costs = torch.zeros(T, E, 1, device=device)
        self.done = torch.zeros(T, E, dtype=torch.bool, device=device)
        self.final_observations = torch.zeros(E, obs_dim, device=device)

    def batch_major(self) -> dict[str, torch.Tensor]:
        """[B, T, ...] views; l = -reward is the cost of the task critic."""
        return {
            "observations": self.observations.transpose(0, 1),
            "actions": self.actions.transpose(0, 1),
            "logp": self.log_prob.transpose(0, 1),
            "l": -self.rewards.transpose(0, 1),
            "h": self.costs.transpose(0, 1),
            "done": self.done.transpose(0, 1),
        }


class DGPPOAgent(Agent):
    monitor = None

    def __init__(self, *, graph, preprocessor, policy, Vl, Vh, env, cfg: DGPPOAgentCfg, device) -> None:
        base_env = env.unwrapped
        super().__init__(
            models={"policy": policy, "Vl": Vl, "Vh": Vh},
            memory=None,
            observation_space=base_env.observation_space,
            state_space=base_env.state_space,
            action_space=base_env.action_space,
            device=device,
            cfg=cfg,
        )
        self.policy, self.Vl, self.Vh = policy.to(device), Vl.to(device), Vh.to(device)
        self.graph, self.preprocessor = graph, preprocessor
        self.env, self.base_env = env, base_env
        self._rank = torch.distributed.get_rank() if is_distributed() else 0
        broadcast_modules((self.policy, self.Vl, self.Vh))
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)
        self._vl_opt = torch.optim.Adam(self.Vl.parameters(), lr=cfg.lr_vl)
        self._vh_opt = torch.optim.Adam(self.Vh.parameters(), lr=cfg.lr_vh)
        self.checkpoint_modules = {
            "policy": self.policy,
            "Vl": self.Vl,
            "Vh": self.Vh,
            "policy_optimizer": self._policy_opt,
            "Vl_optimizer": self._vl_opt,
            "Vh_optimizer": self._vh_opt,
            "observation_preprocessor": self.preprocessor,
        }
        n = env.num_envs
        if n < 2 or n % 2:
            raise ValueError(f"DG-PPO splits the environments in two halves; num_envs must be even, got {n}")
        self._splits = {
            "det": torch.arange(0, n // 2, device=self.device),
            "stc": torch.arange(n // 2, n, device=self.device),
        }
        self._rollout: dict[str, Rollout] = {}
        self._cursor = 0
        self._steps = 0
        self._cache = None
        self._outcomes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._recent_outcomes: deque[int] = deque(maxlen=200)

    def init(self, *, trainer_cfg: Any | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        obs_dim = int(self.observation_space.shape[-1])
        # base_env spaces are batched, (num_envs, action_dim): the last axis is the action.
        action_dim = int(self.action_space.shape[-1])
        for split, ids in self._splits.items():
            self._rollout[split] = Rollout(self.cfg.rollouts, len(ids), obs_dim, action_dim, self.device)

    # Collection.

    def act(self, observations: torch.Tensor, states, *, timestep: int, timesteps: int):
        with torch.no_grad():
            graph = self.preprocessor(self.graph(observations))
            action, log_prob, mode = self.policy.act(graph, deterministic=not self.training)
        det = self._splits["det"]
        action[det] = mode[det]
        log_prob[det] = 0.0  # unused
        if self.training:
            # The simulator still holds the state behind these observations.
            constraint = self.base_env.dgppo_constraint_values().detach().clone()
            self._cache = (observations.detach(), constraint, log_prob)
        return action, {"log_prob": log_prob, "mean_action": mode}

    def record_transition(self, *, observations, states, actions, rewards, next_observations, next_states,
                          terminated, truncated, infos, timestep: int, timesteps: int) -> None:
        if self.training:
            cached_observations, constraint, log_prob = self._cache
            # DG-PPO's reward and shaped cost of the state-t transition.
            step_rewards, costs = self.base_env.compute_dgppo_signals(
                observations=observations, actions=actions, constraint_values=constraint
            )
            n = self.env.num_envs
            step_rewards = step_rewards.to(self.device, torch.float32).reshape(n)
            done = (terminated | truncated).to(self.device, torch.bool).reshape(n)
            t = self._cursor
            for split, ids in self._splits.items():
                rollout = self._rollout[split]
                rollout.observations[t] = cached_observations[ids]
                rollout.actions[t] = actions.reshape(n, -1)[ids]
                rollout.log_prob[t] = log_prob[ids]
                rollout.rewards[t] = step_rewards[ids]
                rollout.costs[t] = costs.reshape(n, -1)[ids]
                rollout.done[t] = done[ids]
                if t + 1 == self.cfg.rollouts:
                    rollout.final_observations[:] = next_observations[ids]
            self._cursor += 1
            if self._rank == 0:
                self._outcomes.append((self.base_env.get_last_episode_status().detach().reshape(-1), done))
            rewards = step_rewards.reshape_as(rewards)
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states, terminated=terminated,
            truncated=truncated, infos=infos, timestep=timestep, timesteps=timesteps,
        )

        if self.monitor is not None:
            self.monitor.record(self, rewards=rewards, actions=actions, terminated=terminated,
                                truncated=truncated, timestep=timestep, timesteps=timesteps)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            self._steps += 1
            if self._steps % self.cfg.rollouts == 0:
                start = time.perf_counter()
                self.update(timestep=timestep, timesteps=timesteps)
                self.track_data("Stats / Algorithm update time (ms)", (time.perf_counter() - start) * 1000.0)
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        if self._rank == 0:
            super().write_checkpoint(timestep=timestep, timesteps=timesteps)

    # Update.

    def cbf_scale(self, timestep: int, timesteps: int) -> float:
        """CBF weight: held during the curriculum, then raised stage by stage."""
        cfg = self.base_env.cfg
        start = int(cfg.curriculum_total_steps) if int(cfg.curriculum_total_steps) > 0 else int(cfg.total_timesteps)
        return scheduled_cbf_scale(
            cbf_weight=self.cfg.cbf_weight, timestep=timestep, timesteps=timesteps, schedule_start_step=start,
            fractions=self.cfg.cbf_schedule_fractions, multipliers=self.cfg.cbf_schedule_multipliers,
        )

    def _graph_batch(self, observations: torch.Tensor):
        """Raw (unnormalized) graphs of [B, T, obs] observations, B-major."""
        return self.graph(observations.reshape(-1, observations.shape[-1]))

    def update(self, *, timestep: int, timesteps: int) -> None:
        stc, det = self._rollout["stc"].batch_major(), self._rollout["det"].batch_major()
        raw_graph, raw_det_graph = self._graph_batch(stc["observations"]), self._graph_batch(det["observations"])
        graph, det_graph = self.preprocessor(raw_graph), self.preprocessor(raw_det_graph)
        final_graph = self.preprocessor(self.graph(self._rollout["stc"].final_observations))
        det_final_graph = self.preprocessor(self.graph(self._rollout["det"].final_observations))
        B, T = stc["l"].shape
        cbf_scale = self.cbf_scale(timestep, timesteps)

        for _ in range(self.cfg.learning_epochs):
            with torch.no_grad():
                bT_vl = self.Vl(graph).reshape(B, T)
                bTp1_vl = torch.cat((bT_vl, self.Vl(final_graph)), dim=1)
                bTh_vh = self.Vh(graph).reshape(B, T, -1)
                bTp1h_vh = torch.cat((bTh_vh, self.Vh(final_graph)[:, None]), dim=1)
                bTp1h_vh_det = torch.cat((self.Vh(det_graph).reshape(B, T, -1), self.Vh(det_final_graph)[:, None]), 1)
                bT_ql = reward_targets(stc["l"], bTp1_vl, stc["done"],
                                       gamma=self.cfg.discount_factor, gae_lambda=self.cfg.gae_lambda)
                bTh_qh_det = constraint_targets(det["h"], bTp1h_vh_det, det["done"],
                                                gamma=self.cfg.discount_factor, gae_lambda=self.cfg.gae_lambda)
                adv = compute_cbf_advantages(bT_ql, bT_vl, bTh_vh, bTp1h_vh, stc["done"], alpha=self.cfg.alpha,
                                             cbf_eps=self.cfg.cbf_eps, cbf_scale=cbf_scale, dt=self.base_env._step_dt)
            for env_ids in torch.tensor_split(torch.randperm(B, device=self.device), self.cfg.mini_batches):
                if env_ids.numel():
                    info = self._update_minibatch(env_ids, stc, graph, det_graph, adv["bT_A"], bT_ql, bTh_qh_det)

        # Advance the feature statistics only after the update used the frozen ones.
        if self._rank == 0:
            self.preprocessor.update(raw_graph)
            self.preprocessor.update(raw_det_graph)
        broadcast_modules((self.preprocessor,))
        self._cursor = 0

        if self._rank == 0:
            info.update({
                "eval/safe_data": adv["bT_is_safe"].float().mean(),
                "DGPPO/cbf/active_frac": (adv["bT_cbf_penalty"] > 0).float().mean(),
                "DGPPO/cbf_schedule/scale": cbf_scale,
                "DGPPO/rollout/reward_mean": (-stc["l"]).mean(),
                "DGPPO/rollout/unsafe_agent_step_rate": (stc["h"].max(dim=-1).values >= 0).float().mean(),
                **self._outcome_rates(),
            })
            values = {name: float(value) for name, value in info.items()}
            for name, value in values.items():
                self.track_data(name, value)
            if self.cfg.experiment.wandb:
                import wandb

                wandb.log(values, step=timestep + 1)

    def _update_minibatch(self, env_ids, stc, graph, det_graph, bT_A, bT_ql, bTh_qh_det) -> dict:
        T = stc["l"].shape[1]
        rows = (env_ids[:, None] * T + torch.arange(T, device=env_ids.device)).reshape(-1)
        batch, det_batch = graph_data_select(graph, rows), graph_data_select(det_graph, rows)
        actions = stc["actions"][env_ids]

        log_prob, entropy = self.policy.evaluate(batch, actions.reshape(-1, actions.shape[-1]))
        ratio = torch.exp(log_prob.reshape(len(env_ids), T) - stc["logp"][env_ids].detach())
        surrogate = compute_policy_surrogate(ratio, bT_A[env_ids], self.cfg.ratio_clip)
        entropy_mean = entropy.mean()
        loss_policy = surrogate["loss_policy"] - self.cfg.entropy_loss_scale * entropy_mean
        policy_grad_norm = apply_optimizer_update(optimizer=self._policy_opt, loss=loss_policy,
                                                  parameters=self.policy.parameters(), grad_clip=self.cfg.grad_norm_clip)

        loss_vl = value_l2_loss(self.Vl(batch).reshape(len(env_ids), T), bT_ql[env_ids], scale=self.cfg.vl_loss_scale)
        loss_vh = value_l2_loss(self.Vh(det_batch).reshape(len(env_ids), T, -1), bTh_qh_det[env_ids],
                                scale=self.cfg.vh_loss_scale)
        vl_grad_norm = apply_optimizer_update(optimizer=self._vl_opt, loss=loss_vl, parameters=self.Vl.parameters(),
                                              grad_clip=self.cfg.grad_norm_clip)
        vh_grad_norm = apply_optimizer_update(optimizer=self._vh_opt, loss=loss_vh, parameters=self.Vh.parameters(),
                                              grad_clip=self.cfg.grad_norm_clip)
        return {
            "Vl/loss": loss_vl.detach(),
            "Vl/grad_norm": vl_grad_norm,
            "Vh/loss_Vh": loss_vh.detach(),
            "Vh/grad_Vh_norm": vh_grad_norm,
            "policy/loss": loss_policy.detach(),
            "policy/clip_frac": surrogate["clip_frac"],
            "policy/entropy": entropy_mean.detach(),
            "policy/grad_norm": policy_grad_norm,
        }

    def _outcome_rates(self) -> dict[str, float]:
        """Outcome rates over the last 200 finished episodes (one host sync per update)."""
        if self._outcomes:
            status = torch.stack([s for s, _ in self._outcomes])
            done = torch.stack([d for _, d in self._outcomes])
            self._recent_outcomes.extend(status[done].cpu().tolist())
            self._outcomes.clear()
        if not self._recent_outcomes:
            return {}
        codes = {name: code for code, name in self.base_env.EPISODE_STATUS_MAP.items()}
        recent = list(self._recent_outcomes)
        return {
            f"DGPPO/recent_episode_outcomes/{metric}": sum(o in {codes.get(n) for n in names} for o in recent) / len(recent)
            for metric, names in OUTCOMES.items()
        }


class DGPPORunner:
    def __init__(self, env: Any, cfg: dict) -> None:
        self.env = env
        agent_cfg = DGPPOAgentCfg.from_dict(cfg.get("agent", cfg))
        set_seed(cfg.get("seed", agent_cfg.seed))
        device = torch.device(env.device)
        graph, preprocessor, policy = build_dgppo_policy(
            agent_cfg, env.unwrapped.obs_layout, action_dim=int(env.action_space.shape[0]), device=device
        )
        Vl, Vh = build_dgppo_critics(agent_cfg, context_dim=preprocessor.ego_context_dim, device=device)
        self.agent = DGPPOAgent(graph=graph, preprocessor=preprocessor, policy=policy, Vl=Vl, Vh=Vh, env=env,
                                cfg=agent_cfg, device=device)
        self.trainer = SequentialTrainer(env=env, agents=self.agent, cfg=cfg.get("trainer", {}))

    def run(self) -> None:
        self.trainer.train()
