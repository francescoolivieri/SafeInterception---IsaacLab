"""Policies for evaluation: PPO and DG-PPO checkpoints and the M1 LQR controller.

All of them expose act(obs) (deterministic, batched) and reset(env_ids); actions are
normalised world velocity and yaw rate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Policy(Protocol):
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic batched eval action (num_envs, action_dim)."""
        ...

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """Reset any per-episode internal state for the given env ids (or all)."""
        ...


@dataclass
class PolicyContext:
    """Everything the loaders need from the live env, gathered once by evaluate.py."""

    env: Any  # skrl-wrapped env (spaces / num_envs / device)
    base_env: Any  # unwrapped DirectRLEnv (obs_layout, cfg)
    obs_layout: Any
    device: Any
    agent_cfg: dict | None = None
    command_scale: Any = None  # normalized velocity action -> [vx, vy, vz, yaw_rate]
    policy_obs_dim: int | None = None
    saved_obs_layout: dict | None = None


def nominal_observation_indices(saved: dict, live) -> list[int]:
    """Map an obstacle-blind checkpoint's fields into a wider live observation.

    A smaller obstacle-aware layout is not a nominal-policy projection.
    """
    extras = saved.get("extras", {})
    if set(extras) - {"previous_action"}:
        raise ValueError("Only obstacle-blind observation layouts can be projected")
    blocks = [(saved[name], getattr(live, name)) for name in ("ego", "target")]
    if "previous_action" in extras:
        action_slice = live.extras.get("previous_action")
        if action_slice is None:
            raise ValueError("Nominal policy requires a previous_action observation")
        blocks.append((extras["previous_action"], action_slice))

    indices = []
    for (start, stop), source in sorted(blocks, key=lambda block: block[0][0]):
        if start != len(indices) or stop <= start:
            raise ValueError("Saved nominal observation blocks must tile the input exactly")
        if stop - start != source.stop - source.start:
            raise ValueError("Nominal observation block width does not match the live layout")
        indices.extend(range(source.start, source.stop))
    if len(indices) != int(saved["obs_dim"]):
        raise ValueError("Saved nominal observation width does not match its named fields")
    return indices


def lqr_pursuit_from_cfg(cfg: dict, ctx: PolicyContext):
    """M1 nominal pursuit from the experiment yaml (policy.type: classical)."""
    from target_interception.algorithms.classical import LQRPursuit

    params = dict(cfg.get("params") or {})
    if ctx.command_scale is None:
        raise ValueError("LQR pursuit needs the velocity command interface")
    q, r = float(params.pop("q", 1.0)), float(params.pop("r", 1.0))
    if params:
        raise ValueError(f"unknown LQR parameters: {sorted(params)}")
    return LQRPursuit(ctx.obs_layout, ctx.command_scale, q=q, r=r, device=ctx.device)


class SkrlCheckpointPolicy:
    """skrl PPO agent loaded from a checkpoint; act returns the policy mean.

    The Runner is built as in train.py, so the model and its observation preprocessor
    match the checkpoint.
    """

    def __init__(self, checkpoint: str, ctx: PolicyContext) -> None:
        from skrl.utils.runner.torch import Runner

        import copy

        agent_cfg = copy.deepcopy(ctx.agent_cfg or {})
        # Disable logging/checkpointing for an eval-only runner.
        agent_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
        agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["wandb"] = False

        env_obs_dim = int(ctx.env.observation_space.shape[0])
        self._obs_dim = int(ctx.policy_obs_dim) if ctx.policy_obs_dim else env_obs_dim
        if self._obs_dim > env_obs_dim:
            raise ValueError("Policy observation width exceeds the live observation width")
        self._observation_indices = None
        runner_env = ctx.env
        if self._obs_dim < env_obs_dim:
            import gymnasium as gym
            import numpy as np

            if ctx.saved_obs_layout is None:
                raise ValueError("Nominal policy projection requires its saved observation layout")
            indices = nominal_observation_indices(ctx.saved_obs_layout, ctx.obs_layout)
            if len(indices) != self._obs_dim:
                raise ValueError("Policy observation width disagrees with its saved layout")
            self._observation_indices = torch.tensor(indices, dtype=torch.long, device=ctx.device)
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_dim,),
                dtype=np.float32,
            )
            runner_env = SimpleNamespace(
                observation_space=observation_space,
                action_space=ctx.env.action_space,
                state_space=observation_space,
                num_envs=int(ctx.env.num_envs),
                device=str(ctx.env.device),
            )
            print(f"[INFO] Nominal policy reads its saved {self._obs_dim}-D fields from the live layout.")

        if agent_cfg.get("networks"):
            from target_interception.algorithms.ppo_graph.runner import GraphPPORunner

            if runner_env is not ctx.env:
                raise ValueError("PPO_GRAPH checkpoints need their complete saved observation layout")
            runner = GraphPPORunner(runner_env, agent_cfg)
        else:
            runner = Runner(runner_env, agent_cfg)
        resume_path = resolve_checkpoint(checkpoint)
        runner.agent.load(resume_path)
        runner.agent.enable_models_training_mode(False)
        self._agent = runner.agent

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self._observation_indices is not None:
                obs = obs.index_select(-1, self._observation_indices)
            # skrl Agent.act(observations, states, *, timestep, timesteps) -> (actions, info).
            actions, info = self._agent.act(obs, None, timestep=0, timesteps=0)
            return info.get("mean_actions", actions)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        pass  # MLP PPO is stateless across steps


class DGPPOPolicy:
    """Load a DG-PPO policy network from a checkpoint and act deterministically.

    Rebuilds the graph policy with the same node/edge dims the runner derives from the
    ObsLayout, loads the policy sub-state-dict, and decodes each observation into a
    graph before taking the distribution mode.
    """

    def __init__(self, checkpoint: str, ctx: PolicyContext) -> None:
        from target_interception.algorithms.dgppo.config import DGPPOAgentCfg
        from target_interception.nn.networks import build_dgppo_policy

        self.device = torch.device(ctx.device)
        cfg = DGPPOAgentCfg.from_dict(dict(ctx.agent_cfg or {}).get("agent", ctx.agent_cfg or {}))
        self.graph, self.preprocessor, self.policy = build_dgppo_policy(
            cfg, ctx.obs_layout, action_dim=int(ctx.env.action_space.shape[0]), device=self.device
        )
        state = torch.load(resolve_checkpoint(checkpoint), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.preprocessor.load_state_dict(state["observation_preprocessor"])
        # build_dgppo_policy only records the device; the parameters still have to move.
        self.policy.to(self.device).eval()

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            graph = self.preprocessor(self.graph(obs.to(self.device)))
            return self.policy.act(graph, deterministic=True)[0]

    def reset(self, env_ids: torch.Tensor | None) -> None:
        pass


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint reference to a local path.

    Evaluation needs the params directory saved next to the checkpoint, so only local
    run checkpoints are accepted.
    """
    if os.path.exists(checkpoint):
        return os.path.abspath(checkpoint)
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def build_policy(policy_cfg: dict, ctx: PolicyContext) -> Policy:
    """Dispatch on policy.type to the matching loader."""
    ptype = policy_cfg.get("type")
    if ptype == "skrl_checkpoint":
        return SkrlCheckpointPolicy(policy_cfg["checkpoint"], ctx)
    if ptype == "dgppo_checkpoint":
        return DGPPOPolicy(policy_cfg["checkpoint"], ctx)
    if ptype == "classical":
        return lqr_pursuit_from_cfg(policy_cfg, ctx)
    raise ValueError(
        f"unknown policy.type '{ptype}' (use 'classical', 'skrl_checkpoint', or 'dgppo_checkpoint')."
    )


class FilteredPolicy:
    """a = filter(o, nominal(o)), e.g. LQRPursuit + CBFQPFilter for M1."""

    def __init__(self, nominal, safety_filter) -> None:
        self.nominal = nominal
        self.filter = safety_filter
        self.last_filter_info: dict = {}

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        # The env clamps actions to [-1, 1]; filter the command the plant would execute,
        # not an unbounded policy mean (the QP's box would otherwise do the clipping).
        action = self.filter(obs, self.nominal.act(obs).clamp(-1.0, 1.0))
        self.last_filter_info = self.filter.info()
        return action

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self.nominal.reset(env_ids)
