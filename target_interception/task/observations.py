"""Observation layout shared by the environment, policies and safety filters.

Layouts are saved with each run so loaders can reconstruct the policy inputs."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

EGO_DIM = 18
TARGET_DIM = 6
# Padding for unused obstacle slots. It must lie outside the DG-PPO graph edge radius so
# padded slots never become obstacle nodes.
_OBSTACLE_PAD_OFFSET = -10.0


@dataclass(frozen=True)
class ObsLayout:
    """Slice map of the flat observation vector."""

    ego: slice
    target: slice
    extras: dict[str, slice] = field(default_factory=dict)
    # Arena box (env-local, shrunk by cfg.arena_margin). Graph builders place the six
    # boundary nodes on its planes; None disables them.
    arena_min: tuple[float, float, float] | None = None
    arena_max: tuple[float, float, float] | None = None

    @property
    def obs_dim(self) -> int:
        return max([self.ego.stop, self.target.stop] + [s.stop for s in self.extras.values()])


def _safe_arena_box(cfg) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    arena_min = getattr(cfg, "arena_min", None)
    arena_max = getattr(cfg, "arena_max", None)
    if arena_min is None or arena_max is None:
        return None, None
    margin = float(getattr(cfg, "arena_margin", 0.0))
    return tuple(float(v) + margin for v in arena_min), tuple(float(v) - margin for v in arena_max)


def obstacle_observation_capacity(cfg) -> int:
    capacity = int(cfg.obstacle_observation_capacity)
    if capacity < 1:
        raise ValueError(f"obstacle_observation_capacity must be positive, got {capacity}")
    return capacity


def build_obs_layout(cfg) -> ObsLayout:
    ego = slice(0, EGO_DIM)
    target = slice(ego.stop, ego.stop + TARGET_DIM)
    base = target.stop
    extras: dict[str, slice] = {}
    mode = cfg.obstacle_observation_mode
    if mode == "pillars":
        capacity = obstacle_observation_capacity(cfg)
        extras["obstacles"] = slice(base, base + capacity * 3)
        extras["obstacle_vel"] = slice(base + capacity * 3, base + capacity * 5)
        base += capacity * 5
    elif mode != "none":
        raise ValueError(f"unknown obstacle_observation_mode {mode!r}")
    extras["previous_action"] = slice(base, base + int(getattr(cfg, "action_space", 4)))
    arena_min, arena_max = _safe_arena_box(cfg)
    return ObsLayout(ego=ego, target=target, extras=extras, arena_min=arena_min, arena_max=arena_max)


def compute_obs_dim(cfg) -> int:
    return build_obs_layout(cfg).obs_dim


def build_observation(env) -> dict:
    from isaaclab.utils import math as math_utils  # lazy: keeps ObsLayout importable without Isaac

    # Measured states (see target_interception.sim.measurement_model).
    agent_pos = env._observed_position_local_cache
    agent_vel = env._observed_velocity_world_cache
    rotation = math_utils.matrix_from_quat(env._observed_quat_cache).reshape(env.num_envs, 9)
    ego = torch.cat((agent_pos, agent_vel, rotation, env._observed_body_rate_cache), dim=-1)

    target_offset = env._observed_target_position_local_cache - agent_pos
    closing_velocity = env._observed_target_velocity_world_cache - agent_vel

    if env.cfg.obstacle_observation_mode == "pillars":
        obstacles = all_obstacle_observation(env, agent_pos, agent_vel)
    else:
        obstacles = agent_pos.new_empty(env.num_envs, 0)
    parts = [ego, target_offset, closing_velocity, obstacles, env._prev_actions]
    return {"policy": torch.cat(parts, dim=-1)}


def all_obstacle_observation(env, agent_pos: torch.Tensor, agent_vel: torch.Tensor) -> torch.Tensor:
    """Return distance-sorted obstacle offsets and relative XY velocities.

    Pad unused slots and reject allocations above the observation capacity.
    Record slot identities for the per-constraint PNCBF."""
    E = env.num_envs
    capacity = obstacle_observation_capacity(env.cfg)
    pad = agent_pos.new_full((E, capacity, 3), _OBSTACLE_PAD_OFFSET)
    rel_pos = pad.clone()
    vel = agent_pos.new_zeros(E, capacity, 2)
    # Which allocated obstacle (static then dynamic, the order of the constraint
    # geometry) each slot shows; -1 for padding. Read by the per-constraint PNCBF.
    identity = torch.full((E, capacity), -1, dtype=torch.long, device=agent_pos.device)

    positions, velocities, active = [], [], []
    if env.cfg.enable_pillars and env._max_static_obstacles > 0:
        positions.append(env._observed_static_obstacle_positions_cache)
        velocities.append(env._observed_static_obstacle_velocities_cache[..., :2])
        active.append(env._static_obstacle_active)
    if env._max_dynamic_obstacles > 0:
        positions.append(env._observed_dynamic_obstacle_positions_cache)
        velocities.append(env._observed_dynamic_obstacle_velocities_cache[..., :2])
        active.append(env._dynamic_obstacle_active)

    if positions:
        pos_all = torch.cat(positions, dim=1) - agent_pos[:, None, :]
        vel_all = torch.cat(velocities, dim=1)
        active_all = torch.cat(active, dim=1)
        allocated = int(pos_all.shape[1])
        if allocated > capacity:
            raise ValueError(
                f"environment allocated {allocated} obstacle slots but the observation capacity is {capacity}; "
                "increase obstacle_observation_capacity so no obstacle can be dropped"
            )
        distance = torch.linalg.vector_norm(pos_all[..., :2], dim=-1)
        distance = torch.where(active_all, distance, torch.full_like(distance, float("inf")))
        order = torch.argsort(distance, dim=1, stable=True)
        active_all = torch.gather(active_all, 1, order)
        pos_all = torch.gather(pos_all, 1, order.unsqueeze(-1).expand(-1, -1, 3))
        vel_all = torch.gather(vel_all, 1, order.unsqueeze(-1).expand(-1, -1, 2))
        rel_pos[:, :allocated] = torch.where(
            active_all.unsqueeze(-1), pos_all, pad[:, :allocated]
        )
        vel[:, :allocated] = torch.where(
            active_all.unsqueeze(-1), vel_all, torch.zeros_like(vel_all)
        )
        identity[:, :allocated] = torch.where(active_all, order, torch.full_like(order, -1))
    env._obstacle_observation_identity = identity

    rel_vel = vel - agent_vel[:, None, :2]
    return torch.cat((rel_pos.reshape(E, -1), rel_vel.reshape(E, -1)), dim=-1)
