"""Fixed timestep curriculum of the penalised graph PPO (M4) in the large arena.

Eight stages with one pool bank each, switched at fixed steps. No performance gates,
replay or phase blending.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass

from .large_arena import LARGE_ARENA_FRAME_ID

PROFILE = "penalty_staged"
POOL_NAME = "penalty_large_staged"
REFERENCE_STEPS = 400_000


@dataclass(frozen=True)
class Bank:
    end: int                                  # step where the stage ends, at REFERENCE_STEPS
    counts: tuple[tuple[int, int], ...]       # static + moving cylinders, equal quotas
    speed: tuple[float, float] | None         # evader speed range [m/s], None = stationary
    distance: tuple[float, float | None] = (1.2, None)  # start distance range [m]
    rows: int = 5000


BANKS = (
    Bank(10_000, ((0, 0),), None, (0.6, 1.0), 1000),
    Bank(26_000, ((0, 0),), None, (1.0, None), 1000),
    Bank(64_000, ((0, 0),), (0.4, 0.8), rows=3000),
    Bank(112_000, ((1, 0), (2, 0), (3, 0)), (0.4, 1.2)),
    Bank(160_000, ((1, 1), (2, 1), (3, 1)), (0.4, 1.2)),
    Bank(220_000, ((2, 2), (4, 2), (6, 2)), (0.4, 2.2)),
    Bank(300_000, ((6, 3), (6, 4), (8, 4), (8, 5)), (0.4, 2.2), rows=10000),
    Bank(400_000, ((10, 5), (11, 5), (11, 4)), (0.4, 2.2), rows=25000),
)
# Stage 5 brings in one slow moving cylinder on a short rail.
SLOW_RAIL_STAGE = 5


def enabled(cfg) -> bool:
    return getattr(cfg.scenario, "curriculum_profile", None) == PROFILE


def bank_at_step(step: int, total: int) -> int:
    """1-based bank, with exact start-inclusive gates at any training budget."""
    total = max(1, int(total))
    # Integer products avoid floating point errors at an exact phase boundary.
    return min(len(BANKS), 1 + bisect_right(
        [bank.end * total for bank in BANKS], max(0, int(step)) * REFERENCE_STEPS
    ))


def apply(cfg) -> None:
    if cfg.world_frame_id != LARGE_ARENA_FRAME_ID or cfg.max_static_obstacles < 11 or cfg.max_dynamic_obstacles < 5:
        raise ValueError(f"{PROFILE} requires the large clutter arena task (11 static / 5 dynamic slots)")
    cfg.scenario.curriculum_profile = PROFILE
    previous = (0,) + tuple(bank.end for bank in BANKS[:-1])
    cfg.curriculum_phase_fractions = tuple(
        (bank.end - start) / REFERENCE_STEPS for bank, start in zip(BANKS, previous)
    )
    cfg.curriculum_blend_fraction = 0.0
    cfg.scenario.phase_obstacle_counts = tuple(bank.counts for bank in BANKS)
    cfg.scenario.pool_name = POOL_NAME
    cfg.scenario.pool_size_by_phase = tuple((i, bank.rows) for i, bank in enumerate(BANKS, 1))
    # Offline and live geometry must agree. The State training task uses zero margin.
    cfg.arena_margin = 0.0
    # Moving stages must retain their authored motion distribution.
    cfg.scenario.augment_evader_freeze_probability = 0.0


def generation_cfg(cfg, phase: int):
    """Per-bank settings, shared by generation and the independent pool audit."""
    if not enabled(cfg):
        return cfg
    bank = BANKS[phase - 1]
    out = deepcopy(cfg)
    scn = out.scenario
    scn.interceptor_min_target_distance = bank.distance[0]
    scn.start_distance_bins = ((bank.distance[0], bank.distance[1]),) if bank.distance[1] else ()
    if bank.speed is not None:
        scn.target_speed_range = bank.speed
    if phase == SLOW_RAIL_STAGE:
        scn.dynamic_speed_range = (0.3, 0.5)
        scn.dynamic_motion_patterns = ("rail",)
        scn.dynamic_rail_length_range = (0.8, 1.1)
        scn.dynamic_rail_length_bins = ((0.8, 1.1),)
        scn.dynamic_rail_length_weights = (1.0,)
    return out


def constrain_reset(env, rows: dict, phase: int, original_spawn) -> None:
    """Keep the respawn augmentation within the start rules of the chosen bank.

    Only the stationary obstacle-free stages change the altitude; later stages keep
    the independently sampled 3D spawns and the original obstacle clearance.
    """
    import torch
    from target_interception.task import target

    bank = BANKS[phase - 1]
    spawn = rows["interceptor_start"]
    wp = rows["target_waypoints"]
    offset = wp[:, 0, :2] - spawn[:, :2]
    distance = torch.linalg.vector_norm(offset, dim=-1)
    valid = distance >= bank.distance[0]
    if bank.distance[1] is not None:
        valid &= distance < bank.distance[1]
    if bank.speed is not None:
        dt = (env._path_waypoint_steps[1:] - env._path_waypoint_steps[:-1]) * env._step_dt
        velocity = (wp[:, 1:] - wp[:, :-1]) / dt[None, :, None].clamp_min(1e-6)
        moving = torch.linalg.vector_norm(velocity, dim=-1) > 0.05
        first = moving.to(torch.int64).argmax(dim=1)
        v = velocity[torch.arange(wp.shape[0], device=wp.device), first, :2]
        alignment = (offset * v).sum(-1) / (
            distance * torch.linalg.vector_norm(v, dim=-1)
        ).clamp_min(1e-6)
        valid &= ~((distance < target.min_start_distance_for_toward_motion(env)) & (alignment < -0.3))
    spawn[:] = torch.where(valid[:, None], spawn, original_spawn)

    if phase <= 2:
        lo, hi = target.altitude_bounds(env)
        total = max(1, int(env.cfg.curriculum_total_steps))
        step = float(env.common_step_counter) * REFERENCE_STEPS / total
        # Stage 2 widens the altitude offset from 0.2 m to the full range over its length.
        start, end = BANKS[0].end, BANKS[1].end
        fraction = min(1.0, max(0.0, (step - start) / (end - start))) if phase == 2 else 0.0
        max_delta = 0.2 + fraction * (hi - lo - 0.2)
        target_z = wp[:, 0, 2]
        lower = torch.clamp(target_z - max_delta, min=lo, max=hi)
        upper = torch.clamp(target_z + max_delta, min=lo, max=hi)
        spawn[:, 2] = lower + torch.rand_like(target_z) * (upper - lower)
