"""Simulator-free configuration used by scenario and benchmark generators."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ScenarioGenerationCfg
from .large_arena import (
    LARGE_ARENA_FRAME_ID, LARGE_ARENA_MAX_M, LARGE_ARENA_MIN_M, NOMINAL_POOL_SIZES, apply_large_arena,
)
from target_interception.world_frame import ARENA_MAX_M, ARENA_MIN_M, WORLD_FRAME_ID


@dataclass
class SimulationTiming:
    dt: float = 1.0 / 500.0


@dataclass
class OfflineTargetInterceptionCfg:
    """Scenario geometry and timing, checked against the runtime config."""

    scenario: ScenarioGenerationCfg = field(default_factory=ScenarioGenerationCfg)
    sim: SimulationTiming = field(default_factory=SimulationTiming)
    curriculum_phase_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    world_frame_id: str = WORLD_FRAME_ID
    arena_min: tuple[float, float, float] = ARENA_MIN_M
    arena_max: tuple[float, float, float] = ARENA_MAX_M
    # Same as the runtime task: generated geometry is authored against the real arena.
    arena_margin: float = 0.0
    episode_length_s: float = 10.0
    decimation: int = 10
    enable_pillars: bool = True
    max_static_obstacles: int = 6
    max_dynamic_obstacles: int = 3
    pillar_radius: float = 0.2
    pillar_height: float = 1.9
    drone_collision_radius: float = 0.09
    dynamic_obstacle_height: float = 1.9
    path_waypoint_dt: float = 0.2
    # Knots are denser over the launch, which is shorter than two coarse knot intervals.
    path_launch_time: float = 0.4
    path_launch_waypoint_dt: float = 0.04


def offline_cfg_for_task(task: str) -> OfflineTargetInterceptionCfg:
    """Return scenario geometry for a registered interception task."""
    task = str(task)
    known = {
        "TargetInterception-State-v0",
        "TargetInterception-State-Large-v0",
        "TargetInterception-State-Penalty-Large-v0",
        "TargetInterception-Nominal-Large-v0",
    }
    if task not in known:
        raise ValueError(f"Unknown target-interception task {task!r}; expected one of {sorted(known)}")
    cfg = OfflineTargetInterceptionCfg()
    if task == "TargetInterception-Nominal-Large-v0":
        cfg.curriculum_phase_fractions = (0.2, 0.8)
        cfg.enable_pillars = False
        cfg.max_static_obstacles = 0
        cfg.max_dynamic_obstacles = 0
    if task in ("TargetInterception-State-Large-v0", "TargetInterception-State-Penalty-Large-v0"):
        apply_large_arena(cfg)
    if task == "TargetInterception-Nominal-Large-v0":
        # Large frame only: the nominal task keeps zero obstacles and its own phase counts.
        cfg.world_frame_id = LARGE_ARENA_FRAME_ID
        cfg.arena_min = LARGE_ARENA_MIN_M
        cfg.arena_max = LARGE_ARENA_MAX_M
        cfg.scenario.pool_size_by_phase = NOMINAL_POOL_SIZES
    return cfg
