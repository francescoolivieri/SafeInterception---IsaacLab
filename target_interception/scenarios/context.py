"""Environment fields needed by the offline scenario sampler."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from . import reference_spline

if TYPE_CHECKING:
    from .generation_stats import GenerationStats


@dataclass
class ScenarioContext:
    """Minimal env-like object exposing what the scenario sampler reads."""

    cfg: object
    device: torch.device
    _arena_min_safe: torch.Tensor
    _arena_max_safe: torch.Tensor
    _step_dt: float
    _path_steps: int
    _path_waypoint_count: int
    _path_waypoint_steps: torch.Tensor
    _max_static_obstacles: int
    _max_dynamic_obstacles: int
    generation_stats: "GenerationStats | None" = None

    @classmethod
    def from_cfg(cls, cfg, device: torch.device | str = "cpu") -> "ScenarioContext":
        device = torch.device(device)

        # Arena (mirrors env __init__: arena_min/max shrunk by arena_margin).
        arena_min = torch.tensor(cfg.arena_min, device=device, dtype=torch.float32)
        arena_max = torch.tensor(cfg.arena_max, device=device, dtype=torch.float32)
        margin = float(cfg.arena_margin)
        arena_min_safe = arena_min + margin
        arena_max_safe = arena_max - margin

        # Path / waypoint layout (mirrors env __init__).
        step_dt = float(cfg.sim.dt) * int(cfg.decimation)
        max_episode_length = math.ceil(float(cfg.episode_length_s) / max(step_dt, 1e-9))
        path_steps = int(max_episode_length) + 1
        last_path_step = max(0, path_steps - 1)
        waypoint_steps = torch.as_tensor(reference_spline.knot_steps(cfg, step_dt, last_path_step), device=device, dtype=torch.long)
        waypoint_count = len(waypoint_steps)

        # Obstacle slot counts (mirrors env __init__).
        max_static = max(1, int(cfg.max_static_obstacles)) if cfg.enable_pillars else 0
        max_dynamic = max(0, int(cfg.max_dynamic_obstacles))

        return cls(
            cfg=cfg,
            device=device,
            _arena_min_safe=arena_min_safe,
            _arena_max_safe=arena_max_safe,
            _step_dt=step_dt,
            _path_steps=path_steps,
            _path_waypoint_count=waypoint_count,
            _path_waypoint_steps=waypoint_steps,
            _max_static_obstacles=max_static,
            _max_dynamic_obstacles=max_dynamic,
        )
