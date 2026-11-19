"""Scenario sampling and pool settings.

The generator and environment share cfg.scenario. Scene dimensions and timing
stay on the environment config."""
from __future__ import annotations

from isaaclab.utils import configclass


# (static, moving) obstacle pairs allowed in curriculum phases 1..4: static target,
# light mixed, medium mixed, dense mixed. Phases past the end reuse the last entry.
DEFAULT_PHASE_OBSTACLE_COUNTS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((1, 1), (2, 1), (3, 1)),
    ((3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)),
    ((5, 2), (5, 3), (6, 2)),
)


@configclass
class ScenarioGenerationCfg:
    """Geometry and sampling parameters for offline interception scenarios."""

    # None keeps the default four-phase curriculum.
    curriculum_profile: str | None = None

    # --- Grid curriculum -------------------------------------------------------
    # Target-planning grid. Occupied cells use the exact target clearance radius
    # (no raster padding); continuous checks run on the fitted and executed route.
    grid_cell_size: float = 0.10
    grid_wall_margin: float = 0.25
    # Keep clutter off the walls, where it rarely affects the interception route.
    obstacle_wall_margin: float = 0.35
    pursuer_occupied_side: float = 0.5
    # Target route segments starting within this many seconds (at cruise speed)
    # avoid the interceptor spawn square; later segments may cross it.
    target_avoids_interceptor_start_s: float = 3.0
    evader_goal_min_distance: float = 1.3
    dynamic_rail_length_range: tuple[float, float] = (0.8, 1.8)
    # Quotas count accepted obstacles, not attempted rails.
    dynamic_rail_length_bins: tuple[tuple[float, float], ...] = (
        (0.8, 1.1),
        (1.1, 1.4),
        (1.4, 1.8),
    )
    dynamic_rail_length_weights: tuple[float, ...] = (0.30, 0.35, 0.35)
    # Reject stop-and-go/wobble trajectories whose realized motion covers only a
    # small part of the authored rail during the episode.
    dynamic_min_effective_span_fraction: float = 0.60
    obstacle_clearance: float = 0.05
    dynamic_speed_range: tuple[float, float] = (0.3, 0.8)
    # Motion patterns sampled (uniformly, per obstacle) when generating scenarios.
    # "rail": constant-speed shuttle between the rail endpoints.
    # "wobble": rail + sinusoidal offset perpendicular to it.
    # "stop_and_go": rail that alternates moving and pausing.
    dynamic_motion_patterns: tuple[str, ...] = ("rail", "wobble", "stop_and_go")
    # Peak obstacle speed is capped at ~1.0 m/s: the wobble's lateral speed peaks at
    # 2*pi*amplitude/period (here 0.63), on top of the 0.75 rail speed ->
    # sqrt(0.75^2 + 0.63^2) ~= 0.98. Keep that bound in mind when retuning these two.
    dynamic_wobble_amplitude: float = 0.2
    dynamic_wobble_period: float = 2.0
    # Stop-and-go move and pause durations (s), each uniform in this range per obstacle.
    # Authored benchmark rows use the midpoint.
    dynamic_stop_go_range: tuple[float, float] = (0.5, 2.5)
    # Ramp (s) between rest and rail speed, capped at a third of the move. Also rounds
    # the reversal at the rail ends, for every pattern.
    dynamic_stop_go_ramp: float = 0.3

    # --- Interceptor and target starts -----------------------------------------
    interceptor_wall_clearance: float = 0.22
    interceptor_min_target_distance: float = 1.2
    # The target may initially move toward the interceptor only beyond this
    # horizontal start distance. Closer starts are sampled away/transverse.
    target_min_start_distance_for_toward_motion: float = 2.2
    start_distance_bins: tuple[tuple[float, float], ...] = (
        (1.2, 2.2),
        (2.2, 3.4),
        (3.4, 6.0),
    )
    altitude_bin_count: int = 3

    # Whole-scenario retries before a quota row counts as infeasible.
    scenario_attempts: int = 5000
    # Target routes tried on one pillar layout before the layout is resampled.
    target_path_retries: int = 100
    # Re-rolls of a single rail against the fixed pillars and target route.
    dynamic_obstacle_retries: int = 128

    # --- Target motion ---------------------------------------------------------
    target_motion_modes: tuple[str, ...] = (
        "hover",
        "wander",
    )
    target_speed_range: tuple[float, float] = (0.4, 2.2)
    # Speed timing: start at rest, v <= target_max_speed and
    # (v^2 curvature)^2 + (dv/dt)^2 <= target_acceleration_limit^2.
    # Above the 2.2 top cruise: the cubic overshoots a little, so a cap equal to the
    # request fails the executed-speed gate (cap * 1.02).
    target_max_speed: float = 2.4
    target_acceleration_limit: float = 7.0
    # Gate on the executed cubic only; its peaks sit above the planning limit.
    target_max_acceleration: float = 9.9
    target_radius: float = 0.09
    target_wall_clearance: float = 0.28  # altitude bounds
    # Horizontal body-to-wall gap in world coordinates.
    target_wall_surface_clearance_xy: float = 0.10
    # Junction blend half-width (m), along the raw route.
    target_spline_junction_halfwidth: float = 0.65
    # Corner rounding of the authored fixed benchmark routes only.
    smooth_evader_spacing: float = 0.45
    smooth_evader_chaikin_passes: int = 2
    # A route join may turn by at most ~120 deg (alignment -0.5), with directions
    # measured over this distance on each side of the join.
    smooth_evader_goal_blend_distance: float = 0.4
    smooth_evader_goal_min_alignment: float = -0.5
    # Goals tried per route segment before backing up a segment.
    smooth_evader_goal_attempts: int = 24
    # Segments undone when the route gets stuck with no feasible continuation.
    smooth_evader_backtracks: int = 6

    # --- Train-time augmentation of pool rows (task/augment.py) ------------------
    # Mirror about the arena centre, per axis.
    augment_mirror_probability: float = 0.5
    # Respawn the interceptor anywhere inside the arena (pooled spawns are on the border).
    augment_pursuer_respawn_probability: float = 0.5
    augment_pursuer_respawn_candidates: int = 4
    # Move one random pillar, kept only if it still clears everything.
    augment_static_replace_probability: float = 0.3
    # Park the target at its start, unless a moving obstacle sweeps that point.
    augment_evader_freeze_probability: float = 0.01

    # --- Offline pool ----------------------------------------------------------
    # Scenarios per enabled curriculum phase.
    pool_size: int = 5000
    # Per-phase overrides ((phase, rows), ...); other phases use pool_size.
    pool_size_by_phase: tuple[tuple[int, int], ...] = ()
    pool_name: str | None = None
    # Generator seed, separate from the training seed.
    pool_seed: int = 1729
    # None -> data/scenario_pools.
    pool_dir: str | None = None

    # --- Obstacle counts per curriculum phase ----------------------------------
    # (static, moving) pairs per phase, each with an equal quota; pairs above the
    # max_*_obstacles caps are dropped. Speed quotas are in scenarios/coverage.py.
    phase_obstacle_counts: tuple[tuple[tuple[int, int], ...], ...] = DEFAULT_PHASE_OBSTACLE_COUNTS


def _configured_phase_obstacle_counts(cfg, phase: int) -> list[tuple[int, int]]:
    """(static, moving) pairs from cfg.scenario.phase_obstacle_counts for one phase."""
    table = getattr(cfg.scenario, "phase_obstacle_counts", None) or DEFAULT_PHASE_OBSTACLE_COUNTS
    phase = min(max(1, int(phase)), len(table))
    return [(int(static), int(dynamic)) for static, dynamic in table[phase - 1]]


def phase_obstacle_limits(cfg, phase: int) -> tuple[int, int]:
    """Largest static and moving counts a phase may request (sizes lazy obstacle spawning)."""
    choices = _configured_phase_obstacle_counts(cfg, phase)
    return max(static for static, _ in choices), max(dynamic for _, dynamic in choices)


def phase_obstacle_count_choices(env, phase: int) -> list[tuple[int, int]]:
    return [
        (static, dynamic)
        for static, dynamic in _configured_phase_obstacle_counts(env.cfg, phase)
        if 0 <= static <= env._max_static_obstacles and 0 <= dynamic <= env._max_dynamic_obstacles
    ]



