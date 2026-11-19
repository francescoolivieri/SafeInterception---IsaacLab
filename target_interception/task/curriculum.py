"""Curriculum scheduling and pooled resets."""
from __future__ import annotations

import torch


def progress(env) -> float:
    total = int(env.cfg.curriculum_total_steps)
    if total <= 0:
        total = int(getattr(env.cfg, "total_timesteps", 1))
    return min(1.0, max(0.0, float(env.common_step_counter) / max(1, total)))


def _phase_schedule(env) -> tuple[int, int | None, float]:
    """Return the current phase, optional next phase and blend probability."""
    from target_interception.scenarios import penalty_curriculum

    if penalty_curriculum.enabled(env.cfg):
        return penalty_curriculum.bank_at_step(env.common_step_counter, env.cfg.curriculum_total_steps), None, 0.0
    fractions = env.cfg.curriculum_phase_fractions
    if not fractions:
        return 1, None, 0.0

    p = progress(env)
    end = 0.0
    for phase, fraction in enumerate(fractions, start=1):
        fraction = max(0.0, float(fraction))
        end += fraction
        if fraction == 0.0:
            continue
        if p <= end:
            blend_ratio = max(0.0, min(1.0, float(env.cfg.curriculum_blend_fraction)))
            blend_width = fraction * blend_ratio
            next_phase = next(
                (idx for idx in range(phase + 1, len(fractions) + 1) if float(fractions[idx - 1]) > 0.0),
                None,
            )
            if blend_width > 0.0 and next_phase is not None:
                prob_next = max(0.0, min(1.0, (p - (end - blend_width)) / blend_width))
                return phase, next_phase, prob_next
            return phase, None, 0.0
    enabled = [phase for phase, fraction in enumerate(fractions, start=1) if float(fraction) > 0.0]
    return (enabled[-1] if enabled else 1), None, 0.0


def sample_phase_batch(env, count: int) -> torch.Tensor:
    """Sample phases with independent blend coin flips for resetting environments."""
    phase, next_phase, prob_next = _phase_schedule(env)
    out = torch.full((int(count),), int(phase), dtype=torch.long, device=env.device)
    if next_phase is not None and prob_next > 0.0:
        flip = torch.rand(int(count), device=env.device) < prob_next
        out[flip] = int(next_phase)
    return out


def resample(env, env_ids: torch.Tensor) -> None:
    """Draw pooled scenarios, augment them and assign the resetting environments.

    Group by pool bank for batched writes. Evaluation uses its own suite driver."""
    from . import augment

    pool = env._scenario_pool
    device = env.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)
    n = int(env_ids.numel())
    if n == 0:
        return

    phases = sample_phase_batch(env, n)
    bank_phases = pool_phase_batch(pool, phases)

    # Grow obstacle slots once for the max phase actually used (the op is monotonic + global).
    env._ensure_obstacle_slots_for_phase(int(phases.max().item()))

    # Batched draw + scatter, grouped by bank.
    for bp in torch.unique(bank_phases).tolist():
        ids_g = env_ids[bank_phases == bp]
        k = int(ids_g.numel())
        bank = pool[bp]
        idx = torch.randint(0, int(bank["target_waypoints"].shape[0]), (k,), device=device)
        rows = {key: bank[key][idx] for key in _SCENARIO_ROW_KEYS}
        augment.augment_pool_rows(env, rows, phase=bp)
        assign_scenario_rows(env, ids_g, rows, bp)

    env._move_static_obstacles(env_ids)
    env._move_dynamic_obstacles(env_ids)


def pool_phase_batch(pool: dict, phases: torch.Tensor) -> torch.Tensor:
    """Use the nearest available lower bank, or the first bank below the range."""
    available = phases.new_tensor(sorted(pool))
    indices = torch.searchsorted(available, phases, right=True) - 1
    return available[indices.clamp_min(0)]


# Row fields the env reads; positions and velocities are derived when assigning the row.
_SCENARIO_ROW_KEYS = (
    "target_waypoints",
    "interceptor_start",
    "static_xy",
    "static_active",
    "dynamic_waypoints",
    "dynamic_active",
)


def _first_segment_velocity(env, waypoints: torch.Tensor) -> torch.Tensor:
    """Velocity over the first waypoint segment (works batched: waypoint dim is -3)."""
    dt = float(env._path_waypoint_steps[1] - env._path_waypoint_steps[0]) * env._step_dt
    wp = waypoints.movedim(-3, 0)
    return (wp[1] - wp[0]) / max(dt, env._step_dt)


def assign_scenario_rows(env, env_ids: torch.Tensor | int, rows: dict, phases: torch.Tensor | int) -> None:
    """Assign scenario rows and initialise their derived reset buffers.

    Tensor indices select a batch; an integer index selects one unbatched row.
    First positions and dynamic velocities must be available before the next step.
    """
    waypoints = rows["target_waypoints"]
    env._scenario_phase[env_ids] = phases
    env._target_waypoints[env_ids] = waypoints
    env._interceptor_start_pos[env_ids] = rows["interceptor_start"]
    env._target_command_pos[env_ids] = waypoints[..., 0, :]
    env._static_obstacle_positions_xy[env_ids] = rows["static_xy"]
    env._static_obstacle_active[env_ids] = rows["static_active"]
    env._dynamic_obstacle_waypoints[env_ids] = rows["dynamic_waypoints"]
    env._dynamic_obstacle_active[env_ids] = rows["dynamic_active"]
    env._dynamic_obstacle_positions[env_ids] = rows["dynamic_waypoints"][..., 0, :, :]
    env._dynamic_obstacle_velocities[env_ids] = _first_segment_velocity(env, rows["dynamic_waypoints"])
