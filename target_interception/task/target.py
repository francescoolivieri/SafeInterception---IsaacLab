"""Target bounds and kinematic waypoint execution.

Wander routes use fixed geometric smoothing and curvature-based speed timing.
References stay at a sampled altitude. Their knots are executed on the episode
clock by the interpolant in scenarios/reference_spline.py.
"""
from __future__ import annotations


import numpy as np
import torch

from target_interception.scenarios import reference_spline
from target_interception.scenarios.generation_stats import count as count_generation



MOTION_MODE_ID = {"hover": 0, "wander": 2}
INITIAL_MOTION_ID = {"hover": 0, "toward": 1, "transverse": 2, "away": 3}


def target_radius(env) -> float:
    return float(env.cfg.scenario.target_radius)


def wall_clearance(env) -> float:
    return max(float(env.cfg.scenario.target_wall_clearance), target_radius(env))


def arena_bounds(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Target centre bounds: altitude from wall_clearance, horizontal from the body-to-wall gap."""
    lo = env._arena_min_safe + wall_clearance(env)
    hi = env._arena_max_safe - wall_clearance(env)
    margin = target_radius(env) + max(0.0, float(env.cfg.scenario.target_wall_surface_clearance_xy))
    lo[:2] = torch.as_tensor(env.cfg.arena_min[:2], device=lo.device) + margin
    hi[:2] = torch.as_tensor(env.cfg.arena_max[:2], device=hi.device) - margin
    return lo, hi


def start_distance_min(env) -> float:
    return float(env.cfg.scenario.interceptor_min_target_distance)


def min_start_distance_for_toward_motion(env) -> float:
    """Horizontal start separation required before initial toward motion is allowed."""
    return float(env.cfg.scenario.target_min_start_distance_for_toward_motion)


def altitude_bounds(env) -> tuple[float, float]:
    clearance = wall_clearance(env)
    lo = float(env._arena_min_safe[2]) + clearance
    hi = float(env._arena_max_safe[2]) - clearance
    if hi <= lo:
        lo = float(env._arena_min_safe[2]) + target_radius(env)
        hi = float(env._arena_max_safe[2]) - target_radius(env)
    return lo, hi


def _knot_times(env) -> np.ndarray:
    return env._path_waypoint_steps.detach().cpu().double().numpy() * float(env._step_dt)


def _step_basis(env, device, dtype, cubic: bool) -> torch.Tensor:
    """Cached [4, path_steps, K] weights at every integer policy step."""
    cache = env.__dict__.setdefault("_target_step_basis", {})
    key = (cubic, str(device), dtype)
    if key not in cache:
        steps = np.arange(int(env._path_steps), dtype=np.float64) * float(env._step_dt)
        weights = reference_spline.basis(_knot_times(env), steps, cubic=cubic)
        cache[key] = torch.as_tensor(weights, device=device, dtype=dtype)
    return cache[key]


def eval_target_steps(env, waypoints: torch.Tensor, env_ids: torch.Tensor, step: torch.Tensor):
    """Target position, velocity, acceleration and jerk at integer policy steps."""
    weights = _step_basis(env, waypoints.device, waypoints.dtype, True)
    return torch.einsum("jek,ekd->jed", weights[:, step.to(torch.long)], waypoints[env_ids]).unbind(0)


def eval_target(env, waypoints: torch.Tensor, env_ids: torch.Tensor, step: torch.Tensor):
    """Target position, velocity and acceleration at fractional policy steps (offline tools)."""
    times = torch.as_tensor(step).detach().cpu().double().numpy() * float(env._step_dt)
    weights = reference_spline.basis(_knot_times(env), times, cubic=True)[:3]
    weights = torch.as_tensor(weights, device=waypoints.device, dtype=waypoints.dtype)
    motion = torch.einsum("jek,ekd->jed", weights, waypoints[env_ids])
    return motion[0], motion[1], motion[2]


def sample_policy_steps(env, waypoints: torch.Tensor, *, cubic: bool, knot_dim: int = -2) -> torch.Tensor:
    """Resample knots onto every policy step; knot_dim indexes the knot axis."""
    weights = _step_basis(env, waypoints.device, waypoints.dtype, cubic)[0]
    return (waypoints.movedim(knot_dim, -1) @ weights.T).movedim(-1, knot_dim)


def reference_kinematics(env, waypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Executed velocity and acceleration of one [K, 3] reference, in float64."""
    weights = _step_basis(env, waypoints.device, torch.float64, True)
    waypoints = waypoints.double()
    return weights[1] @ waypoints, weights[2] @ waypoints


def reduce_waypoint_speed(env, waypoints: torch.Tensor, decrement: float = 0.1) -> torch.Tensor:
    """Slower retiming of an offline row that failed physical replay.

    The route runs on a slower clock over the same episode, which lowers the peak
    reference speed by about decrement and drops the unused tail. Rerun the full row
    audit afterwards: moving obstacles keep the episode clock, so the new timing can
    create a new encounter.
    """
    if waypoints.ndim != 2 or waypoints.shape != (env._path_waypoint_count, 3):
        raise ValueError(
            "waypoints must have shape "
            f"({env._path_waypoint_count}, 3), got {tuple(waypoints.shape)}"
        )
    decrement = float(decrement)
    if decrement <= 0.0:
        raise ValueError("speed decrement must be positive")
    velocity = reference_kinematics(env, waypoints)[0]
    peak = float(torch.linalg.vector_norm(velocity, dim=-1).max().item())
    if peak <= decrement + 1e-6:
        return waypoints[0].expand_as(waypoints).clone()
    scale = (peak - decrement) / peak
    sample_steps = env._path_waypoint_steps.to(torch.float64) * scale
    rows = torch.zeros(env._path_waypoint_count, dtype=torch.long, device=waypoints.device)
    return eval_target(env, waypoints.unsqueeze(0), rows, sample_steps)[0]


# The cubic reconstruction can exceed the planned cruise speed by under 1%.
SPEED_RECONSTRUCTION_MARGIN = 1.02


def reference_kinematics_violations(env, waypoints: torch.Tensor) -> tuple[str, ...]:
    """Return every violated limit of the executed reference."""
    velocity, acceleration = reference_kinematics(env, waypoints)
    scn = env.cfg.scenario
    checks = {
        "speed": torch.linalg.vector_norm(velocity, dim=-1)
        <= SPEED_RECONSTRUCTION_MARGIN * float(getattr(scn, "target_max_speed", 2.2)) + 1e-5,
        "vertical_velocity": torch.abs(velocity[:, 2]) <= 1e-6,
        "acceleration": torch.linalg.vector_norm(acceleration, dim=-1)
        <= float(getattr(scn, "target_max_acceleration", 6.5)) + 1e-5,
    }
    return tuple(name for name, check in checks.items() if not bool(check.all().item()))


def reference_kinematics_ok(env, waypoints: torch.Tensor) -> bool:
    """Final kinematic gate for every generated motion family."""
    violations = reference_kinematics_violations(env, waypoints)
    for reason in violations:
        count_generation(env, f"motion.{reason}")
    return not violations


def first_velocity(env, waypoints: torch.Tensor) -> torch.Tensor:
    """Velocity of the first moving knot interval (initial heading and bucket)."""
    dt = (env._path_waypoint_steps[1:] - env._path_waypoint_steps[:-1]).to(waypoints.dtype) * env._step_dt
    velocity = (waypoints[1:] - waypoints[:-1]) / dt.unsqueeze(-1).clamp_min(1e-6)
    moving = torch.nonzero(torch.linalg.vector_norm(velocity, dim=-1) > 0.05, as_tuple=False).flatten()
    return velocity[int(moving[0].item())] if moving.numel() else torch.zeros(3, device=env.device)


def eval_waypoints(env, waypoints: torch.Tensor, env_ids: torch.Tensor, step: torch.Tensor):
    """Linear evaluation of a waypoint schedule at fractional policy steps.

    Moving obstacles always use this; the target uses eval_target_steps.
    """
    step = step.to(device=env.device, dtype=torch.float32)
    boundaries = env._path_waypoint_steps[1:].to(torch.float32)
    seg = torch.searchsorted(boundaries, step, right=True).clamp(max=env._path_waypoint_count - 2)
    start = env._path_waypoint_steps[seg].to(torch.float32)
    end = env._path_waypoint_steps[seg + 1].to(torch.float32)
    duration = (end - start).clamp_min(1.0)
    tau = ((step - start) / duration).clamp(0.0, 1.0)
    p0 = waypoints[env_ids, seg]
    p1 = waypoints[env_ids, seg + 1]
    broadcast_tail = (1,) * (p0.ndim - 1)
    tau = tau.view(-1, *broadcast_tail)
    duration = duration.view(-1, *broadcast_tail)
    pos = p0 * (1.0 - tau) + p1 * tau
    vel = (p1 - p0) / (duration * env._step_dt).clamp_min(env._step_dt)
    return pos, vel


def update_episode_motion(env, env_ids: torch.Tensor | None = None) -> None:
    """Refresh target commands and analytic moving-obstacle state."""
    if env_ids is None:
        env_ids = env._robot._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    step = env.episode_length_buf[env_ids].clamp(0, env._path_steps - 1)
    pos, vel, acc, jerk = eval_target_steps(env, env._target_waypoints, env_ids, step)
    env._target_command_pos[env_ids] = pos
    env._target_command_vel[env_ids] = vel
    if hasattr(env, "_target_policy_step_pos"):
        env._target_policy_step_pos[env_ids] = pos
        env._target_policy_step_vel[env_ids] = vel
        # The reference holds its final pose once the schedule has ended.
        moving = (env.episode_length_buf[env_ids] < env._path_steps - 1).to(vel.dtype).unsqueeze(-1)
        env._target_kinematic_step_vel[env_ids] = vel * moving
        env._target_kinematic_step_acc[env_ids] = acc * moving
        env._target_kinematic_step_jerk[env_ids] = jerk * moving
        env._target_command_acc[env_ids] = acc * moving
        env._target_kinematic_substep = 0

    if env._max_dynamic_obstacles > 0:
        active_ids = env_ids[torch.any(env._dynamic_obstacle_active[env_ids], dim=1)]
        if active_ids.numel():
            active_step = torch.clamp(env.episode_length_buf[active_ids], 0, env._path_steps - 1).to(torch.float32)
            env._dynamic_obstacle_positions[active_ids], env._dynamic_obstacle_velocities[active_ids] = eval_waypoints(
                env, env._dynamic_obstacle_waypoints, active_ids, active_step
            )
            env._move_dynamic_obstacles(active_ids)


def update_kinematic_command_substep(env) -> None:
    """Evaluate the step's cubic exactly at every physics substep (Taylor form)."""
    substep = int(env._target_kinematic_substep)
    if substep:
        from .kinematic_spline import evaluate_substep

        evaluate_substep(env, substep * float(env.sim.cfg.dt))
    env._target_kinematic_substep = substep + 1


def initial_motion_bucket(interceptor_start: torch.Tensor, waypoints: torch.Tensor, velocity: torch.Tensor) -> str:
    speed = torch.linalg.vector_norm(velocity[:2])
    if float(speed.item()) <= 0.05:
        return "hover"
    away = waypoints[0, :2] - interceptor_start[:2]
    away /= torch.linalg.vector_norm(away).clamp_min(1e-6)
    alignment = float(torch.dot(velocity[:2] / speed, away).item())
    if alignment > 0.3:
        return "away"
    if alignment < -0.3:
        return "toward"
    return "transverse"
