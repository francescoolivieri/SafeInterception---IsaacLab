"""Offline path rounding and speed timing; no simulator dynamics.

Scalar timing loops use NumPy on the CPU. Final waypoints return to the caller's
Torch device and are validated as executed (see reference_spline.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def arc_lengths(xy: torch.Tensor) -> torch.Tensor:
    lengths = torch.linalg.vector_norm(torch.diff(xy, dim=0), dim=-1)
    return torch.cat((xy.new_zeros(1), lengths.cumsum(0)))


def interpolate(distance: torch.Tensor, values: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of vector values at increasing path distances."""
    if len(values) == 1:
        return values[:1].expand(len(samples), -1)
    samples = samples.clamp(0.0, distance[-1])
    segment = torch.searchsorted(distance[1:], samples).clamp(max=len(distance) - 2)
    fraction = (samples - distance[segment]) / (distance[segment + 1] - distance[segment]).clamp_min(1e-6)
    return torch.lerp(values[segment], values[segment + 1], fraction.unsqueeze(-1))


def resample(xy: torch.Tensor, spacing: float) -> torch.Tensor:
    if len(xy) <= 2:
        return xy
    distance = arc_lengths(xy)
    count = max(2, math.ceil(float(distance[-1]) / max(spacing, 1e-6)) + 1)
    samples = torch.linspace(0.0, float(distance[-1]), count, device=xy.device, dtype=xy.dtype)
    return interpolate(distance, xy, samples)


def round_corners(xy: torch.Tensor, passes: int) -> torch.Tensor:
    """Chaikin corner cutting, with the route's endpoints preserved."""
    for _ in range(max(0, passes)):
        if len(xy) < 3:
            break
        a, b = xy[:-1], xy[1:]
        cuts = torch.stack((0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b), dim=1)
        xy = torch.cat((xy[:1], cuts.flatten(0, 1), xy[-1:]))
    return xy


def geometric_path_free(grid, xy: torch.Tensor) -> bool:
    """Check the target's segments against continuous obstacle geometry."""
    from . import spline_path

    return spline_path.path_is_clear(xy.detach().cpu().double().numpy(), grid["collision_geometry"])


def _step_speed(u0: float, s: float, a_max: float, kappa: float) -> float:
    """Largest v reachable from v0^2 = u0 over s metres within the shared bound.

    With u = v^2 the tangential budget is sqrt(a_max^2 - kappa^2 u^2), so
    u - u0 = 2 s sqrt(a_max^2 - kappa^2 u^2), i.e.
    (1 + c) u^2 - 2 u0 u + u0^2 - 4 s^2 a_max^2 = 0 with c = 4 s^2 kappa^2.
    """
    c = 4.0 * s * s * kappa * kappa
    disc = (1.0 + c) * 4.0 * s * s * a_max * a_max - c * u0 * u0
    if disc <= 0.0:
        return math.sqrt(u0)
    return math.sqrt(max(u0, (u0 + math.sqrt(disc)) / (1.0 + c)))


def speed_profile(xy: np.ndarray, peak: float, a_max: float, spacing: float = 0.05):
    """Fastest speed along a route that starts at rest.

    Two limits: v <= peak and one horizontal acceleration bound
    (v^2 kappa)^2 + (dv/dt)^2 <= a_max^2 (a quadrotor tilts its thrust the same way to
    turn as to brake). At a vertex the turn alone caps v <= sqrt(a_max / kappa); between
    points a backward braking pass and a forward acceleration pass spend the tangential
    budget left by the larger curvature of the two ends. Long segments get collinear
    points every spacing metres so speed can change along a straight. Returns arc length
    and speed at those points; acceleration is constant in between.
    """
    if a_max <= 0.0:
        raise ValueError("The acceleration limit must be positive")
    lengths = np.maximum(np.linalg.norm(np.diff(xy, axis=0), axis=-1), 1e-6)
    kappa_vertex = np.zeros(len(xy))
    if len(xy) > 2:
        direction = np.diff(xy, axis=0) / lengths[:, None]
        angle = np.arccos(np.clip(np.sum(direction[:-1] * direction[1:], axis=-1), -1.0, 1.0))
        kappa_vertex[1:-1] = angle / (0.5 * (lengths[:-1] + lengths[1:]))
    pieces = np.maximum(1, np.ceil(lengths / spacing - 1e-9).astype(int))
    fraction = np.concatenate([np.arange(n) / n for n in pieces])
    start = np.r_[0.0, np.cumsum(lengths)][:-1]
    distance = np.r_[np.repeat(start, pieces) + fraction * np.repeat(lengths, pieces), lengths.sum()]
    vertex = np.r_[0, np.cumsum(pieces)]
    kappa = np.zeros(len(distance))
    kappa[vertex] = kappa_vertex
    speed = np.full(len(distance), peak)
    speed[vertex] = np.minimum(peak, np.sqrt(a_max / np.maximum(kappa_vertex, 1e-6)))
    speed[0] = 0.0
    step = np.diff(distance)
    kappa_step = np.maximum(kappa[:-1], kappa[1:])
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], _step_speed(speed[i + 1] ** 2, step[i], a_max, kappa_step[i]))
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], _step_speed(speed[i - 1] ** 2, step[i - 1], a_max, kappa_step[i - 1]))
    return distance, speed


def profiled_waypoints(env, xy: torch.Tensor, step_dist: torch.Tensor, z: float) -> torch.Tensor:
    """Time a route with speed_profile and sample it at the knot times.

    The cruise speed is the largest requested step_dist / dt, capped by
    target_max_speed. Acceleration is constant between profile points, so the
    distance at each knot time is exact. A route shorter than the episode leaves
    the target parked at its end.
    """
    waypoints = xy.new_zeros((env._path_waypoint_count, 3))
    waypoints[:, :2] = xy[0]
    waypoints[:, 2] = z
    if len(xy) < 2:
        return waypoints
    cfg = env.cfg.scenario
    dt = torch.diff(env._path_waypoint_steps).cpu().numpy() * env._step_dt
    peak = min(float(np.max(step_dist.detach().cpu().numpy() / dt)), float(getattr(cfg, "target_max_speed", 2.2)))
    if peak <= 1e-8:
        return waypoints
    route = xy.detach().cpu().double().numpy()
    distance, speed = speed_profile(route, peak, float(getattr(cfg, "target_acceleration_limit", 7.0)))
    length = np.diff(distance)
    vertex_time = np.r_[0.0, np.cumsum(2.0 * length / np.maximum(speed[:-1] + speed[1:], 1e-9))]
    knot_time = np.r_[0.0, np.cumsum(dt)]
    # s_i + v_i tau + a_i tau^2 / 2 inside each profile interval.
    i = np.clip(np.searchsorted(vertex_time, knot_time, side="right") - 1, 0, len(length) - 1)
    tau = knot_time - vertex_time[i]
    acceleration = (speed[i + 1] ** 2 - speed[i] ** 2) / (2.0 * np.maximum(length[i], 1e-12))
    travelled = np.minimum(distance[i] + speed[i] * tau + 0.5 * acceleration * tau**2, distance[i + 1])
    waypoints[:, :2] = interpolate(arc_lengths(xy), xy, xy.new_tensor(travelled))
    return waypoints


def timed_route_rejection(env, grid, candidate: torch.Tensor) -> str | None:
    """Collision, kinematics and route-length checks on the executed reference."""
    from target_interception.task import target

    executed = target.sample_policy_steps(env, candidate, cubic=True)
    if not geometric_path_free(grid, executed[:, :2]):
        return "timed_collision"
    if not target.reference_kinematics_ok(env, candidate):
        return "kinematics_rejected"
    # Reject an exhausted route instead of parking before the episode ends.
    if float(torch.linalg.vector_norm(candidate[-1] - candidate[-2])) <= 1e-6:
        return "route_exhausted"
    return None


def spline_waypoints(env, grid, route, junctions, step_dist, z):
    """Fit once, time the curve, then validate the executed reference."""
    from . import spline_path
    from .generation_stats import count, time_stage
    geometry = grid["spline_geometry"]
    count(env, "spline.attempts")
    with time_stage(env, "spline_fit"):
        curve = spline_path.fit_route(
            route.detach().cpu().double().numpy(), geometry, junctions,
            env.cfg.scenario.target_spline_junction_halfwidth,
        )
    if curve is None:
        count(env, "spline.fit_rejected")
        return None
    xy = route.new_tensor(curve)
    with time_stage(env, "spline_timing"):
        candidate = profiled_waypoints(env, xy, step_dist, z)
    rejection = timed_route_rejection(env, grid, candidate)
    if rejection is not None:
        count(env, f"spline.{rejection}")
        return None
    count(env, "spline.accepted")
    # The untimed curve lets the evaluation suite replay this route at other speeds.
    env.last_accepted_route_xy = xy
    return candidate
