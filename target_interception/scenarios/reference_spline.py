"""Cubic target and linear obstacle interpolation.

Generation, validation and runtime share this interpolant. basis() returns
position, velocity, acceleration and jerk weights for the supplied knot times."""
from __future__ import annotations

import numpy as np


def knot_steps(cfg, step_dt: float, last_step: int) -> np.ndarray:
    """Policy steps of the knots: dense over the launch, then every path_waypoint_dt.

    A launch at 7 m/s^2 ends within 0.35 s, and the spline can't follow anything much
    shorter than two knot intervals, so the first path_launch_time gets its own spacing.
    Knots sit on whole policy steps, so a spacing off that clock gets rounded.
    """
    coarse = max(1, round(float(cfg.path_waypoint_dt) / step_dt))
    fine = max(1, round(float(cfg.path_launch_waypoint_dt) / step_dt))
    launch = round(float(cfg.path_launch_time) / step_dt)
    return np.r_[np.arange(0, launch, fine), np.arange(launch, last_step, coarse), last_step]


def _end_slope(h: np.ndarray, count: int) -> np.ndarray:
    """Three-point one-sided first derivative at the last knot, as knot weights."""
    end = np.zeros(count)
    a, b = h[-2], h[-1]
    end[-3:] = (b / (a * (a + b)), -(a + b) / (a * b), (2 * b + a) / (b * (a + b)))
    return end


def second_derivative_matrix(times: np.ndarray) -> np.ndarray:
    """Matrix W with knot accelerations M = W @ p for a clamped cubic spline.

    Every row starts from rest, so the start slope is exactly zero. The last knot has no such
    guarantee and takes its slope from the positions.
    """
    times = np.asarray(times, dtype=np.float64)
    count = len(times)
    if count < 3:
        return np.zeros((count, count))
    h = np.diff(times)
    if np.any(h <= 0.0):
        raise ValueError("Spline knot times must be strictly increasing")
    lhs = np.zeros((count, count))
    rhs = np.zeros((count, count))
    for i in range(1, count - 1):
        lhs[i, i - 1 : i + 2] = (h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i])
        rhs[i, i - 1 : i + 2] = (6.0 / h[i - 1], -6.0 / h[i - 1] - 6.0 / h[i], 6.0 / h[i])
    lhs[0, :2] = (2.0 * h[0], h[0])
    rhs[0, :2] = (-6.0 / h[0], 6.0 / h[0])
    lhs[-1, -2:] = (h[-1], 2.0 * h[-1])
    rhs[-1, -2:] = (6.0 / h[-1], -6.0 / h[-1])
    rhs[-1] += 6.0 * _end_slope(h, count)
    return np.linalg.solve(lhs, rhs)


def basis(times: np.ndarray, queries: np.ndarray, *, cubic: bool) -> np.ndarray:
    """Position/velocity/acceleration/jerk weights [4, Q, K] at the query times."""
    times = np.asarray(times, dtype=np.float64)
    queries = np.clip(np.asarray(queries, dtype=np.float64), times[0], times[-1])
    count = len(times)
    out = np.zeros((4, len(queries), count))
    rows = np.arange(len(queries))
    # A query on a knot belongs to the segment that starts there.
    seg = np.clip(np.searchsorted(times[1:], queries, side="right"), 0, count - 2)
    h = times[seg + 1] - times[seg]
    u = queries - times[seg]
    out[0, rows, seg] = (h - u) / h
    out[0, rows, seg + 1] = u / h
    out[1, rows, seg] = -1.0 / h
    out[1, rows, seg + 1] = 1.0 / h
    if not cubic or count < 3:
        return out
    # Weights on the knot accelerations M_i, M_{i+1}, mapped back through M = W p.
    on_m = np.zeros((4, len(queries), count))
    v = h - u
    on_m[0, rows, seg] = v**3 / (6.0 * h) - h * v / 6.0
    on_m[0, rows, seg + 1] = u**3 / (6.0 * h) - h * u / 6.0
    on_m[1, rows, seg] = -(v**2) / (2.0 * h) + h / 6.0
    on_m[1, rows, seg + 1] = u**2 / (2.0 * h) - h / 6.0
    on_m[2, rows, seg] = v / h
    on_m[2, rows, seg + 1] = u / h
    on_m[3, rows, seg] = -1.0 / h
    on_m[3, rows, seg + 1] = 1.0 / h
    return out + on_m @ second_derivative_matrix(times)
