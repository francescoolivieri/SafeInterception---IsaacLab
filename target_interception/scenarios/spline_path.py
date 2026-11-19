"""Single cubic B-spline fitting of an A* route.

p(s) = sum_j B_j(s) c_j, where s is raw-route arc length, r(s) the raw route and
the control points c_j are the unknowns. At samples s_i:

    minimize    sum_i  F_i ||p(s_i) - r(s_i)||^2  +  LAMBDA_2 ||p''(s_i)||^2
    subject to  p = r at both endpoints
                |p(s_i) - r(s_i)| <= delta_i       per axis (corridor)
                lo <= c_j <= hi                    (arena)
                n_i . p(s_i) >= n_i . o + rho      (pillar o, tangent line seen from r(s_i))
"""
from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BSpline

KNOT_SPACING = .15      # m of s, largest distance between knots
SAMPLE_SPACING = .02    # m of s, distance between samples s_i
H_JUNCTION = .65        # h, half-width of the junction blend b(s)
F_NUM, F_DEN = .12, .00035   # F = F_NUM / (F_DEN + b (F_NUM - F_DEN)): ~343 inside, 1 at a junction
DELTA_0, DELTA_1 = .15, .30  # delta = DELTA_0 + DELTA_1 b
LAMBDA_2 = .15          # weight of the curvature term ||p''||^2
PILLAR_BUFFER = .025    # rho = inflated pillar radius + PILLAR_BUFFER
DELTA_TOLERANCE = .002  # allowed solver overshoot of delta in the final check


def junction_profile(s, junctions, h=H_JUNCTION):
    """Cosine relaxation: return positional weight F(s) and per-axis corridor delta(s)."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Spline junction halfwidth must be positive and finite")
    d = (np.abs(s[:, None] - np.asarray(junctions)[None]).min(axis=1)
         if len(junctions) else np.full(len(s), np.inf))
    b = .5 * (1 + np.cos(np.pi * np.minimum(d / h, 1)))
    return F_NUM / (F_DEN + b * (F_NUM - F_DEN)), DELTA_0 + b * DELTA_1


def _clean_route(route):
    xy = np.asarray(route, dtype=np.float64)
    if len(xy) == 0:
        return xy
    return xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]


def _arc_lengths(xy):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _reference_at(xy, s):
    s_xy = _arc_lengths(xy)
    return np.column_stack([np.interp(s, s_xy, xy[:, axis]) for axis in range(2)])


def path_is_clear(xy, geom):
    """Check every segment of the actual polyline, including segment interiors."""
    xy = np.asarray(xy)
    if not np.isfinite(xy).all() or np.any(xy < geom["lo"]) or np.any(xy > geom["hi"]):
        return False
    delta = np.diff(xy, axis=0)
    length2 = np.maximum(np.sum(delta * delta, axis=1), 1e-20)
    for center in geom["centers"]:
        fraction = np.clip(np.sum((center - xy[:-1]) * delta, axis=1) / length2, 0, 1)
        nearest = xy[:-1] + fraction[:, None] * delta
        if np.any(np.linalg.norm(nearest - center, axis=1) < geom["radius"]):
            return False
    return True


def fit_route(raw, geom, junctions=(), h=H_JUNCTION):
    """Fit one cubic with a quadratic smoothness objective and safe halfspaces.

    Raw arc length is the parameter, not physical time. Tangent halfspaces around
    inflated pillars preserve their side at each sample.
    The objective penalizes fit error and the second parameter derivative;
    it is a geometric surrogate, not minimum-time optimization.
    Constraints are sampled every 2 cm of raw arc length. Exact segment checks
    validate the returned polyline; None means this route could not be fitted.
    """
    raw = _clean_route(raw)
    if len(raw) < 2 or not np.isfinite(raw).all():
        return None
    lo, hi = geom["lo"], geom["hi"]
    if np.any(raw[[0, -1]] < lo) or np.any(raw[[0, -1]] > hi):
        return None

    # 1. Fixed cubic basis over raw-route distance s. Only control points c_j vary.
    length = _arc_lengths(raw)[-1]
    breaks = np.linspace(0., length, max(2, int(np.ceil(length / KNOT_SPACING)) + 1))
    knots = np.r_[np.zeros(4), breaks[1:-1], np.full(4, length)]
    n_ctrl = len(knots) - 4
    s = np.linspace(0., length, max(2, int(np.ceil(length / SAMPLE_SPACING)) + 1))
    r = _reference_at(raw, s)                                             # r(s_i)
    B = sparse.csc_matrix(BSpline.design_matrix(s, knots, 3))             # B[i, j] = B_j(s_i)
    D2 = sparse.csc_matrix(BSpline(knots, np.eye(n_ctrl), 3)(s, nu=2))   # D2[i, j] = B_j''(s_i)
    # 2. Quadratic objective 1/2 x^T P x + q^T x with x = [c_x; c_y]:
    # per axis, fidelity + curvature = c^T H c - 2 (B^T F r)^T c + const.
    F, delta = junction_profile(s, junctions, h)
    H = B.T @ sparse.diags(F) @ B + D2.T @ (LAMBDA_2 * sparse.eye(len(s))) @ D2
    H += 1e-8 * sparse.eye(n_ctrl)
    P = sparse.block_diag([2 * H, 2 * H], format="csc")
    q = np.r_[-2 * (B.T @ (F * r[:, 0])), -2 * (B.T @ (F * r[:, 1]))]
    # 3. Linear constraints l <= A x <= u on sampled positions and control points.
    p_lo = np.maximum(r - delta[:, None], lo)
    p_hi = np.minimum(r + delta[:, None], hi)
    if np.any(p_lo > p_hi):
        return None
    p_lo[[0, -1]] = p_hi[[0, -1]] = r[[0, -1]]
    A_blocks = [sparse.block_diag([B, B], format="csc")]
    l_blocks, u_blocks = [p_lo.T.ravel()], [p_hi.T.ravel()]
    # Bounding control points bounds the entire spline (convex hull), also between samples.
    A_blocks.append(sparse.eye(2 * n_ctrl, format="csc"))
    l_blocks.append(np.repeat(lo, n_ctrl))
    u_blocks.append(np.repeat(hi, n_ctrl))

    def halfspaces(n, support):
        # Rows n_i . p(s_i) >= support_i. Each p(s_i) already lies in [p_lo, p_hi].
        # A supporting plane containing that entire box adds no restriction.
        # Keep a numerical margin: pruning never weakens the feasible set.
        minimum = np.sum(n * np.where(n >= 0, p_lo, p_hi), axis=1)
        maximum = np.sum(n * np.where(n >= 0, p_hi, p_lo), axis=1)
        if np.any(maximum < support - 1e-10):
            # At least one whole position box is outside a required plane.
            # This is an infeasibility certificate, independent of OSQP.
            return False
        needed = minimum < support + 1e-10
        if not np.any(needed):
            return True
        B_needed = B[needed]
        n, support = n[needed], support[needed]
        A_blocks.append(sparse.hstack([sparse.diags(n[:, 0]) @ B_needed,
                                       sparse.diags(n[:, 1]) @ B_needed], format="csc"))
        l_blocks.append(support)
        u_blocks.append(np.full(len(support), np.inf))
        return True

    for o in geom["centers"]:
        n = r - o
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-10)
        # support_i = n_i . o + rho, with rho = radius + PILLAR_BUFFER
        if not halfspaces(n, n @ o + geom["radius"] + PILLAR_BUFFER):
            return None
    # 4. Solve, then validate the polyline that timing will actually use.
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=sparse.vstack(A_blocks, format="csc"),
                 l=np.concatenate(l_blocks), u=np.concatenate(u_blocks), verbose=False,
                 eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, polish=True)
    result = solver.solve()
    if result.info.status_val not in (1, 2):   # 1 solved, 2 solved inaccurate
        return None
    c = result.x.reshape(2, n_ctrl).T
    # Remove solver-tolerance wall overshoot without clipping the curve itself.
    c = np.clip(c, lo, hi)
    c[[0, -1]] = raw[[0, -1]]
    p = BSpline(knots, c, 3)(s)
    if np.max(np.abs(p - r) - delta[:, None]) > DELTA_TOLERANCE:
        return None
    return p if path_is_clear(p, geom) else None
