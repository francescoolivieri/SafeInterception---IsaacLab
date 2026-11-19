"""M1/M2 safety filter: first-order CBF-QP on the velocity command (single-integrator model)."""
from __future__ import annotations

import numpy as np
import torch
from qpsolvers import SolverError, solve_qp

FEASIBILITY_TOL = 1.0e-5


class CBFQPFilter:
    """Smallest change of the nominal command that satisfies every barrier condition.

    One row of G u <= b per obstacle slot and per arena plane, in physical velocity:
        cylinder j:   -n_j^T (u - v_y_j) <= -alpha h_j,   h_j = d_safe - ||(p_p - p_y_j)_xy||
        arena:        -u <= -alpha (p_min - p_p),          u <= -alpha (p_p - p_max)
    Padding slots sit far outside the arena and never bind. The QP itself is solved in the
    normalised action a = u / S, so the rows are scaled by S before solve_cbf_qp.
    """

    def __init__(self, obs_layout, *, d_safe: float, alpha: float, command_scale, w_delta: float = 1.0e4,
                 device=None) -> None:
        if obs_layout.extras.get("obstacles") is None or obs_layout.extras.get("obstacle_vel") is None:
            raise ValueError("CBF-QP needs obstacle positions and velocities in the observation")
        if d_safe <= 0.0 or alpha <= 0.0:
            raise ValueError("d_safe and alpha must be positive")
        self.layout = obs_layout
        self.d_safe = float(d_safe)
        self.alpha = float(alpha)
        self.w_delta = float(w_delta)
        self.device = torch.device(device or "cpu")
        self.S = torch.as_tensor(command_scale, dtype=torch.float32, device=self.device).reshape(-1)[:3]
        self._info: dict = {}

    def to(self, device) -> "CBFQPFilter":
        self.device = torch.device(device)
        self.S = self.S.to(self.device)
        return self

    def constraints(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_env = obs.shape[0]
        ego = obs[:, self.layout.ego]
        p_p, v_p = ego[:, :3], ego[:, 3:6]
        offset = obs[:, self.layout.extras["obstacles"]].reshape(n_env, -1, 3)[..., :2]  # p_y - p_p
        v_y = obs[:, self.layout.extras["obstacle_vel"]].reshape(n_env, -1, 2) + v_p[:, None, :2]
        dist = torch.linalg.vector_norm(offset, dim=-1)
        n = -offset / dist.clamp_min(1.0e-6).unsqueeze(-1)  # from the cylinder to the pursuer
        h = self.d_safe - dist

        G = [torch.cat((-n, n.new_zeros(n.shape[:-1] + (1,))), dim=-1)]
        b = [-self.alpha * h - torch.sum(n * v_y, dim=-1)]
        if self.layout.arena_min is not None and self.layout.arena_max is not None:
            p_min = torch.as_tensor(self.layout.arena_min, dtype=obs.dtype, device=obs.device)
            p_max = torch.as_tensor(self.layout.arena_max, dtype=obs.dtype, device=obs.device)
            I = torch.eye(3, dtype=obs.dtype, device=obs.device)
            G.append(torch.cat((-I, I)).expand(n_env, -1, -1))
            b.append(-self.alpha * torch.cat((p_min - p_p, p_p - p_max), dim=-1))
        return torch.cat(G, dim=1), torch.cat(b, dim=1)

    def __call__(self, obs: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        a_nom = nominal_action.to(self.device)
        G, b = self.constraints(obs)
        G = G * self.S  # rows in normalised action units
        a_star, delta = solve_cbf_qp(a_nom[:, :3], G, b, self.w_delta)
        action = a_nom.clone()
        action[:, :3] = a_star  # yaw rate passes through
        self._info = qp_stats(a_nom[:, :3], a_star, G, b, delta)
        return action.to(nominal_action.device)

    def info(self) -> dict:
        return dict(self._info)


def solve_cbf_qp(a_nom: torch.Tensor, G: torch.Tensor, b: torch.Tensor, w_delta: float = 1.0e4):
    """One QP per environment, min 1/2 ||a - a_nom||^2 (+ 1/2 w_delta delta^2) s.t. G a <= b (+ delta), |a| <= 1.

    The slack delta is only added when the QP without it is infeasible.
    """
    a_nom_np = a_nom.detach().cpu().double().numpy()
    G_np, b_np = G.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()
    n = a_nom_np.shape[1]
    ones = np.ones(n)
    a_star, delta = [], []
    for a0, Gi, bi in zip(a_nom_np, G_np, b_np):
        a = _quadprog(np.eye(n), -a0, Gi, bi, -ones, ones)
        if a is not None and np.max(Gi @ a - bi) <= FEASIBILITY_TOL:
            a_star.append(a)
            delta.append(0.0)
            continue
        x = _quadprog(np.diag(np.r_[ones, w_delta]), np.r_[-a0, 0.0], np.hstack((Gi, -np.ones((len(bi), 1)))), bi,
                      np.r_[-ones, 0.0], np.r_[ones, np.inf])
        if x is None:
            raise RuntimeError("CBF-QP solver failed")
        a_star.append(x[:n])
        delta.append(x[n])
    return a_nom.new_tensor(np.asarray(a_star)), a_nom.new_tensor(delta)


def _quadprog(P, q, G, h, lb, ub):
    try:
        return solve_qp(P, q, G, h, lb=lb, ub=ub, solver="quadprog")
    except SolverError:
        return None


def solve_cbf_qp_batched(
    nominal: torch.Tensor,
    G: torch.Tensor,
    b: torch.Tensor,
    *,
    w_delta: float = 1.0e4,
    iterations: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate solve_cbf_qp for many rows, batched on the device.

    Hildreth's dual coordinate ascent on min |u - a|^2 / 2 s.t. the CBF rows G u <= b
    (soft, slack weight w_delta as in the CPU fallback) and the hard [-1, 1] box, one
    multiplier at a time in closed form. A fixed sweep count is fast but inexact, fine
    for policy-iteration data collection; evaluation uses the exact quadprog path.
    """
    count, rows, controls = G.shape
    eye = torch.eye(controls, device=G.device, dtype=G.dtype).expand(count, -1, -1)
    A = torch.cat((G, eye, -eye), dim=1)
    c = torch.cat((b, b.new_ones(count, 2 * controls)), dim=1)
    inverse_weight = torch.cat((A.new_full((rows,), 1.0 / w_delta), A.new_zeros(2 * controls)))
    denominator = A.square().sum(dim=-1) + inverse_weight
    denominator = torch.where(denominator > 1e-12, denominator, torch.ones_like(denominator))
    multiplier = A.new_zeros(count, A.shape[1])
    action = nominal.clone()
    for _ in range(iterations):
        for row in range(A.shape[1]):
            residual = (A[:, row] * action).sum(dim=-1) - c[:, row] - multiplier[:, row] * inverse_weight[row]
            updated = (multiplier[:, row] + residual / denominator[:, row]).clamp_min(0.0)
            action = action - (updated - multiplier[:, row])[:, None] * A[:, row]
            multiplier[:, row] = updated
    slack = (multiplier[:, :rows] * inverse_weight[:rows]).amax(dim=-1)
    return action.clamp(-1.0, 1.0), slack


def qp_stats(a_nom, a_star, G, b, delta) -> dict:
    change = torch.linalg.vector_norm(a_star - a_nom, dim=-1)
    residual = (G @ a_star.unsqueeze(-1)).squeeze(-1) - b
    violation = residual.clamp_min(0.0)
    feasible = violation.amax(dim=1) <= FEASIBILITY_TOL
    return {
        "intervention_mean": float(change.mean()),
        "intervention_max": float(change.max()),
        "intervened_fraction": float((change > 1.0e-4).float().mean()),
        "active_constraints_mean": float((residual.abs() < 1.0e-4).float().sum(dim=1).mean()),
        "qp_violation_max": float(violation.max()),
        "feasible_fraction": float(feasible.float().mean()),
        "feasible": bool(feasible.all()),
        "slack_mean": float(delta.mean()),
        "slack_max": float(delta.max()),
    }
