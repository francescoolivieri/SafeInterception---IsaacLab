"""M1 nominal pursuit: LQR on the single integrator dp_p/dt = u."""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_continuous_are


class LQRPursuit:
    """u_LQR = v_e + K_LQR (p_e - p_p), returned as the normalised velocity action u / S.

    K_LQR solves the Riccati equation of the error dynamics de/dt = u - v_e, e = p_p - p_e,
    so it is the gain of a static target and v_e enters as a feedforward term.
    """

    def __init__(self, obs_layout, command_scale, *, q: float = 1.0, r: float = 1.0, device="cpu") -> None:
        if q <= 0.0 or r <= 0.0:
            raise ValueError("q and r must be positive")
        A, B = np.zeros((3, 3)), np.eye(3)
        P = solve_continuous_are(A, B, q * np.eye(3), r * np.eye(3))
        self.K = torch.as_tensor(B.T @ P / r, dtype=torch.float32, device=device)
        self.S = torch.as_tensor(command_scale, dtype=torch.float32, device=device).reshape(-1)[:3]
        self.layout = obs_layout

    def command(self, p_p, p_e, v_e):
        return v_e + (p_e - p_p) @ self.K.T

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        ego, target = obs[:, self.layout.ego], obs[:, self.layout.target]
        p_p, v_p = ego[:, :3], ego[:, 3:6]
        p_e = p_p + target[:, :3]  # the observation holds p_e - p_p
        v_e = v_p + target[:, 3:6]  # and v_e - v_p
        u = self.command(p_p, p_e, v_e)
        action = obs.new_zeros(obs.shape[0], 4)  # yaw rate stays zero
        action[:, :3] = (u / self.S).clamp(-1.0, 1.0)
        return action

    def reset(self, env_ids) -> None:
        pass
