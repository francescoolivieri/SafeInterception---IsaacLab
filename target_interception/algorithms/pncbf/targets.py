"""Full-rollout PNCBF targets, as upstream compute_all_disc_avoid_terms (MIT-REALM/pncbf, df188fac).

The endpoint bootstrap has unit weight; the trainer applies tgt_rhs.
"""
from __future__ import annotations

import math

import torch


def pncbf_value_targets(
    constraints: torch.Tensor,
    bootstrap: torch.Tensor,
    done: torch.Tensor,
    *,
    discount_lambda: float,
    step_dt: float,
    terminal_constraints: torch.Tensor | None = None,
) -> torch.Tensor:
    """[T, B, H] targets from constraints [T+1, B, H], bootstrap [B, H] and done [T, B].

    Done rows end at terminal_constraints [T, B, H] and never bootstrap across the reset.
    """
    if constraints.ndim != 3 or constraints.shape[0] < 2 or bootstrap.shape != constraints.shape[1:]:
        raise ValueError("PNCBF constraints must be [T+1, B, H] and bootstrap [B, H], with T >= 1")
    if done.shape != constraints[:-1].shape[:2]:
        raise ValueError("PNCBF done must be [T, B]")
    if terminal_constraints is None:
        if bool(done.any()):
            raise ValueError("PNCBF completed transitions require pre-reset terminal constraints")
        terminal_constraints = constraints[1:]
    elif terminal_constraints.shape != constraints[:-1].shape:
        raise ValueError("PNCBF terminal constraints must be [T, B, H]")
    if not math.isfinite(discount_lambda) or discount_lambda < 0 or not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("PNCBF requires finite discount_lambda >= 0 and step_dt > 0")

    a = discount_lambda * step_dt
    gamma = math.exp(-a)
    # Step integral of lambda exp(-lambda s) h(s) with h linear; Taylor terms near lambda = 0.
    if a < 1e-4:
        left = a / 2 - a * a / 6 + a**3 / 24
        right = a / 2 - a * a / 3 + a**3 / 8
    else:
        coeff = -math.expm1(-a) / a
        left, right = 1.0 - coeff, coeff - gamma

    future = torch.maximum(bootstrap, constraints[-1])
    targets = torch.empty_like(constraints[:-1])
    done = done.to(device=constraints.device, dtype=torch.bool)
    for t in reversed(range(len(targets))):
        endpoint = torch.where(done[t, :, None], terminal_constraints[t], constraints[t + 1])
        continuation = torch.where(done[t, :, None], endpoint, future)
        integral = left * constraints[t] + right * endpoint
        future = torch.maximum(constraints[t], integral + gamma * continuation)
        targets[t] = future
    return targets
