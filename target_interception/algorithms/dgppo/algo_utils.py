"""DG-PPO update math: value targets, CBF advantage, PPO surrogate, optimizer step.

Shapes: b envs, T rollout steps, h constraint heads (one: the worst constraint).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F



def constraint_targets(bTh_h, bTp1h_Vh, bT_done, *, gamma: float, gae_lambda: float) -> torch.Tensor:
    """GAE targets Q_h of the discounted maximum future constraint, anchored to the worst head."""
    anchor = bTh_h.max(dim=-1, keepdim=True).values.expand_as(bTh_h)
    return avoid_value_targets(bTh_h, bTp1h_Vh, gamma=gamma, gae_lambda=gae_lambda,
                               continue_mask=~bT_done, anchor=anchor)


def reward_targets(bT_l, bTp1_Vl, bT_done, *, gamma: float, gae_lambda: float) -> torch.Tensor:
    """GAE targets Q_l of the discounted cost l = -reward (an episode end stops the bootstrap)."""
    B, T = bT_l.shape
    continues = (~bT_done).to(bT_l.dtype)
    Ql = torch.zeros_like(bT_l)
    time_ids = torch.arange(T + 1, device=bT_l.device)
    # rows[:, k] is the (k+1)-step return; row 0 is bootstrapped by the last value.
    rows = torch.zeros((B, T + 1), device=bT_l.device, dtype=bT_l.dtype)
    rows[:, 0] = bTp1_Vl[:, -1]
    gae_coeffs = torch.zeros((T + 1,), device=bT_l.device, dtype=bT_l.dtype)
    gae_coeffs[0] = 1.0
    for step, t in enumerate(reversed(range(T))):
        mask = (time_ids <= step).to(bT_l.dtype)[None, :]
        rows = mask * (bT_l[:, t][:, None] + gamma * continues[:, t][:, None] * rows)
        Ql[:, t] = torch.einsum("bt,t->b", rows, gae_coeffs)
        rows[:, step + 1] = bTp1_Vl[:, t]
        gae_coeffs = torch.roll(gae_coeffs, shifts=1, dims=0)
        gae_coeffs[0] = gae_lambda ** (step + 1)
        gae_coeffs[1] = (gae_lambda**step) * (1.0 - gae_lambda)
    return Ql


def scheduled_cbf_scale(*, cbf_weight: float, timestep: int, timesteps: int, schedule_start_step: int,
                        fractions: Sequence[float], multipliers: Sequence[float]) -> float:
    """CBF weight: multipliers[0] until schedule_start_step, then one stage per fraction."""
    total = max(1, int(timesteps))
    step = max(0, min(int(timestep) + 1, total))
    start = max(0, min(int(schedule_start_step), total))
    if step <= start or total <= start:
        return float(cbf_weight) * float(multipliers[0])
    progress = (step - start) / (total - start)
    fractions = [max(0.0, float(f)) for f in fractions]
    boundary = 0.0
    for fraction, multiplier in zip(fractions, multipliers, strict=True):
        boundary += fraction / sum(fractions)
        if progress <= boundary:
            return float(cbf_weight) * float(multiplier)
    return float(cbf_weight) * float(multipliers[-1])


def compute_cbf_advantages(bT_Ql, bT_Vl, bTh_Vh, bTp1h_Vh, bT_done, *, alpha: float, cbf_eps: float,
                           cbf_scale: float, dt: float) -> dict[str, torch.Tensor]:
    """DG-PPO advantage (sign for PPO maximization).

    The task advantage Q_l - V_l, standardized over time, is used only where the discrete
    CBF condition (V_h(t+1) - V_h(t)) / dt + alpha V_h(t) <= 0 holds for every head;
    elsewhere it is replaced by the violation max(cond + eps, 0) times cbf_scale. Done
    steps use V_h(t) as the endpoint, so the difference never crosses a reset.
    """
    bT_Al = bT_Ql - bT_Vl
    bT_Al = (bT_Al - bT_Al.mean(dim=1, keepdim=True)) / (bT_Al.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
    bTh_next_Vh = torch.where(bT_done[:, :, None], bTh_Vh, bTp1h_Vh[:, 1:])
    bTh_cbf = (bTh_next_Vh - bTh_Vh) / dt + alpha * bTh_Vh
    bT_is_safe = (bTh_cbf <= 0).all(dim=-1)
    bT_penalty = torch.clamp(bTh_cbf + cbf_eps, min=0.0).max(dim=-1).values * cbf_scale
    bT_A = torch.where(bT_is_safe, bT_Al, torch.zeros_like(bT_Al)) + bT_penalty
    return {
        "bT_A": -bT_A,
        "bT_is_safe": bT_is_safe,
        "bT_cbf_penalty": bT_penalty,
        "bT_Vh_worst": bTh_Vh.max(dim=-1).values,
    }


def compute_policy_surrogate(ratio: torch.Tensor, advantages: torch.Tensor, clip_eps: float) -> dict[str, torch.Tensor]:
    """Clipped PPO surrogate (pessimistic max over clipped and unclipped)."""
    loss1 = -ratio * advantages
    loss2 = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return {"loss_policy": torch.maximum(loss1, loss2).mean(), "clip_frac": (loss2 > loss1).float().mean()}


def value_l2_loss(prediction: torch.Tensor, target: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Optax l2_loss(...).mean(): 0.5 * mean squared error."""
    return scale * 0.5 * F.mse_loss(prediction, target)


def apply_optimizer_update(*, optimizer, loss, parameters: Iterable[torch.nn.Parameter], grad_clip: float):
    """Backward, average across ranks, clip and step. A non-finite gradient norm skips the step."""
    from .distributed import average_gradients

    params = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    average_gradients(params)
    grads = [param.grad.detach() for param in params if param.grad is not None]
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    if not bool(torch.isfinite(grad_norm).item()):
        optimizer.zero_grad(set_to_none=True)
        return grad_norm
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    return grad_norm


def avoid_value_targets(
    constraints: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    continue_mask: torch.Tensor,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lambda-return targets [B, T, ...] of the discounted maximum future constraint.

    values [B, T + 1, ...] bootstrap the rollout end; continue_mask [B, T] is 0 at episode ends,
    where the target is h. anchor (default: constraints) enters the (1 - gamma) term.
    """
    B, T = constraints.shape[:2]
    tail = constraints.shape[2:]
    device, dtype = constraints.device, constraints.dtype
    h = constraints.reshape(B, T, -1)
    V = values.reshape(B, T + 1, -1)
    anchor = h if anchor is None else anchor.reshape(B, T, -1)
    step_continue = continue_mask.to(device=device, dtype=dtype).reshape(B, T)

    targets = torch.empty_like(h)
    time_ids = torch.arange(T + 1, device=device)
    # rows[:, k] is the (k+1)-step backup; row 0 is bootstrapped by values[:, T].
    rows = torch.zeros((B, T + 1, h.shape[-1]), device=device, dtype=dtype)
    rows[:, 0] = V[:, -1]
    gae_coeffs = torch.zeros((T + 1,), device=device, dtype=dtype)
    gae_coeffs[0] = 1.0
    for step, t in enumerate(reversed(range(T))):
        mask = (time_ids <= step).to(dtype)[None, :, None]
        continues = step_continue[:, t, None, None]
        discount = gamma * continues
        step_anchor = torch.where(continues > 0, anchor[:, t, None], h[:, t, None])
        rows = mask * torch.maximum(h[:, t, None], (1.0 - discount) * step_anchor + discount * rows)
        targets[:, t] = torch.einsum("bkf,k->bf", rows, gae_coeffs)
        rows[:, step + 1] = V[:, t]
        gae_coeffs = torch.roll(gae_coeffs, shifts=1, dims=0)
        gae_coeffs[0] = gae_lambda ** (step + 1)
        gae_coeffs[1] = (gae_lambda**step) * (1.0 - gae_lambda)
    return targets.reshape(B, T, *tail)
