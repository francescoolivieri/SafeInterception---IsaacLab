"""Signed safety constraints shared by all methods, h <= 0 safe.

DG-PPO's V_h and the PNCBF certificate are trained on these; benchmark collisions use the
env's physical geometry.
"""
from __future__ import annotations

import torch


def arena_constraint_value(
    position: torch.Tensor,
    safe_min,
    safe_max,
) -> torch.Tensor:
    """Maximum signed exceedance of an axis-aligned box, with a final head axis."""
    return arena_constraint_components(position, safe_min, safe_max).amax(dim=-1, keepdim=True)


def arena_constraint_components(position: torch.Tensor, safe_min, safe_max) -> torch.Tensor:
    """Six signed box-plane costs, ordered lower xyz then upper xyz."""
    low = torch.as_tensor(safe_min, device=position.device, dtype=position.dtype)
    high = torch.as_tensor(safe_max, device=position.device, dtype=position.dtype)
    return torch.cat((low - position, position - high), dim=-1)


def obstacle_constraint_components(relative_position: torch.Tensor, clearance: float) -> torch.Tensor:
    """One signed horizontal-clearance cost per obstacle."""
    distance = torch.linalg.vector_norm(relative_position[..., :2], dim=-1)
    return float(clearance) - distance


def safety_constraint_value(position: torch.Tensor, arena_min, arena_max, obstacles=()) -> torch.Tensor:
    """Worst of the arena and every active obstacle, [N, 1].

    obstacles holds (centers_xy [N, M, 2], active [N, M], contact_distance) groups, one per
    obstacle type. Pillars span the flight volume, so only horizontal distance counts.
    """
    arena = arena_constraint_value(position, arena_min, arena_max)
    return torch.maximum(arena, _nearest_obstacle_value(position, obstacles))


def _nearest_obstacle_value(position: torch.Tensor, obstacles) -> torch.Tensor:
    """Worst active obstacle constraint [N, 1] (-inf without one)."""
    costs, valid = [], []
    for centers, active, contact in obstacles:
        if centers.shape[1]:
            costs.append(obstacle_constraint_components(centers - position[:, None, :2], contact))
            valid.append(active.to(torch.bool))
    if not costs:
        return position.new_full((position.shape[0], 1), -torch.inf)
    cost, valid = torch.cat(costs, dim=1), torch.cat(valid, dim=1)
    return torch.where(valid, cost, torch.full_like(cost, -torch.inf)).amax(dim=-1, keepdim=True)
