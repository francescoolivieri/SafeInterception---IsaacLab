"""Arena violation masks used by the M4 penalty, termination and evaluation scoring."""
from __future__ import annotations

import torch


def arena_limit_masks(
    position: torch.Tensor,
    arena_min: torch.Tensor,
    arena_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Altitude and xy violations against the given scoring box."""
    altitude = (position[:, 2] < arena_min[2]) | (position[:, 2] > arena_max[2])
    xy = torch.any(
        (position[:, :2] < arena_min[:2]) | (position[:, :2] > arena_max[:2]),
        dim=-1,
    )
    return altitude, xy
