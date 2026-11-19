"""MLP building block."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected layers with orthogonal init and optional per-layer LayerNorm.

    The activation (after the LayerNorm) follows every layer; act_final=False leaves the
    last layer linear.
    """

    def __init__(
        self,
        hid_sizes: Sequence[int],
        in_dim: int,
        act: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
        act_final: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.hid_sizes = tuple(hid_sizes)
        self.act = act
        self.act_final = act_final
        sizes = (in_dim, *self.hid_sizes)
        self.layers = nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:]))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(h) for h in self.hid_sizes]) if use_layernorm else None
        for layer in self.layers:  # after all layers exist, to keep the seeded init order
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.act_final:
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
                x = self.act(x)
        return x
