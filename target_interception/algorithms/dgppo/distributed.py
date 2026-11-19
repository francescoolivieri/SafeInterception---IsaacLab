"""Broadcast and gradient averaging for the DG-PPO modules.

skrl sets up the process group from torchrun's environment, but DG-PPO's networks are
plain nn.Modules rather than skrl Models, so they get their own broadcast/reduce here.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@torch.no_grad()
def broadcast_modules(modules: Iterable[torch.nn.Module], *, source: int = 0) -> None:
    """Make parameters and buffers identical to the source rank."""
    if not is_distributed():
        return
    for module in modules:
        for tensor in module.state_dict().values():
            torch.distributed.broadcast(tensor, src=source)


@torch.no_grad()
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Average dense gradients across ranks in a fixed parameter order.

    Missing local gradients participate as zeros. This keeps the collective
    shape/order stable even if a conditional model path leaves a parameter
    unused on one rank.
    """
    if not is_distributed():
        return

    params = [parameter for parameter in parameters if parameter.requires_grad]
    groups: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in params:
        if parameter.grad is not None and parameter.grad.is_sparse:
            raise RuntimeError("DG-PPO distributed training does not support sparse gradients")
        groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)

    world_size = torch.distributed.get_world_size()
    for grouped_params in groups.values():
        flat = torch.cat(
            [
                parameter.grad.detach().reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
                for parameter in grouped_params
            ]
        )
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)

        offset = 0
        for parameter in grouped_params:
            averaged = flat[offset : offset + parameter.numel()].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = averaged.clone()
            else:
                parameter.grad.copy_(averaged)
            offset += parameter.numel()
