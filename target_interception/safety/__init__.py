"""Safety-filter composition for analytic CBF and learned PNCBF."""
from __future__ import annotations

from .cbf_qp import CBFQPFilter

__all__ = ["CBFQPFilter", "make_filter"]


def make_filter(filter_cfg: dict, *, obs_layout, command_scale, device):
    filter_type = filter_cfg.get("type")
    params = dict(filter_cfg.get("params") or {})
    if filter_type == "cbf_qp":
        result = CBFQPFilter(
            obs_layout,
            d_safe=float(params.pop("d_safe", 0.35)),
            alpha=float(params.pop("alpha", 1.0)),
            command_scale=command_scale,
            device=device,
        )
    elif filter_type == "pncbf":
        # Lazy import avoids making the shared constraint geometry depend on
        # the PNCBF model package during normal task/reward imports.
        from .pncbf import PNCBFFilter

        checkpoint = params.pop("checkpoint", None)
        if not checkpoint:
            raise ValueError("PNCBF filter requires params.checkpoint")
        result = PNCBFFilter(
            obs_layout,
            checkpoint=str(checkpoint),
            alpha_safe=float(params.pop("alpha_safe", 2.0)),
            alpha_unsafe=float(params.pop("alpha_unsafe", 2.0)),
            value_shift=float(params.pop("value_shift", 0.01)),
            command_scale=command_scale,
            device=device,
        )
    else:
        raise ValueError(f"unknown filter.type {filter_type!r}; expected 'cbf_qp' or 'pncbf'")
    if params:
        raise ValueError(f"unknown {filter_type} filter parameters: {sorted(params)}")
    return result
