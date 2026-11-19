"""Discount-rate schedule following MIT-REALM/pncbf's quadcircle recipe."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PNCBFDiscountSchedule:
    """Warmup, horizon growth, then lambda decay; fractions of the optimizer-step budget.

    lambda = -log(1 - 1 / horizon) / dt. The bootstrap weight is upstream's tgt_rhs.
    """

    initial_horizon: float = 20.0
    final_horizon: float = 100.0
    final_lambda: float = 0.05
    warmup_fraction: float = 100 / 320
    horizon_end_fraction: float = 170 / 320
    decay_end_fraction: float = 270 / 320
    initial_bootstrap_weight: float = 1.0
    final_bootstrap_weight: float = 1.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("PNCBF discount schedule values must be finite")
        if not 1 < self.initial_horizon <= self.final_horizon or self.final_lambda < 0:
            raise ValueError("PNCBF discount horizons must increase from >1 and final_lambda must be non-negative")
        if not 0 <= self.warmup_fraction < self.horizon_end_fraction < self.decay_end_fraction <= 1:
            raise ValueError("PNCBF discount stage fractions must be ordered in [0, 1]")
        if not (0 <= self.initial_bootstrap_weight <= 1 and 0 <= self.final_bootstrap_weight <= 1):
            raise ValueError("PNCBF bootstrap weights must lie in [0, 1]")

    def value(self, optimizer_steps: int, total_optimizer_steps: int, step_dt: float) -> float:
        if total_optimizer_steps < 1 or optimizer_steps < 0 or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("PNCBF discount schedule requires a positive budget/dt and non-negative step")
        final_horizon_rate = -math.log1p(-1.0 / self.final_horizon) / step_dt
        if self.final_lambda > final_horizon_rate:
            raise ValueError("PNCBF final_lambda must not exceed the rate at final_horizon")
        progress = optimizer_steps / total_optimizer_steps
        if progress <= self.horizon_end_fraction:
            fraction = max(0.0, (progress - self.warmup_fraction) /
                           (self.horizon_end_fraction - self.warmup_fraction))
            horizon = self.initial_horizon + fraction * (self.final_horizon - self.initial_horizon)
            return -math.log1p(-1.0 / horizon) / step_dt
        fraction = min(1.0, (progress - self.horizon_end_fraction) /
                       (self.decay_end_fraction - self.horizon_end_fraction))
        return final_horizon_rate + fraction * (self.final_lambda - final_horizon_rate)

    def bootstrap_weight(self, optimizer_steps: int, total_optimizer_steps: int) -> float:
        if total_optimizer_steps < 1 or optimizer_steps < 0:
            raise ValueError("PNCBF bootstrap weight requires a positive budget and non-negative step")
        progress = optimizer_steps / total_optimizer_steps
        fraction = min(1.0, max(0.0, (progress - self.warmup_fraction) /
                                (self.horizon_end_fraction - self.warmup_fraction)))
        return self.initial_bootstrap_weight + fraction * (self.final_bootstrap_weight - self.initial_bootstrap_weight)
