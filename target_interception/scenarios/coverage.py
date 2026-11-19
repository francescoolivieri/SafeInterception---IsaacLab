"""Balanced coverage requests for randomized training pools.

Each row gets one bucket per scenario dimension; marginal quotas are exact to one row,
then shuffled. Generation retries a request until it is filled.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


MOTION_MODES = (
    "hover",
    "wander",
)
INITIAL_MOTION = ("toward", "transverse", "away")
TARGET_SPEED_BIN_EDGES = (0.4, 0.8, 1.2, 1.5, 1.8)
TARGET_SPEED_WEIGHTS = {
    2: (0.30, 0.30, 0.20, 0.20),
    3: (0.10, 0.20, 0.35, 0.35),
    4: (0.10, 0.20, 0.30, 0.40),
}


@dataclass(frozen=True)
class CoverageRequest:
    static_count: int
    dynamic_count: int
    distance_bin: int
    interceptor_altitude_bin: int
    target_altitude_bin: int
    speed_bin: int
    initial_motion: str
    motion_mode: str
    dynamic_length_bins: tuple[int, ...] = ()

    def quota_values(self) -> dict[str, str | int]:
        return {
            "obstacle_counts": f"{self.static_count}+{self.dynamic_count}",
            "start_distance": self.distance_bin,
            "interceptor_altitude": self.interceptor_altitude_bin,
            "target_altitude": self.target_altitude_bin,
            "target_speed": self.speed_bin,
            "initial_motion": self.initial_motion,
            "motion_mode": self.motion_mode,
        }


def obstacle_count_choices(env, phase: int) -> tuple[tuple[int, int], ...]:
    """Feasible count pairs for one curriculum phase."""
    from target_interception.scenarios.config import phase_obstacle_count_choices

    return tuple(phase_obstacle_count_choices(env, phase)) or ((0, 0),)


def _balanced_index(row: int, size: int, buckets: int, stride: int) -> int:
    if buckets <= 1:
        return 0
    # Multiplication by an odd stride decorrelates dimensions while each
    # marginal still receives floor/ceil(size / buckets) rows.
    return ((row * stride) % size) * buckets // size


def _weighted_values(size: int, values: tuple, weights: tuple[float, ...]) -> list:
    """Return an exact largest-remainder categorical schedule."""
    if size <= 0:
        return []
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must have the same non-zero length")
    weight_sum = sum(max(0.0, float(weight)) for weight in weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one categorical weight must be positive")
    raw = [size * max(0.0, float(weight)) / weight_sum for weight in weights]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(len(values)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    out = []
    for value, count in zip(values, counts):
        out.extend([value] * count)
    if len(out) > 1:
        permutation = torch.randperm(len(out)).tolist()
        out = [out[index] for index in permutation]
    return out


def build_coverage_schedule(
    env, phase: int, size: int, *,
    obstacle_choices: tuple[tuple[int, int], ...] | None = None,
) -> list[CoverageRequest]:
    """Shuffled quota schedule for one pool bank."""
    size = max(1, int(size))
    counts = obstacle_choices or obstacle_count_choices(env, phase)
    altitude_bins = max(1, int(getattr(env.cfg.scenario, "altitude_bin_count", 3)))
    modes = tuple(getattr(env.cfg.scenario, "target_motion_modes", MOTION_MODES)) or MOTION_MODES
    moving_modes = tuple(mode for mode in modes if mode != "hover") or ("wander",)

    from . import penalty_curriculum

    staged = penalty_curriculum.enabled(env.cfg)
    stationary = penalty_curriculum.BANKS[phase - 1].speed is None if staged else phase <= 1
    if stationary:
        motion_modes = ["hover"] * size
        initial_motion = ["hover"] * size
    else:
        motion_modes = _weighted_values(size, moving_modes, (1.0,) * len(moving_modes))
        # target.py samples the direction; close starts only exclude "toward".
        initial_motion = ["random"] * size

    if stationary:
        speed_bins = [0] * size  # stationary target; no requested cruise speed
    else:
        # Bin widths differ; proportional weights give uniform requested speeds. The last two
        # staged banks use the speed mix of the final phase of the four-phase curriculum.
        weights = ((0.4, 0.4, 0.3, 0.3) if phase < 7 else TARGET_SPEED_WEIGHTS[4]) if staged else TARGET_SPEED_WEIGHTS[phase]
        speed_bins = _weighted_values(size, (0, 1, 2, 3), weights)

    obstacle_counts = _weighted_values(size, counts, (1.0,) * len(counts))

    total_dynamic = sum(dynamic for _, dynamic in obstacle_counts)
    length_bins = tuple(getattr(env.cfg.scenario, "dynamic_rail_length_bins", ((0.8, 1.8),)))
    length_weights = tuple(
        getattr(env.cfg.scenario, "dynamic_rail_length_weights", (1.0,) * len(length_bins))
    )
    flat_length_bins = _weighted_values(total_dynamic, tuple(range(len(length_bins))), length_weights)
    row_length_bins: list[tuple[int, ...]] = []
    offset = 0
    for _, dynamic_count in obstacle_counts:
        row_length_bins.append(tuple(flat_length_bins[offset : offset + dynamic_count]))
        offset += dynamic_count

    requests = []
    for row in range(size):
        mode = motion_modes[row]
        initial = initial_motion[row]
        static_count, dynamic_count = obstacle_counts[row]
        requests.append(
            CoverageRequest(
                static_count=static_count,
                dynamic_count=dynamic_count,
                distance_bin=0 if staged and penalty_curriculum.BANKS[phase - 1].distance[1] is not None else -1,
                interceptor_altitude_bin=_balanced_index(row, size, altitude_bins, 11),
                target_altitude_bin=_balanced_index(row, size, altitude_bins, 13),
                speed_bin=speed_bins[row],
                initial_motion=initial,
                motion_mode=mode,
                dynamic_length_bins=row_length_bins[row],
            )
        )

    order = torch.randperm(size).tolist()
    return [requests[index] for index in order]


def quota_histogram(requests: list[CoverageRequest]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for request in requests:
        for name, value in request.quota_values().items():
            values = out.setdefault(name, {})
            key = str(value)
            values[key] = values.get(key, 0) + 1
    return out
