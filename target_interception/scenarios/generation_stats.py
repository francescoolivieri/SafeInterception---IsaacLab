"""Low-overhead, optional diagnostics for offline scenario generation."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator, Mapping


@dataclass
class GenerationStats:
    """Process-local counters and timers, merged by the parent.

    Holds no random state, so it never changes seeded results.
    """

    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def count(self, reason: str, amount: int = 1) -> None:
        self.counts[str(reason)] += int(amount)

    def add_time(self, stage: str, elapsed: float) -> None:
        self.seconds[str(stage)] += max(0.0, float(elapsed))

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(stage, time.perf_counter() - start)

    def merge(self, other: "GenerationStats | Mapping[str, Mapping[str, float]]") -> None:
        payload = other.to_dict() if isinstance(other, GenerationStats) else other
        for key, value in payload.get("counts", {}).items():
            self.counts[str(key)] += int(value)
        for key, value in payload.get("seconds", {}).items():
            self.seconds[str(key)] += float(value)

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "seconds": dict(sorted(self.seconds.items())),
        }


def generation_stats(env) -> GenerationStats | None:
    """Return diagnostics attached to an offline context, if enabled."""
    return getattr(env, "generation_stats", None)


def count(env, reason: str, amount: int = 1) -> None:
    stats = generation_stats(env)
    if stats is not None:
        stats.count(reason, amount)


@contextmanager
def time_stage(env, stage: str) -> Iterator[None]:
    stats = generation_stats(env)
    if stats is None:
        yield
    else:
        with stats.time(stage):
            yield
