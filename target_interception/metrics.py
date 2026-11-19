"""Rollout metrics for comparing interception policies; benchmark.py reads the summary.

Safety margin: signed clearance from the drone collision body to the nearest obstacle
(0 contact, < 0 collision). Average speed is path length / episode duration, over captures.
"""
from __future__ import annotations

import math
import statistics

import torch

NEAR_MISS_M = 0.05  # an episode is a near miss when its minimum clearance drops below this


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score 95% interval for a binomial rate."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


class RolloutMetrics:
    """Accumulate per-episode interception and safety metrics across parallel envs."""

    def __init__(
        self,
        num_envs: int,
        *,
        capture_status: int = 1,
        collision_status: int = 6,
        safety_statuses: tuple[int, ...] | None = None,
        timeout_status: int = 4,
        step_dt: float = 1.0,
        horizon_s: float | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.capture_status = int(capture_status)
        self.collision_status = int(collision_status)
        self.safety_statuses = frozenset(
            int(status) for status in (safety_statuses if safety_statuses is not None else (collision_status,))
        )
        self.timeout_status = int(timeout_status)
        self.step_dt = float(step_dt)
        self.horizon_s = horizon_s
        self.trial_indices: list[int] = []

        self._path = [0.0] * self.num_envs
        self._steps = [0] * self.num_envs
        self._min_sep = [math.inf] * self.num_envs
        self._collided = [False] * self.num_envs
        self._last_pos: torch.Tensor | None = None

        # Finalized episode records (all index-aligned: one entry per finished episode).
        self.captures: list[bool] = []
        self.collisions: list[bool] = []
        self.statuses: list[int] = []
        self.min_separations: list[float] = []
        self.path_lengths: list[float] = []
        self.episode_lengths: list[int] = []
        self.groups: list[int] = []  # per-episode group tag (eval tier code; 0 if untagged)
        self.pursuer_speed_mean: list[float] = []  # path length / episode duration

        # Global filter-intervention accumulators (averaged over steps).
        self._filter_steps = 0
        self._intervention_sum = 0.0
        self._intervened_fraction_sum = 0.0
        self._filter_feasible_fraction_sum = 0.0
        self._filter_slack_sum = 0.0
        self._filter_qp_violation_max = 0.0

    def update(
        self,
        *,
        agent_pos: torch.Tensor,
        nearest_obstacle_dist: torch.Tensor,
        done: torch.Tensor,
        status: torch.Tensor,
        collision: torch.Tensor | None = None,
        filter_info: dict | None = None,
        group: torch.Tensor | None = None,
        step_distance: torch.Tensor | None = None,
    ) -> None:
        agent_pos = agent_pos.detach()
        nearest = nearest_obstacle_dist.detach().reshape(-1)
        done = done.detach().reshape(-1).bool()
        status = status.detach().reshape(-1).to(torch.int64)
        coll = None if collision is None else collision.detach().reshape(-1).bool()
        # group: scenario code of the episode that just ended (kept across the auto-reset).
        # None skips the grouped breakdown.
        ph = None if group is None else group.detach().reshape(-1).to(torch.int64)

        if self._last_pos is None:
            self._last_pos = agent_pos.clone()

        if step_distance is None:
            step_dist = torch.linalg.vector_norm(agent_pos - self._last_pos, dim=-1)
        else:
            step_dist = step_distance.detach().reshape(-1)
        self._last_pos = agent_pos.clone()
        for e in range(self.num_envs):
            self._steps[e] += 1
            self._min_sep[e] = min(self._min_sep[e], float(nearest[e].item()))
            if step_distance is not None or not bool(done[e]):
                self._path[e] += float(step_dist[e].item())
            if coll is not None and bool(coll[e]):
                self._collided[e] = True
            # Evaluation defines a collision outcome as any safety violation:
            # floor/ceiling, XY boundary, or obstacle contact.
            if int(status[e].item()) in self.safety_statuses:
                self._collided[e] = True

            if bool(done[e]):
                captured = int(status[e].item()) == self.capture_status
                self.captures.append(captured)
                self.collisions.append(self._collided[e])
                self.statuses.append(int(status[e].item()))
                self.min_separations.append(self._min_sep[e])
                self.path_lengths.append(self._path[e])
                self.episode_lengths.append(self._steps[e])
                self.groups.append(int(ph[e].item()) if ph is not None else 0)
                self.pursuer_speed_mean.append(self._path[e] / (self._steps[e] * self.step_dt))
                self._path[e] = 0.0
                self._steps[e] = 0
                self._min_sep[e] = math.inf
                self._collided[e] = False

        if filter_info:
            self._filter_steps += 1
            self._intervention_sum += float(filter_info.get("intervention_mean", 0.0))
            self._intervened_fraction_sum += float(filter_info.get("intervened_fraction", 0.0))
            self._filter_feasible_fraction_sum += float(filter_info.get("feasible_fraction", 1.0))
            self._filter_slack_sum += float(filter_info.get("slack_mean", 0.0))
            self._filter_qp_violation_max = max(
                self._filter_qp_violation_max, float(filter_info.get("qp_violation_max", 0.0))
            )

    @property
    def num_episodes(self) -> int:
        return len(self.captures)

    def _group_summary(self, idxs: list[int]) -> dict:
        """Summary over a subset of finished episodes; continuous metrics also get <name>_std."""
        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else None

        def _std(xs):
            return float(statistics.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)

        captures = [self.captures[i] for i in idxs]
        collisions = [self.collisions[i] for i in idxs]
        statuses = [self.statuses[i] for i in idxs]
        min_seps = [self.min_separations[i] for i in idxs]
        path_lengths = [self.path_lengths[i] for i in idxs]
        episode_lengths = [self.episode_lengths[i] for i in idxs]
        avg_speeds = [self.pursuer_speed_mean[i] for i in idxs if self.captures[i]]
        # Time-to-capture is only defined for captured episodes (TTC == episode length there).
        ttc = [self.episode_lengths[i] for i in idxs if self.captures[i]]
        result = {
            "num_episodes": len(idxs),
            "capture_rate": _mean([float(c) for c in captures]),
            "collision_rate": _mean([float(c) for c in collisions]),
            # Every safety status already sets _collided, so these two rates are the same number.
            "safety_violation_rate": _mean([float(c) for c in collisions]),
            "time_to_capture_mean": _mean(ttc),
            "time_to_capture_std": _std(ttc),
            "time_to_capture_mean_s": _mean([x * self.step_dt for x in ttc]),
            "time_to_capture_std_s": _std([x * self.step_dt for x in ttc]),
            "min_safety_margin_mean": _mean(min_seps),
            "min_safety_margin_std": _std(min_seps),
            "min_safety_margin_min": float(min(min_seps)) if min_seps else None,
            "path_length_mean": _mean(path_lengths),
            "path_length_std": _std(path_lengths),
            "episode_length_mean": _mean(episode_lengths),
            "episode_length_mean_s": _mean([x * self.step_dt for x in episode_lengths]),
            "num_captured": int(sum(captures)),
            "num_collided": int(sum(collisions)),
            "num_safety_violations": int(sum(collisions)),
            "num_timed_out": int(sum(status == self.timeout_status for status in statuses)),
            "capture_rate_ci95": wilson_interval(int(sum(captures)), len(idxs)),
            "safety_violation_rate_ci95": wilson_interval(int(sum(collisions)), len(idxs)),
            "near_miss_rate": _mean([float(m < NEAR_MISS_M) for m in min_seps]),
            # Captured episodes only: how fast the policy flies when it succeeds.
            "pursuer_speed_mean": _mean(avg_speeds),
            "pursuer_speed_std": _std(avg_speeds),
        }
        if self.horizon_s is not None:
            penalized = [self.episode_lengths[i] * self.step_dt if self.captures[i]
                         else self.horizon_s for i in idxs]
            result["penalized_capture_time_mean_s"] = _mean(penalized)
        return result

    def summary_for_indices(self, indices: list[int]) -> dict:
        return self._group_summary(indices) if indices else {"num_episodes": 0}

    def summary_for_groups(self, codes) -> dict:
        """Metric summary over the episodes whose group tag is in codes."""
        wanted = set(int(c) for c in codes)
        idxs = [i for i, g in enumerate(self.groups) if g in wanted]
        return self._group_summary(idxs) if idxs else {"num_episodes": 0}

    def summary(self) -> dict:
        """Summary over all finished episodes; summary_for_groups gives the breakdown."""
        n = self.num_episodes
        if n == 0:
            return {"num_episodes": 0}
        summary = self._group_summary(list(range(n)))
        if self._filter_steps > 0:
            summary["filter_intervention_mean"] = self._intervention_sum / self._filter_steps
            summary["filter_intervened_fraction"] = self._intervened_fraction_sum / self._filter_steps
            summary["filter_feasible_fraction"] = self._filter_feasible_fraction_sum / self._filter_steps
            summary["filter_slack_mean"] = self._filter_slack_sum / self._filter_steps
            summary["filter_qp_violation_max"] = self._filter_qp_violation_max
        return summary
