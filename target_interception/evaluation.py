"""Suite rollout, trajectory logging and metric summaries for scripts/evaluate.py."""
from __future__ import annotations


import torch

from target_interception.metrics import RolloutMetrics
from target_interception.scenarios.scenario_defs import EVALUATION_SCENARIO_CODES, FAMILIES, SCENARIO_LABELS


class TrajectoryLog:
    """Every episode for the qualitative trajectory figures (episodes.pt).

    Positions come from the last-step snapshot, which survives the auto-reset; obstacles
    are captured at episode start because a reset overwrites them.
    """

    def __init__(self, base_env) -> None:
        self.base_env = base_env
        n = int(base_env.num_envs)
        self._pursuer: list[list] = [[] for _ in range(n)]
        self._evader: list[list] = [[] for _ in range(n)]
        self._pursuer_speed: list[list] = [[] for _ in range(n)]
        self._evader_speed: list[list] = [[] for _ in range(n)]
        self._margin: list[list] = [[] for _ in range(n)]
        self._obstacles: list[dict | None] = [None] * n
        self._min_margin = [float("inf")] * n
        self._collided = [False] * n
        self.episodes: list[dict] = []
        self._counts: dict[int, int] = {}
        self._capture_obstacles(range(n))

    def _capture_obstacles(self, env_ids) -> None:
        """Snapshot the (immutable-within-episode) obstacle geometry for these envs."""
        env = self.base_env
        for e in env_ids:
            e = int(e)
            entry: dict = {"static_xy": [], "dynamic_paths": []}
            if getattr(env.cfg, "enable_pillars", False) and env._max_static_obstacles > 0:
                active = env._static_obstacle_active[e]
                entry["static_xy"] = env._static_obstacle_positions_xy[e][active].detach().cpu().tolist()
                entry["pillar_radius"] = float(env.cfg.pillar_radius)
            if env._max_dynamic_obstacles > 0:
                active = env._dynamic_obstacle_active[e]
                wp = env._dynamic_obstacle_waypoints[e]  # [num_waypoints, max_dynamic, 3]
                entry["dynamic_paths"] = [
                    wp[:, slot, :2].detach().cpu().tolist() for slot in range(wp.shape[1]) if bool(active[slot])
                ]
                entry["dynamic_radius"] = float(env.cfg.pillar_radius)
            self._obstacles[e] = entry

    def seed(self, env_ids, pos: torch.Tensor, ref: torch.Tensor) -> None:
        """Start newly reset episodes with their actual initial positions."""
        for e in env_ids:
            e = int(e)
            self._pursuer[e] = [[float(pos[e, 0]), float(pos[e, 1])]]
            self._evader[e] = [[float(ref[e, 0]), float(ref[e, 1])]]
            self._pursuer_speed[e] = []
            self._evader_speed[e] = []
            self._margin[e] = []
            self._min_margin[e] = float("inf")
            self._collided[e] = False

    def record_step(self, snapshot: dict) -> None:
        pos = snapshot["pos_local"]
        ref = snapshot["ref_pos"]
        margin = snapshot["minimum_safety_margin"]
        # Planar speeds, matching the logged XY trajectories.
        pursuer_speed = torch.linalg.vector_norm(snapshot["vel_world"][:, :2], dim=-1)
        evader_speed = torch.linalg.vector_norm(snapshot["evader_vel"][:, :2], dim=-1)
        for e in range(len(self._pursuer)):
            self._pursuer[e].append([float(pos[e, 0]), float(pos[e, 1])])
            self._evader[e].append([float(ref[e, 0]), float(ref[e, 1])])
            self._pursuer_speed[e].append(float(pursuer_speed[e]))
            self._evader_speed[e].append(float(evader_speed[e]))
            self._margin[e].append(float(margin[e]))
            self._min_margin[e] = min(self._min_margin[e], float(margin[e]))
            self._collided[e] = self._collided[e] or bool(snapshot["safety_violation"][e])

    def finish(self, env_id: int, tier: int, status: int) -> None:
        """Finalize a just-ended episode and snapshot the auto-reset episode's obstacles."""
        e = int(env_id)
        tier = int(tier)
        if tier > 0:
            scenario_index = self._counts.get(tier, 0) + 1
            self.episodes.append({
                "tier": tier,
                "scenario_index": scenario_index,
                "status": int(status),
                "collided": self._collided[e],
                "duration_steps": max(0, len(self._pursuer[e]) - 1),
                "min_safety_margin": self._min_margin[e],
                "pursuer_xy": list(self._pursuer[e]),
                "evader_xy": list(self._evader[e]),
                "pursuer_speed": list(self._pursuer_speed[e]),
                "evader_speed": list(self._evader_speed[e]),
                "safety_margin": list(self._margin[e]),
                "obstacles": self._obstacles[e],
            })
            self._counts[tier] = scenario_index
        self._pursuer[e] = []
        self._evader[e] = []
        self._pursuer_speed[e] = []
        self._evader_speed[e] = []
        self._margin[e] = []
        self._min_margin[e] = float("inf")
        self._collided[e] = False
        self._capture_obstacles([e])


def run_suite(
    *,
    env,
    base_env,
    policy,
    obs,
    suite_total: int,
    metrics: RolloutMetrics,
    traj_log: TrajectoryLog | None = None,
    is_running=lambda: True,
) -> None:
    """Step the policy until every suite scenario has finished exactly once.

    obs is the observation returned by the initial env.reset(). Parked envs
    (scenario code 0: the suite is exhausted for that env) are excluded from the
    counted metrics.
    """
    ego = base_env.obs_layout.ego
    current_pos = obs[:, ego][:, :3].clone()
    if traj_log is not None:
        traj_log.seed(range(int(env.num_envs)), current_pos, base_env._reference_pos)

    while metrics.num_episodes < suite_total and is_running():
        driver = getattr(base_env, "_eval_driver", None)
        trial_indices = dict(driver.assigned_indices) if driver is not None else {}
        with torch.inference_mode():
            action = policy.act(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)

        done = (terminated | truncated).reshape(-1)
        reset_pos = obs[:, ego][:, :3]
        # The snapshot preserves terminal state + geometry-derived safety data across auto-reset.
        snapshot = base_env.get_last_step_snapshot()
        pos_local = snapshot["pos_local"]
        tier = snapshot["scenario_tier"]
        status = base_env.get_last_episode_status()
        for e in torch.nonzero(done & (tier > 0), as_tuple=False).flatten().tolist():
            metrics.trial_indices.append(trial_indices.get(e, -1))
        metrics.update(
            agent_pos=pos_local,
            nearest_obstacle_dist=snapshot["minimum_safety_margin"],
            done=done & (tier > 0),
            status=status,
            collision=snapshot["safety_violation"],
            filter_info=getattr(policy, "last_filter_info", None),
            group=tier,
            step_distance=torch.linalg.vector_norm(pos_local - current_pos, dim=-1),
        )
        if traj_log is not None:
            traj_log.record_step(snapshot)
        if done.any():
            done_ids = done.nonzero(as_tuple=False).reshape(-1)
            if traj_log is not None:
                for e in done_ids.tolist():
                    traj_log.finish(e, int(tier[e].item()), int(status[e].item()))
                    if int(tier[e]) > 0 and traj_log.episodes:
                        traj_log.episodes[-1]["suite_index"] = trial_indices.get(e, -1)
            policy.reset(done_ids)
            if traj_log is not None:
                traj_log.seed(done_ids.tolist(), reset_pos, base_env._reference_pos)
        current_pos = torch.where(done[:, None], reset_pos, pos_local).clone()


def compose_summary(metrics: RolloutMetrics, suite_metadata: dict | None = None) -> dict:
    """Overall metrics plus family, category and per-speed breakdowns."""
    summary = metrics.summary()
    by_scenario = {}
    for code in EVALUATION_SCENARIO_CODES:
        scenario = metrics.summary_for_groups([code])
        if scenario["num_episodes"] > 0:
            by_scenario[SCENARIO_LABELS[code]] = scenario
    if by_scenario:
        summary["by_scenario"] = by_scenario
    trials = (suite_metadata or {}).get("trials", [])
    if trials:
        if len(metrics.trial_indices) != metrics.num_episodes or any(
            idx < 0 or idx >= len(trials) for idx in metrics.trial_indices
        ):
            raise ValueError("Evaluation records are missing valid suite trial identities")
        records = [trials[idx] for idx in metrics.trial_indices]
        speeds = sorted({row["requested_speed"] for row in records})

        def select(key, value, speed=None):
            return metrics.summary_for_indices([
                i for i, row in enumerate(records)
                if row[key] == value and (speed is None or row["requested_speed"] == speed)
            ])

        categories = list(dict.fromkeys(row["category"] for row in records))
        summary["by_family"] = {family: select("family", family) for family in FAMILIES}
        summary["by_family_speed"] = {
            family: {str(speed): select("family", family, speed) for speed in speeds} for family in FAMILIES
        }
        summary["by_category_speed"] = {
            category: {str(speed): select("category", category, speed) for speed in speeds}
            for category in categories
        }
    return summary


def episode_records(metrics: RolloutMetrics, suite_metadata: dict | None = None) -> list[dict]:
    """Per-episode scalars of every scored episode (scenario safety boxplot)."""
    records = [
        {
            "tier": int(g),
            "min_safety_margin": float(ms),
            "captured": bool(c),
            "collided": bool(col),
            "status": int(status),
            "duration_steps": int(steps),
            "duration_s": float(steps * metrics.step_dt),
            "pursuer_speed_mean": float(speed),
        }
        for g, ms, c, col, status, steps, speed in zip(
            metrics.groups,
            metrics.min_separations,
            metrics.captures,
            metrics.collisions,
            metrics.statuses,
            metrics.episode_lengths,
            metrics.pursuer_speed_mean,
        )
    ]
    trials = (suite_metadata or {}).get("trials", [])
    if trials:
        if len(metrics.trial_indices) != len(records):
            raise ValueError("Missing trial indices")
        for record, idx in zip(records, metrics.trial_indices):
            if not 0 <= idx < len(trials):
                raise ValueError("Invalid trial index")
            record.update(trials[idx])
            record["suite_index"] = idx
    return records
