"""Evaluate an experiment on the fixed suite for its arena.

Rebuilds the checkpoint's observation layout and writes metrics and,
unless --metrics-only is set, the recorded episodes (episodes.pt)."""

# Launch Isaac Sim first.

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an experiment on the fixed benchmark suite.")
parser.add_argument("--experiment", type=str, required=True, help="Path to a configs/*.yaml file.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Override policy.checkpoint from the experiment (useful for smoke/repeated runs).",
)
parser.add_argument(
    "--filter-checkpoint",
    type=str,
    default=None,
    help="Override filter.params.checkpoint (used by learned safety filters).",
)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Parallel envs (throughput only; episodes = suite size). Overrides the yaml.")
parser.add_argument("--output_dir", type=str, default=None, help="Where to write metrics.json. Overrides the yaml.")
parser.add_argument("--suite-name", type=str, default=None, help="Override evaluation_suite in the experiment YAML.")
parser.add_argument(
    "--ood-room",
    action="store_true",
    help="OOD check: evaluate the large-arena experiment in the room (room task and room suite).",
)
parser.add_argument(
    "--evaluation-seed", type=int, default=0,
    help="Seed of the noise and randomisation (thesis: 0 for the trained methods, 0 and 42 for M1).",
)
parser.add_argument(
    "--metrics-only",
    action="store_true",
    help="Write metrics.json only; skip the trajectory log (used by benchmark.py).",
)


AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything else imports after the sim app exists.

import json
import os
from collections.abc import Mapping

import gymnasium as gym
import torch
import yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

from _cli_utils import (
    apply_dot_overrides,
    apply_env_overrides_from_agent_cfg,
    enforce_algorithm_env_contracts,
    resolve_training_context,
    verify_action_contract,
    verify_obs_layout_matches_training,
)

import target_interception.task  # noqa: F401  (registers the gym ids)
from target_interception.evaluation import TrajectoryLog, compose_summary, episode_records, run_suite
from target_interception.metrics import RolloutMetrics
from target_interception.policies import PolicyContext, build_policy
from target_interception.safety import make_filter
from target_interception.scenarios.benchmark_suite import EvalScenarioDriver, default_suite_name, load_benchmark_suite
from target_interception.scenarios.large_arena import ROOM_TASK_FOR_LARGE_TASK
from target_interception.task.target_interception_env import (
    REASON_ALTITUDE,
    REASON_BOUNDARY,
    REASON_CAPTURE,
    REASON_COLLISION,
)
from target_interception.policies import FilteredPolicy

# Every method uses the same frozen scenarios; runtime DR/noise still depends on
# the evaluation seed and asynchronous reset order. Episode count equals suite size.
_TOP_KEYS = {
    "name",
    "task",
    "num_envs",
    "env_overrides",
    "policy",
    "filter",
    "evaluation_suite",
    "output_dir",
}


def load_experiment() -> dict:
    with open(args_cli.experiment) as f:
        exp = yaml.safe_load(f) or {}
    unknown = set(exp) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unknown experiment keys: {sorted(unknown)} (allowed: {sorted(_TOP_KEYS)})")
    if not exp.get("task"):
        raise ValueError("experiment must specify a 'task'.")
    if not exp.get("policy"):
        raise ValueError("experiment must specify a 'policy' block.")
    if args_cli.num_envs is not None:
        exp["num_envs"] = args_cli.num_envs
    if args_cli.checkpoint is not None:
        exp["policy"] = dict(exp["policy"])
        exp["policy"]["checkpoint"] = args_cli.checkpoint
    if args_cli.filter_checkpoint is not None:
        if not exp.get("filter"):
            raise ValueError("--filter-checkpoint requires an experiment with a filter")
        exp["filter"] = dict(exp["filter"])
        exp["filter"]["params"] = dict(exp["filter"].get("params") or {})
        exp["filter"]["params"]["checkpoint"] = args_cli.filter_checkpoint
    if args_cli.ood_room:
        if exp["task"] not in ROOM_TASK_FOR_LARGE_TASK:
            raise ValueError(f"--ood-room needs a large-arena task, got {exp['task']!r}")
        exp["task"] = ROOM_TASK_FOR_LARGE_TASK[exp["task"]]
        exp["name"] = f"{exp.get('name', 'experiment')}_ood_room"
        # The yaml's suite belongs to the large arena; the room's own suite applies.
        exp.pop("evaluation_suite", None)
        if exp.get("output_dir"):
            exp["output_dir"] = f"{str(exp['output_dir']).rstrip('/')}_ood_room"
    if args_cli.output_dir:
        exp["output_dir"] = args_cli.output_dir
    if args_cli.suite_name:
        exp["evaluation_suite"] = args_cli.suite_name
    exp.setdefault("num_envs", 16)
    exp.setdefault("env_overrides", {})
    exp.setdefault("output_dir", os.path.join("outputs", "evaluate", exp.get("name", exp["task"])))
    return exp


def build_env_cfg(exp: dict, tctx: dict, filter_cfg: dict | None):
    """Env cfg for a deterministic suite evaluation, set up as the checkpoint was trained."""
    env_cfg = parse_env_cfg(exp["task"], device=args_cli.device, num_envs=int(exp["num_envs"]))
    apply_dot_overrides(env_cfg, exp.get("env_overrides", {}))
    # Seeded, so the noisy evaluation is reproducible.
    env_cfg.seed = args_cli.evaluation_seed

    # Rebuild the observations and measurement model the checkpoint was trained with,
    # from the config saved next to it.
    if tctx["algorithm"] is not None:
        apply_env_overrides_from_agent_cfg(env_cfg, tctx["agent_cfg"])
        saved_measurement = (tctx.get("saved_env_cfg") or {}).get("measurement_model")
        if isinstance(saved_measurement, Mapping):
            for key, value in saved_measurement.items():
                if hasattr(env_cfg.measurement_model, key):
                    setattr(env_cfg.measurement_model, key, value)
        enforce_algorithm_env_contracts(env_cfg, tctx["algorithm"])

    # Collisions end the episode for every method in the benchmark; set after all overrides.
    env_cfg.terminate_on_safety_violation = True
    # Score violations at the physical arena; training uses arena_margin.
    env_cfg.evaluation_arena_margin = 0.0
    return env_cfg


def main() -> None:
    exp = load_experiment()
    task = exp["task"]
    num_envs = int(exp["num_envs"])
    policy_cfg = exp["policy"]
    filter_cfg = exp.get("filter")
    tctx = resolve_training_context(policy_cfg, task)
    env_cfg = build_env_cfg(exp, tctx, filter_cfg)
    exp["evaluation_suite"] = exp.get("evaluation_suite") or default_suite_name(env_cfg)
    print(
        f"[INFO] Evaluating '{exp.get('name', task)}' on {task} | num_envs={num_envs}"
    )
    verify_action_contract(tctx, env_cfg)

    # Size the obstacle slots to the suite (may exceed the training caps) and spawn them all.
    suite = load_benchmark_suite(env_cfg, args_cli.device, exp["evaluation_suite"])
    env_cfg.max_static_obstacles = suite.max_static_obstacles
    env_cfg.max_dynamic_obstacles = suite.max_dynamic_obstacles
    env_cfg.lazy_obstacle_spawning = False

    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    policy_obs_dim = verify_obs_layout_matches_training(tctx, base_env.obs_layout, ignore_arena=args_cli.ood_room)

    # Attach before the first reset, so resets walk the suite in order.
    base_env._eval_driver = EvalScenarioDriver(suite, base_env.device)

    env = SkrlVecEnvWrapper(env)
    ctx = PolicyContext(
        env=env,
        base_env=base_env,
        obs_layout=base_env.obs_layout,
        device=env.device,
        agent_cfg=tctx["agent_cfg"],
        command_scale=base_env.velocity_command_scale,
        policy_obs_dim=policy_obs_dim,
        saved_obs_layout=tctx["saved_obs_layout"],
    )
    policy = build_policy(policy_cfg, ctx)
    if filter_cfg:
        safety_filter = make_filter(
            filter_cfg,
            obs_layout=base_env.obs_layout,
            command_scale=base_env.velocity_command_scale,
            device=env.device,
        )
        policy = FilteredPolicy(policy, safety_filter)
        print(f"[INFO] Filter: {filter_cfg['type']} (params={filter_cfg.get('params')})")

    obs, _ = env.reset()
    policy.reset(None)
    metrics = RolloutMetrics(
        num_envs,
        capture_status=REASON_CAPTURE,
        collision_status=REASON_COLLISION,
        safety_statuses=(REASON_ALTITUDE, REASON_BOUNDARY, REASON_COLLISION),
        step_dt=float(base_env._step_dt),
        horizon_s=float(env_cfg.episode_length_s),
    )
    traj_log = None if args_cli.metrics_only else TrajectoryLog(base_env)

    print(f"[INFO] Rolling the {suite.total}-scenario suite across {num_envs} envs ...")
    run_suite(
        env=env,
        base_env=base_env,
        policy=policy,
        obs=obs,
        suite_total=suite.total,
        metrics=metrics,
        traj_log=traj_log,
        is_running=simulation_app.is_running,
    )

    if metrics.num_episodes != suite.total:
        raise RuntimeError(f"Incomplete evaluation: {metrics.num_episodes}/{suite.total} trials")
    if sorted(metrics.trial_indices) != list(range(suite.total)):
        raise RuntimeError("Evaluation must score every suite trial exactly once")
    summary = compose_summary(metrics, suite.metadata)
    records = episode_records(metrics, suite.metadata)
    result = {
        "name": exp.get("name", task),
        "task": task,
        "num_envs": num_envs,
        "benchmark_suite": exp["evaluation_suite"],
        "suite_size": suite.total,
        "evaluation_seed": args_cli.evaluation_seed,
        "records": records,
        "policy": policy_cfg,
        "filter": filter_cfg,
        "metrics": summary,
    }
    os.makedirs(exp["output_dir"], exist_ok=True)
    out_path = os.path.join(exp["output_dir"], "metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Wrote metrics to {out_path}")
    print(json.dumps(summary, indent=2))

    if traj_log is not None:
        payload = {
            "episodes": traj_log.episodes,
            "records": records,
            "meta": {
                "name": exp.get("name", task),
                "task": task,
                "step_dt": float(base_env._step_dt),
                "capture_radius": float(base_env.cfg.pos_tolerance),
                "arena_min": [float(x) for x in base_env.cfg.arena_min],
                "arena_max": [float(x) for x in base_env.cfg.arena_max],
                "enable_walls": bool(base_env.cfg.enable_walls),
                "episode_length_s": float(env_cfg.episode_length_s),
            },
        }
        ep_path = os.path.join(exp["output_dir"], "episodes.pt")
        torch.save(payload, ep_path)
        print(f"[INFO] Wrote {len(traj_log.episodes)} episode trajectories to {ep_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
