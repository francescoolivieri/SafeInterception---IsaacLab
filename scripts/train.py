"""Train PPO, graph PPO, DG-PPO or a PNCBF certificate."""

# Launch Isaac Sim first.

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "PPO_GRAPH", "DGPPO", "PNCBF"],
    help="The RL algorithm used for training.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run one synchronized training process per GPU (launch with torchrun).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--nominal-checkpoint",
    type=str,
    default=None,
    help="PNCBF only: frozen obstacle-blind nominal velocity-PPO checkpoint to certify.",
)
parser.add_argument(
    "--policy-filter-checkpoint",
    type=str,
    default=None,
    help="PNCBF only, policy iteration: collect data with the nominal policy filtered by this certificate.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--log-suite",
    type=str,
    default=None,
    help="Override the logs/projects/<task>/<suite> segment.",
)
parser.add_argument(
    "--log-run-name",
    type=str,
    default=None,
    help="Override the timestamped run-directory name.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything else imports after the sim app exists.

import gymnasium as gym
import random
import torch
from datetime import datetime
import skrl
if skrl.__version__ != "2.1.0":
    raise RuntimeError(f"This project uses skrl 2.1.0; found {skrl.__version__}.")

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import target_interception.task  # noqa: F401  (registers the gym ids)

from _cli_utils import (
    apply_env_overrides_from_agent_cfg,
    resolve_network_presets,
    resolve_training_context,
    configure_wandb_logging,
    obs_layout_to_dict,
    verify_action_contract,
    verify_obs_layout_matches_training,
)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


from target_interception.monitoring import PerformanceTracker, WandbBridge, TrainingMonitor, PPORunner


def _build_runner(env, agent_cfg: dict):
    """Algorithm dispatch: skrl Runner for PPO, DGPPORunner for DG-PPO."""
    if algorithm == "ppo_graph":
        from target_interception.algorithms.ppo_graph.runner import GraphPPORunner

        return GraphPPORunner(env, agent_cfg)
    if algorithm == "dgppo":
        from target_interception.algorithms.dgppo.agent import DGPPORunner

        return DGPPORunner(env, agent_cfg)
    return PPORunner(env, agent_cfg)


def _pncbf_nominal_context(env_cfg) -> dict:
    """PNCBF certifies a frozen obstacle-blind velocity PPO; check it before gym.make."""
    if not args_cli.nominal_checkpoint:
        raise ValueError("--algorithm PNCBF requires --nominal-checkpoint")
    if args_cli.checkpoint:
        raise ValueError("PNCBF training does not resume from --checkpoint")
    context = resolve_training_context(
        {"type": "skrl_checkpoint", "checkpoint": args_cli.nominal_checkpoint}, args_cli.task
    )
    if (context.get("saved_env_cfg") or {}).get("obstacle_observation_mode") != "none":
        raise ValueError("The PNCBF nominal PPO must be trained on the obstacle-blind Nominal task")
    if env_cfg.obstacle_observation_mode != "pillars":
        raise ValueError("PNCBF certificate training needs pillar observations")
    verify_action_contract(context, env_cfg)
    return context


def _build_pncbf_trainer(env, base_env, agent_cfg: dict, context: dict, log_dir: str):
    """Load the frozen nominal policy and fit its certificate into this run directory."""
    from skrl.utils import set_seed

    from target_interception.algorithms.pncbf import PNCBFTrainConfig, PNCBFTrainer
    from target_interception.policies import PolicyContext, build_policy

    policy_cfg = {"type": "skrl_checkpoint", "checkpoint": args_cli.nominal_checkpoint}
    policy = build_policy(
        policy_cfg,
        PolicyContext(
            env=env,
            base_env=base_env,
            obs_layout=base_env.obs_layout,
            device=env.device,
            agent_cfg=context["agent_cfg"],
            command_scale=base_env.velocity_command_scale,
            policy_obs_dim=verify_obs_layout_matches_training(context, base_env.obs_layout),
            saved_obs_layout=context["saved_obs_layout"],
        ),
    )
    if args_cli.policy_filter_checkpoint:
        # Policy iteration: certify the nominal policy behind an earlier certificate (thesis gains).
        from target_interception.safety.pncbf import PNCBFFilter
        from target_interception.policies import FilteredPolicy

        params = dict(agent_cfg["agent"]["policy_filter"])
        safety_filter = PNCBFFilter(base_env.obs_layout, checkpoint=params.pop("checkpoint"),
                                    command_scale=base_env.velocity_command_scale, collection=True,
                                    device=env.device, **params)
        policy = FilteredPolicy(policy, safety_filter)
    # The nominal skrl Runner reseeds global RNGs from its saved agent config.
    set_seed(agent_cfg["seed"])
    run_info = {"task": args_cli.task, "certificate_seed": agent_cfg["seed"], "nominal_checkpoint": context["checkpoint"]}
    dump_yaml(os.path.join(log_dir, "params", "nominal_policy.yaml"), run_info)
    config = PNCBFTrainConfig.from_dict({k: v for k, v in agent_cfg["agent"].items() if k != "experiment"})
    return PNCBFTrainer(env=env, policy=policy, obs_layout=base_env.obs_layout,
                        command_scale=base_env.velocity_command_scale, step_dt=base_env.step_dt,
                        config=config, output_dir=log_dir, run_info=run_info)


def _validated_log_segment(value: str, *, option: str) -> str:
    """--log-suite and --log-run-name must be a single folder name."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not value or value in {".", ".."} or any(char not in allowed for char in value):
        raise ValueError(f"{option} must be one non-empty path segment containing only letters, digits, '-', '_' or '.'.")
    return value


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Train with a skrl agent."""
    # override config with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    num_envs = int(env_cfg.scene.num_envs)

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires a CUDA device")
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    distributed = bool(skrl.config.torch.is_distributed)
    if args_cli.distributed != distributed:
        raise RuntimeError(
            "--distributed must be launched with torchrun (WORLD_SIZE > 1); "
            "do not use torchrun without --distributed"
        )
    rank = int(skrl.config.torch.rank) if distributed else 0
    world_size = int(skrl.config.torch.world_size) if distributed else 1
    is_primary = rank == 0
    if distributed:
        print(
            f"[INFO] Distributed training rank={rank}/{world_size - 1} "
            f"device={env_cfg.sim.device} local_envs={num_envs} global_envs={num_envs * world_size}",
            flush=True,
        )

    # The budget is trainer.timesteps of the agent yaml (override with agent.trainer.timesteps=N).
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Let the env know the total training budget (drives the interception curriculum schedule).
    total_timesteps = int(agent_cfg.get("trainer", {}).get("timesteps", 0))
    if total_timesteps:
        env_cfg.total_timesteps = total_timesteps
        env_cfg.curriculum_total_steps = total_timesteps

    # Apply this method's task settings.
    apply_env_overrides_from_agent_cfg(env_cfg, agent_cfg)
    from target_interception.scenarios import penalty_curriculum
    if penalty_curriculum.enabled(env_cfg):
        penalty_curriculum.apply(env_cfg)
    resolve_network_presets(agent_cfg)
    nominal_context = _pncbf_nominal_context(env_cfg) if algorithm == "pncbf" else None
    if algorithm == "pncbf" and args_cli.policy_filter_checkpoint:
        # Set here, not in _build_pncbf_trainer: params/agent.yaml is dumped before the env exists
        # and has to record which earlier certificate this policy-iteration round ran behind.
        agent_cfg["agent"]["policy_filter"] = {"alpha_safe": 1.5, "alpha_unsafe": 1.5, "value_shift": 0.01,
                                               "checkpoint": os.path.abspath(args_cli.policy_filter_checkpoint)}
    # seeds
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    if algorithm == "pncbf" and distributed:
        # Distributed PNCBF ranks collect independent data: distinct scenario streams.
        env_cfg.seed = agent_cfg["seed"] + rank

    # skrl's private TensorBoard writer cannot be patched reliably by W&B, so
    # the monitoring hook mirrors the compact PPO/task metrics directly.
    configure_wandb_logging(agent_cfg)

    # Runs go to logs/projects/<task>/<suite>/<timestamp>.
    task_id = str(args_cli.task)
    task_dir = "target_interception" if task_id.startswith("TargetInterception-") else task_id
    variant = task_id.removeprefix("TargetInterception-").removesuffix("-v0")
    default_suite = f"{variant}_{algorithm}".replace("-", "_").lower()
    suite = (
        _validated_log_segment(args_cli.log_suite, option="--log-suite")
        if args_cli.log_suite is not None
        else default_suite
    )
    log_root_path = os.path.abspath(os.path.join("logs", "projects", task_dir, suite))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = None
    if is_primary:
        log_dir = (
            _validated_log_segment(args_cli.log_run_name, option="--log-run-name")
            if args_cli.log_run_name is not None
            else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    if distributed:
        shared_log_dir = [log_dir]
        torch.distributed.broadcast_object_list(shared_log_dir, src=0)
        log_dir = shared_log_dir[0]
    assert log_dir is not None
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    wandb_kwargs = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
    wandb_kwargs.setdefault("dir", log_dir)
    if not wandb_kwargs.get("name"):
        wandb_kwargs["name"] = f"{default_suite}_{os.path.basename(log_dir)}"
    wandb_config = wandb_kwargs.setdefault("config", {})
    wandb_config.update(
        {
            "task": task_id,
            "algorithm": algorithm,
            "seed": int(agent_cfg["seed"]),
            "num_envs": num_envs * world_size,
            "total_frames": total_timesteps * num_envs * world_size,
            "scenario_pool": env_cfg.scenario.pool_name,
        }
    )

    # skrl disables TensorBoard/checkpoint intervals on follower ranks, but its
    # base agent does not yet disable W&B there. Keep all persistent output on
    # rank zero; every rank still participates in synchronized optimization.
    if not is_primary:
        agent_cfg["agent"]["experiment"]["wandb"] = False

    # dump resolved configs
    if is_primary:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # create the environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    # serialize the observation layout into the run config for eval-time consumers
    if is_primary:
        dump_yaml(os.path.join(log_dir, "params", "obs_layout.yaml"), obs_layout_to_dict(base_env.obs_layout))

    # wrap for skrl
    env = SkrlVecEnvWrapper(env)
    print("Obs space:", env.single_observation_space, "Action space:", env.single_action_space)

    # build the runner (algorithm dispatch)
    if algorithm == "pncbf":
        runner = _build_pncbf_trainer(env, base_env, agent_cfg, nominal_context, log_dir)
    else:
        runner = _build_runner(env, agent_cfg)

    # optional WandB mirroring + throughput / env-metric logging
    wandb_bridge = WandbBridge()
    if is_primary and algorithm != "pncbf":
        perf_tracker = PerformanceTracker(num_envs=num_envs * world_size)
        runner.agent.monitor = TrainingMonitor(
            base_env,
            wandb_bridge,
            perf_tracker,
            global_num_envs=num_envs * world_size,
        )

    # resume from checkpoint if requested
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    run_failed = False
    try:
        if algorithm == "pncbf":
            runner.run(timesteps=agent_cfg["trainer"]["timesteps"])
        else:
            runner.run()
        total_timesteps = int(agent_cfg.get("trainer", {}).get("timesteps", 0))
        if total_timesteps > 0 and is_primary and algorithm != "pncbf":
            # Very short smoke runs can finish before skrl's periodic
            # checkpoint hook creates this directory.
            Path(log_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
            runner.agent.write_checkpoint(timestep=total_timesteps, timesteps=total_timesteps)
    except Exception:
        run_failed = True
        raise
    finally:
        wandb_bridge.finish(exit_code=1 if run_failed else 0)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
