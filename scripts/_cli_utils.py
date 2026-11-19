"""Config and checkpoint helpers shared by the scripts."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf


# Config helpers

def configure_wandb_logging(agent_cfg: Any) -> None:
    """Log skrl metrics directly to W&B.

    skrl binds its writer before wandb.init, so TensorBoard syncing misses losses."""
    experiment_cfg = agent_cfg.get("agent", {}).get("experiment", {})
    if not bool(experiment_cfg.get("wandb", False)):
        return
    wandb_kwargs = experiment_cfg.get("wandb_kwargs", {})
    wandb_kwargs.setdefault("sync_tensorboard", False)
    experiment_cfg["wandb_kwargs"] = wandb_kwargs
    agent_cfg["agent"]["experiment"] = experiment_cfg


def _as_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def apply_env_overrides_from_agent_cfg(env_cfg: Any, agent_cfg: Any) -> None:
    """Apply the agent yaml's env: block onto the env cfg (before gym.make)."""
    cfg = _as_plain_container(agent_cfg)
    if not isinstance(cfg, Mapping):
        return
    env_overrides = _as_plain_container(cfg.get("env", cfg.get("environment", None)))
    if not isinstance(env_overrides, Mapping) or not env_overrides:
        return

    applied: list[str] = []

    def apply_one(target: Any, key: str, value: Any, path: str) -> None:
        if key.startswith("_"):
            return
        if not hasattr(target, key):
            print(f"[WARN] Ignoring agent env override '{path}': env config has no such attribute.")
            return
        current = getattr(target, key)
        value = _as_plain_container(value)
        if isinstance(value, Mapping) and not isinstance(current, Mapping):
            for child_key, child_value in value.items():
                apply_one(current, child_key, child_value, f"{path}.{child_key}")
            return
        setattr(target, key, value)
        applied.append(path)

    for key, value in env_overrides.items():
        apply_one(env_cfg, key, value, str(key))

    if applied:
        print(f"[INFO] Applied agent env overrides: {', '.join(sorted(applied))}")


def obs_layout_to_dict(layout: Any) -> dict:
    """Serialize observation slices and arena bounds for checkpoint loaders."""
    def _s(sl: slice) -> list[int]:
        return [int(sl.start), int(sl.stop)]

    return {
        "ego": _s(layout.ego),
        "target": _s(layout.target),
        "extras": {key: _s(value) for key, value in layout.extras.items()},
        "arena_min": _floats_or_none(layout.arena_min),
        "arena_max": _floats_or_none(layout.arena_max),
        "obs_dim": int(layout.obs_dim),
    }


def _floats_or_none(values) -> list | None:
    return None if values is None else [float(v) for v in values]


def resolve_network_presets(agent_cfg: dict) -> None:
    """Expand named network presets before saving the training config."""
    from target_interception.nn.networks import resolve_graph_input, resolve_networks

    for block in (agent_cfg.get("agent") or {}, agent_cfg):
        if block.get("networks"):
            block["networks"] = resolve_networks(block["networks"])
            block["graph_input"] = resolve_graph_input(block.get("graph_input"))


def enforce_algorithm_env_contracts(env_cfg: Any, algorithm_name: str) -> None:
    """Check that DG-PPO can build its graph from the configured obstacle observations."""
    if algorithm_name != "dgppo":
        return
    mode = env_cfg.obstacle_observation_mode
    if mode == "pillars":
        pass  # privileged obstacle block is already graph-compatible
    elif mode == "none":
        print(
            "[WARN] DG-PPO with obstacle_observation_mode='none': no obstacle nodes, so the "
            "world obstacles still contribute to safety costs but go unobserved."
        )
    else:
        raise ValueError(
            f"Unsupported obstacle_observation_mode '{mode}' for DG-PPO. "
            "Expected one of 'pillars', 'none'."
        )


ALGO_BY_POLICY = {"skrl_checkpoint": "ppo", "dgppo_checkpoint": "dgppo"}
ACTION_CONTRACT_FIELDS = (
    "action_space",
    "velocity_command_scale_m_s",
    "velocity_yaw_rate_scale_rad_s",
    "sim_frequency",
    "policy_rate_hz",
    "decimation",
    "command_delay_s",
)


def resolve_training_context(policy_cfg: dict, task: str) -> dict:
    """Load the configs saved beside a checkpoint, so evaluation runs with its training settings.

    Missing params are an error: current defaults may describe a different model.
    """
    from pathlib import Path

    from isaaclab.utils.io import load_yaml
    from target_interception.policies import resolve_checkpoint

    ptype = policy_cfg.get("type")
    algorithm = ALGO_BY_POLICY.get(ptype)
    ctx = {
        "algorithm": algorithm, "checkpoint": None,
        "agent_cfg": None, "saved_env_cfg": None, "saved_obs_layout": None,
    }
    if algorithm is None:
        if ptype == "classical":
            return ctx
        raise ValueError(
            f"unsupported policy type {ptype!r}; use 'classical', 'skrl_checkpoint', "
            "or 'dgppo_checkpoint'."
        )

    checkpoint = policy_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"policy type '{ptype}' requires a 'checkpoint'.")
    ckpt_path = resolve_checkpoint(checkpoint)
    ctx["checkpoint"] = ckpt_path

    params_dir = Path(ckpt_path).resolve().parent.parent / "params"  # <run>/checkpoints/x.pt -> <run>/params
    agent_yaml = params_dir / "agent.yaml"
    required = [agent_yaml, params_dir / "env.yaml", params_dir / "obs_layout.yaml"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"Checkpoint {ckpt_path} is missing its saved params at {params_dir}: "
            f"{', '.join(missing)}. There is no fallback to the registry defaults, which "
            "may not match the actions and observations it was trained with."
        )

    ctx["agent_cfg"] = load_yaml(str(agent_yaml)) or {}
    if algorithm == "ppo" and ctx["agent_cfg"].get("networks"):
        ctx["algorithm"] = "ppo_graph"
    ctx["saved_obs_layout"] = load_yaml(str(params_dir / "obs_layout.yaml")) or {}
    ctx["saved_env_cfg"] = load_yaml(str(params_dir / "env.yaml")) or {}
    print(f"[INFO] Using training config saved at {params_dir}.")
    return ctx


def verify_action_contract(tctx: dict, env_cfg: Any) -> None:
    """Reject checkpoints whose action channels mean something else in this env."""
    if tctx.get("algorithm") is None:
        return
    saved = tctx["saved_env_cfg"]
    mismatches = []
    for key in ACTION_CONTRACT_FIELDS:
        trained = _normalize_contract_value(saved.get(key))
        live = _normalize_contract_value(getattr(env_cfg, key, None))
        if trained != live:
            mismatches.append(f"{key}: eval={live!r} != train={trained!r}")
    if mismatches:
        raise ValueError("Checkpoint action settings do not match this env:\n  - " + "\n  - ".join(mismatches))
    print("[INFO] Action settings match the checkpoint's training environment.")


def _normalize_contract_value(value: Any) -> Any:
    """Tuples become lists, so saved YAML and live config values compare equal."""
    if isinstance(value, (tuple, list)):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize_contract_value(item) for key, item in value.items()}
    return value


def verify_obs_layout_matches_training(tctx: dict, obs_layout: Any, *, ignore_arena: bool = False) -> Optional[int]:
    """Check the live layout against the saved one; returns the policy's observation width.

    A nominal obstacle-blind policy may pick its saved fields out of a wider filter env.
    Everything else must match, except the arena box under ignore_arena (OOD room):
    it is not an observation block and the graph builders read it from the live env.
    """
    if tctx.get("algorithm") is None:
        return None
    saved = tctx["saved_obs_layout"]
    if saved is None:
        raise ValueError(
            "Checkpoint has no saved params/obs_layout.yaml, so its observation "
            "layout cannot be checked. Retrain in this repository."
        )
    live = obs_layout_to_dict(obs_layout)
    saved = dict(saved)
    policy_obs_dim = int(saved["obs_dim"])
    if policy_obs_dim < int(live["obs_dim"]):
        from target_interception.policies import nominal_observation_indices

        if tctx["algorithm"] != "ppo" or (tctx.get("saved_env_cfg") or {}).get("obstacle_observation_mode") != "none":
            raise ValueError("Only nominal obstacle-blind PPO checkpoints may use a wider live observation")
        nominal_observation_indices(saved, obs_layout)
        keys = ("ego", "target")
    else:
        keys = sorted(set(saved) | set(live))
    if ignore_arena:
        keys = [k for k in keys if k not in ("arena_min", "arena_max")]
        print(f"[INFO] OOD arena: trained arena {saved.get('arena_min')}..{saved.get('arena_max')}, "
              f"evaluated in {live.get('arena_min')}..{live.get('arena_max')}.")
    diffs = [f"{k}: eval={live.get(k)} != train={saved.get(k)}" for k in keys if live.get(k) != saved.get(k)]
    if diffs:
        raise ValueError(
            f"Eval observation layout does not match the training layout for checkpoint "
            f"{tctx['checkpoint']}:\n  - " + "\n  - ".join(diffs) +
            "\nThe policy was trained with different observations; align the env "
            "config (registry / experiment env_overrides) with that run."
        )
    print("[INFO] obs_layout matches the checkpoint's training layout.")
    return policy_obs_dim


def apply_dot_overrides(cfg: Any, overrides: Mapping[str, Any]) -> None:
    """Apply dot-path overrides onto the env cfg; an unknown path is an error (typo guard)."""
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        if not hasattr(node, parts[-1]):
            raise ValueError(f"env_overrides path '{key}' does not exist on the env cfg.")
        setattr(node, parts[-1], value)
