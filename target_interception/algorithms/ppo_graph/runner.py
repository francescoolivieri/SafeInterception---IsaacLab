"""Standard skrl PPO with the DG-PPO graph architecture (M4)."""
from __future__ import annotations

import copy
from typing import Any

import torch
from skrl import config
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from target_interception.algorithms.dgppo.distributed import broadcast_modules
from target_interception.monitoring import MonitoredPPO
from target_interception.nn.networks import (
    build_network,
    resolve_graph_input,
    resolve_networks,
)
from target_interception.nn.preprocessing import build_graph_input

from .models import GraphObservationEncoder, GraphPPOPolicy, GraphPPOValue


def graph_ppo_architecture(cfg: dict) -> tuple[dict, dict]:
    """(graph_input, {policy, value}) of a graph PPO config."""
    networks = resolve_networks(cfg["networks"])
    if set(networks) != {"policy", "value"}:
        raise ValueError(f"graph PPO needs policy and value networks, got {sorted(networks)}")
    return resolve_graph_input(cfg.get("graph_input")), networks


def build_graph_ppo_models(env, cfg: dict):
    """Actor and critic adapters around one shared graph encoder."""
    device = torch.device(env.device)
    graph_input, networks = graph_ppo_architecture(cfg)
    if networks["value"]["head"] != {"type": "value", "outputs": 1}:
        raise ValueError("PPO_GRAPH needs a single-output value network")
    graph, feature_preprocessor = build_graph_input(graph_input, env.unwrapped.obs_layout, device=device)
    encoder = GraphObservationEncoder(graph, feature_preprocessor)
    context_dim = feature_preprocessor.ego_context_dim
    spaces = {
        "observation_space": env.observation_space,
        "state_space": env.state_space,
        "action_space": env.action_space,
        "device": device,
    }
    policy = build_network(networks["policy"], context_dim=context_dim, action_dim=int(env.action_space.shape[0]),
                           device=device)
    value = build_network(networks["value"], context_dim=context_dim, device=device)
    return (
        GraphPPOPolicy(core=policy, encoder=encoder, **spaces),
        GraphPPOValue(core=value, encoder=encoder, **spaces),
        encoder,
        feature_preprocessor,
    )


def _ppo_cfg(raw: dict, *, device) -> PPO_CFG:
    cfg = copy.deepcopy(raw)
    cfg.pop("class", None)
    if cfg.get("observation_preprocessor") is not None:
        raise ValueError(
            "PPO_GRAPH requires agent.observation_preprocessor: null; "
            "physical features are normalized after graph construction"
        )
    cfg["observation_preprocessor"] = None
    cfg["observation_preprocessor_kwargs"] = {}
    if cfg.get("value_preprocessor") == "RunningStandardScaler":
        cfg["value_preprocessor"] = RunningStandardScaler
        value_preprocessor_kwargs = dict(cfg.get("value_preprocessor_kwargs") or {})
        value_preprocessor_kwargs.update({"size": 1, "device": device})
        cfg["value_preprocessor_kwargs"] = value_preprocessor_kwargs
    elif cfg.get("value_preprocessor") is not None:
        raise ValueError("PPO_GRAPH only supports the RunningStandardScaler value preprocessor")
    if cfg.get("learning_rate_scheduler") == "KLAdaptiveLR":
        cfg["learning_rate_scheduler"] = KLAdaptiveLR
    elif cfg.get("learning_rate_scheduler") is not None:
        raise ValueError("PPO_GRAPH only supports the KLAdaptiveLR scheduler")
    if float(cfg.pop("rewards_shaper_scale", 1.0)) != 1.0:
        raise ValueError("PPO_GRAPH does not scale rewards")
    if isinstance(cfg.get("experiment"), dict):
        cfg["experiment"] = ExperimentCfg(**cfg["experiment"])
    return PPO_CFG(**cfg)


class GraphPPO(MonitoredPPO):
    """skrl PPO that also updates the graph-feature scaler after each update."""

    def __init__(self, *, graph_encoder, feature_preprocessor, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "graph_encoder", graph_encoder)
        object.__setattr__(self, "feature_preprocessor", feature_preprocessor)
        self.checkpoint_modules["observation_preprocessor"] = feature_preprocessor

    def update(self, *, timestep: int, timesteps: int) -> None:
        # super().update must see the same frozen statistics used to collect the
        # rollout, otherwise the stored and recomputed log probabilities drift.
        super().update(timestep=timestep, timesteps=timesteps)
        observations = self.memory.get_tensor_by_name("observations")
        rank = int(config.torch.rank) if config.torch.is_distributed else 0
        if rank == 0:
            self.graph_encoder.update(observations)
        broadcast_modules((self.feature_preprocessor,))
        self.graph_encoder.clear_cache()


class GraphPPORunner:
    """Graph models around skrl's standard PPO agent."""

    def __init__(self, env: Any, cfg: dict) -> None:
        self.env = env
        set_seed(cfg.get("seed"))
        policy, value, encoder, feature_preprocessor = build_graph_ppo_models(env, cfg)
        agent_cfg = _ppo_cfg(cfg.get("agent", {}), device=env.device)
        memory_cfg = dict(cfg.get("memory") or {})
        memory_cfg.pop("class", None)
        memory_size = int(memory_cfg.pop("memory_size", -1))
        if memory_size < 0:
            memory_size = int(agent_cfg.rollouts)
        if memory_cfg:
            raise ValueError(f"unknown PPO_GRAPH memory options: {sorted(memory_cfg)}")
        memory = RandomMemory(memory_size=memory_size, num_envs=int(env.num_envs), device=env.device)
        self.agent = GraphPPO(
            models={"policy": policy, "value": value},
            memory=memory,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            cfg=agent_cfg,
            graph_encoder=encoder,
            feature_preprocessor=feature_preprocessor,
        )
        trainer_cfg = dict(cfg.get("trainer") or {})
        trainer_cfg.pop("class", None)
        self.trainer = SequentialTrainer(env=env, agents=self.agent, cfg=trainer_cfg)

    def run(self) -> None:
        self.trainer.train()
