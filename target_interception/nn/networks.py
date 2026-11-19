"""Policy and value networks of the graph methods (M4 graph PPO, M5 DG-PPO).

Every network: GNN encoder over the relation graph -> pursuer-node embedding joined with
the ego context -> fusion MLP -> head (squashed Gaussian action or value outputs).
Architectures are named in the method configs and defined once in
task/agents/networks.yaml; build_network turns a resolved spec into a module.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from target_interception.nn.distributions import TanhNormal
from target_interception.nn.gnn import AttentionAggregationGNN
from target_interception.nn.graph import EDGE_DIM, NODE_DIM, GraphData
from target_interception.nn.mlp import MLP
from target_interception.nn.preprocessing import build_graph_input


def _encoder(spec: Mapping[str, Any]) -> AttentionAggregationGNN:
    if spec["type"] != "graph" or spec["backbone"] != "attention_aggregation":
        raise ValueError(f"unsupported encoder {spec!r}; only the attention_aggregation GNN is available")
    return AttentionAggregationGNN(
        in_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        msg_dim=spec["msg_dim"],
        out_dim=spec["out_dim"],
        hid_msg=spec["hid_msg"],
        hid_aggr=spec["hid_aggr"],
        hid_update=spec["hid_update"],
        n_layers=spec["layers"],
    )


class GraphValueNet(nn.Module):
    """V(graph), (n_graphs, n_out)."""

    def __init__(self, encoder: Mapping, fusion, *, context_dim: int, n_out: int, device=None) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")  # read by skrl
        self.n_out = int(n_out)
        self.gnn = _encoder(encoder)
        self.head = MLP(hid_sizes=fusion, in_dim=encoder["out_dim"] + context_dim, act=F.relu, act_final=True)
        self.value_out = nn.Linear(self.head.hid_sizes[-1], self.n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph: GraphData) -> torch.Tensor:
        x = torch.cat((self.gnn(graph), graph.ego_context), dim=-1)
        return self.value_out(self.head(x))


class GraphPolicy(nn.Module):
    """Squashed Gaussian policy pi(a | graph)."""

    def __init__(self, encoder: Mapping, fusion, *, context_dim: int, action_dim: int, scale_hid: int,
                 scale_final: float, std_dev_init: float, std_dev_min: float, device=None) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")  # read by skrl
        self.std_dev_min = std_dev_min
        self.std_dev_init_inv = math.log(math.exp(std_dev_init) - 1.0)  # softplus^-1
        self.gnn = _encoder(encoder)
        self.mlp = MLP(hid_sizes=fusion, in_dim=encoder["out_dim"] + context_dim, act=F.relu, act_final=True)
        # Small initial weights keep the initial means near zero.
        self.scale_hid = nn.Linear(self.mlp.hid_sizes[-1], scale_hid)
        nn.init.orthogonal_(self.scale_hid.weight)
        with torch.no_grad():
            self.scale_hid.weight.mul_(scale_final)
        nn.init.zeros_(self.scale_hid.bias)
        self.mean_head = nn.Linear(scale_hid, action_dim)
        self.std_head = nn.Linear(scale_hid, action_dim)  # softplus keeps the std positive
        for layer in (self.mean_head, self.std_head):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def distribution(self, graph: GraphData) -> TanhNormal:
        h = self.scale_hid(self.mlp(torch.cat((self.gnn(graph), graph.ego_context), dim=-1)))
        std = F.softplus(self.std_head(h) + self.std_dev_init_inv) + self.std_dev_min
        return TanhNormal(self.mean_head(h), std)

    def act(self, graph: GraphData, *, deterministic: bool = False):
        """(action, log_prob, mode); the action is the mode when deterministic."""
        dist = self.distribution(graph)
        action = dist.mode() if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.mode()

    def evaluate(self, graph: GraphData, action: torch.Tensor):
        """(log_prob, entropy) of taken actions."""
        dist = self.distribution(graph)
        return dist.log_prob(action), dist.entropy()


def build_network(spec: Mapping[str, Any], *, context_dim: int, action_dim: int | None = None, device=None):
    """Network of a resolved {encoder, fusion, head} spec."""
    head, fusion = spec["head"], spec["fusion"]
    if head["type"] == "gaussian":
        return GraphPolicy(
            spec["encoder"], fusion, context_dim=context_dim, action_dim=action_dim,
            scale_hid=head["scale_hid"], scale_final=head["scale_final"],
            std_dev_init=head["std_init"], std_dev_min=head["std_min"], device=device,
        )
    if head["type"] == "value":
        return GraphValueNet(spec["encoder"], fusion, context_dim=context_dim, n_out=head["outputs"],
                             device=device)
    raise ValueError(f"unknown head type {head['type']!r}")


PRESETS_PATH = Path(__file__).resolve().parents[1] / "task/agents/networks.yaml"
_GRAPH_KEYS = ("backbone", "layers", "out_dim", "msg_dim", "hid_msg", "hid_aggr", "hid_update")
_HEAD_KEYS = {"gaussian": ("scale_hid", "scale_final", "std_init", "std_min"), "value": ("outputs",)}


@lru_cache(maxsize=1)
def _presets() -> dict:
    with PRESETS_PATH.open() as stream:
        return yaml.safe_load(stream)


def resolve_graph_input(value: Mapping | str | None = None) -> dict[str, Any]:
    """Graph input spec; None selects the shared default from the presets."""
    spec = dict(_presets()["graph_input"] if value in (None, "default") else value)
    missing = {"mode", "obs_radius", "normalize", "clip_threshold"} - set(spec)
    if missing:
        raise ValueError(f"graph_input is missing {sorted(missing)}")
    return spec


def resolve_network(value: Mapping | str) -> dict[str, Any]:
    """Complete {encoder, fusion, head} spec of a preset name or a resolved spec."""
    if isinstance(value, str):
        table = _presets()
        if value not in table["networks"]:
            raise ValueError(f"unknown network {value!r}; presets: {sorted(table['networks'])}")
        spec = copy.deepcopy(table["networks"][value])
    else:
        spec = copy.deepcopy(dict(value))
    encoder, head = spec.get("encoder") or {}, spec.get("head") or {}
    if encoder.get("type") != "graph" or any(key not in encoder for key in _GRAPH_KEYS):
        raise ValueError(f"invalid encoder spec {encoder!r}")
    if head.get("type") not in _HEAD_KEYS or any(key not in head for key in _HEAD_KEYS[head["type"]]):
        raise ValueError(f"invalid head spec {head!r}")
    return spec


def resolve_networks(values: Mapping[str, Mapping | str]) -> dict[str, dict[str, Any]]:
    return {role: resolve_network(value) for role, value in values.items()}


def build_dgppo_policy(cfg, obs_layout, *, action_dim: int, device):
    """(RelationGraph, GraphScaler, GraphPolicy)."""
    graph, preprocessor = build_graph_input(cfg.graph_input, obs_layout, device=device)
    policy = build_network(cfg.networks["policy"], context_dim=preprocessor.ego_context_dim,
                           action_dim=action_dim, device=device)
    return graph, preprocessor, policy


def build_dgppo_critics(cfg, *, context_dim: int, device) -> tuple[GraphValueNet, GraphValueNet]:
    """Task critic V_l and constraint critic V_h (one head: the worst constraint)."""
    for role in ("vl", "vh"):
        if cfg.networks[role]["head"] != {"type": "value", "outputs": 1}:
            raise ValueError(f"DG-PPO {role} needs a single value output")
    vl = build_network(cfg.networks["vl"], context_dim=context_dim, device=device)
    vh = build_network(cfg.networks["vh"], context_dim=context_dim, device=device)
    return vl, vh
