"""Standardization of graph features and the vehicle-state context."""
from __future__ import annotations

import torch
import torch.nn as nn
from skrl.resources.preprocessors.torch import RunningStandardScaler

from target_interception.nn.graph import EDGE_DIM, PAD_TYPE, GraphData

ATTITUDE_DIM = 9  # rotation matrix


class GraphScaler(nn.Module):
    """Standardize the edges and assemble the ego context.

    Ego context: [velocity (3), attitude (9), body rate (3), previous action (4)]. Edges,
    velocity and body rate are standardized with running statistics; the rotation matrix
    and the previous action keep their native range.
    """

    def __init__(self, obs_layout, *, clip_threshold: float = 5.0, epsilon: float = 1e-8, device) -> None:
        super().__init__()
        self.layout = obs_layout
        if obs_layout.ego.stop - obs_layout.ego.start != 3 + 3 + ATTITUDE_DIM + 3:
            raise ValueError("the ego observation must hold position, velocity, rotation matrix and body rate")
        previous = obs_layout.extras["previous_action"]
        self.ego_context_dim = 6 + ATTITUDE_DIM + (previous.stop - previous.start)
        kwargs = dict(epsilon=float(epsilon), clip_threshold=float(clip_threshold), device=device)
        self.relation_scaler = RunningStandardScaler(size=EDGE_DIM, **kwargs)
        self.ego_motion_scaler = RunningStandardScaler(size=6, **kwargs)

    def _motion(self, observations: torch.Tensor) -> torch.Tensor:
        ego = observations[:, self.layout.ego]
        return torch.cat((ego[:, 3:6], ego[:, 15:18]), dim=-1)

    def forward(self, graph: GraphData, *, no_grad: bool = True) -> GraphData:
        observations = graph.observations
        ego = observations[:, self.layout.ego]
        motion = self.ego_motion_scaler(self._motion(observations), no_grad=no_grad)
        previous = observations[:, self.layout.extras["previous_action"]]
        ego_context = torch.cat((motion[:, :3], ego[:, 6:15], motion[:, 3:], previous), dim=-1)
        edges = self.relation_scaler(graph.edges, no_grad=no_grad)
        return graph._replace(edges=edges, ego_context=ego_context)

    @torch.no_grad()
    def update(self, graph: GraphData) -> None:
        """Update the running statistics after a rollout has used the frozen ones."""
        active = graph.node_types[graph.receivers] != PAD_TYPE
        if bool(active.any().item()):
            self.relation_scaler(graph.edges[active], train=True)
        self.ego_motion_scaler(self._motion(graph.observations), train=True)


def build_graph_input(graph_input, obs_layout, *, device):
    """(RelationGraph, GraphScaler) of a resolved graph_input spec."""
    from target_interception.nn.graph import RelationGraph

    if graph_input["mode"] != "relation" or not graph_input["normalize"]:
        raise ValueError(f"only the normalized relation graph is implemented, got {dict(graph_input)}")
    graph = RelationGraph(obs_layout, radius=float(graph_input["obs_radius"]))
    return graph, GraphScaler(obs_layout, clip_threshold=float(graph_input["clip_threshold"]),
                                           device=device)
