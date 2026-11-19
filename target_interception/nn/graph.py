"""Relation graph of the pursuer's scene, shared by DG-PPO (M5) and graph PPO (M4).

One fixed-size graph per environment, nodes in this order:

    0           pursuer
    1           target (evader)
    2 .. 2+K    obstacle slots (K from the ObsLayout)
    next 6      arena planes [x_min, x_max, y_min, y_max, floor, ceiling]
    last        padding node

Nodes carry only their type one-hot. Edges point into the pursuer and carry the relative
position and world velocity (pursuer minus sender, 6 values): target, then every
obstacle and arena plane. Edges outside the radius are routed to the padding node, so
every graph keeps the same size. Attitude, body rates and the previous action enter the
networks after the encoder as the ego context (see nn/preprocessing.py).

This module does not import torch_geometric.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

AGENT_TYPE, GOAL_TYPE, OBS_TYPE, BOUNDARY_TYPE, PAD_TYPE = 0, 1, 2, 3, -1
# One-hot order: [obstacle, goal, agent, boundary].
_TYPE_BIT = {OBS_TYPE: 0, GOAL_TYPE: 1, AGENT_TYPE: 2, BOUNDARY_TYPE: 3}
NODE_DIM = 4
EDGE_DIM = 6  # relative position (3) + relative world velocity (3)
N_PLANES = 6


@dataclass
class GraphData:
    """A batch of n_graphs equal-size graphs, nodes and edges concatenated."""

    n_graphs: int
    nodes: torch.Tensor  # (n_graphs * nodes_per_graph, NODE_DIM)
    edges: torch.Tensor  # (n_graphs * edges_per_graph, EDGE_DIM)
    receivers: torch.Tensor  # flat node index of each edge's receiver
    senders: torch.Tensor  # flat node index of each edge's sender
    node_types: torch.Tensor  # (n_graphs * nodes_per_graph,)
    observations: torch.Tensor | None = None  # (n_graphs, obs_dim) raw observations
    ego_context: torch.Tensor | None = None  # (n_graphs, context_dim), set by the preprocessor

    def agent_nodes(self, nodes: torch.Tensor) -> torch.Tensor:
        """Features of the pursuer node (node 0) of every graph."""
        return nodes.reshape(self.n_graphs, -1, nodes.shape[-1])[:, 0]

    def _replace(self, **kwargs) -> "GraphData":
        return replace(self, **kwargs)


def graph_data_select(graph: GraphData, indices: torch.Tensor) -> GraphData:
    """Sub-batch of the graphs at indices (flat graph ids)."""
    indices = indices.reshape(-1).to(device=graph.nodes.device, dtype=torch.long)
    count = int(indices.numel())
    per_node = graph.nodes.shape[0] // graph.n_graphs
    per_edge = graph.edges.shape[0] // graph.n_graphs

    def pick(tensor, per):
        return tensor.reshape(graph.n_graphs, per, *tensor.shape[1:])[indices].reshape(count * per, *tensor.shape[1:])

    shift = (torch.arange(count, device=indices.device) - indices)[:, None] * per_node
    receivers = (graph.receivers.reshape(graph.n_graphs, per_edge)[indices] + shift).reshape(-1)
    senders = (graph.senders.reshape(graph.n_graphs, per_edge)[indices] + shift).reshape(-1)
    return GraphData(
        n_graphs=count,
        nodes=pick(graph.nodes, per_node),
        edges=pick(graph.edges, per_edge),
        receivers=receivers,
        senders=senders,
        node_types=pick(graph.node_types, per_node),
        observations=None if graph.observations is None else graph.observations[indices],
        ego_context=None if graph.ego_context is None else graph.ego_context[indices],
    )


class RelationGraph:
    """Decode observations into node states and build the relation graph.

    Node states use the ego layout: position (0:3), world velocity (3:6), the rest zero
    for target, obstacles and planes.
    """

    def __init__(self, obs_layout, *, radius: float) -> None:
        self.layout = obs_layout
        self.radius = float(radius)
        self.state_dim = obs_layout.ego.stop - obs_layout.ego.start
        obstacles = obs_layout.extras["obstacles"]
        self.n_obstacles = (obstacles.stop - obstacles.start) // 3

    def extract(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pursuer [E, D], target [E, D], obstacles [E, K, D]) in the env frame."""
        E = observations.shape[0]
        agent = observations[:, self.layout.ego]
        target = observations[:, self.layout.target]  # p_e - p_p, v_e - v_p
        goal = observations.new_zeros(E, self.state_dim)
        goal[:, :6] = target + agent[:, :6]
        offsets = observations[:, self.layout.extras["obstacles"]].reshape(E, self.n_obstacles, 3)
        obstacles = observations.new_zeros(E, self.n_obstacles, self.state_dim)
        obstacles[..., :2] = offsets[..., :2] + agent[:, None, :2]
        relative_velocity = observations[:, self.layout.extras["obstacle_vel"]].reshape(E, self.n_obstacles, 2)
        obstacles[..., 3:5] = relative_velocity + agent[:, None, 3:5]
        return agent, goal, obstacles

    def planes(self, agent: torch.Tensor) -> torch.Tensor:
        """Arena plane nodes [E, 6, D]: the foot point of the pursuer on each plane."""
        planes = agent.new_zeros(agent.shape[0], N_PLANES, self.state_dim)
        planes[:, :, :3] = agent[:, None, :3]
        for axis in range(3):
            planes[:, 2 * axis, axis] = float(self.layout.arena_min[axis])
            planes[:, 2 * axis + 1, axis] = float(self.layout.arena_max[axis])
        return planes

    def build(self, agent: torch.Tensor, goal: torch.Tensor, obstacles: torch.Tensor) -> GraphData:
        E, K, device = agent.shape[0], obstacles.shape[1], agent.device
        planes = self.planes(agent)
        n_nodes = 2 + K + N_PLANES + 1
        pad = n_nodes - 1

        node_types = torch.full((n_nodes,), PAD_TYPE, dtype=torch.long, device=device)
        node_types[0], node_types[1], node_types[2:2 + K], node_types[2 + K:pad] = (
            AGENT_TYPE, GOAL_TYPE, OBS_TYPE, BOUNDARY_TYPE)
        nodes = agent.new_zeros(n_nodes, NODE_DIM)
        for node_type, bit in _TYPE_BIT.items():
            nodes[node_types == node_type, bit] = 1.0

        p = agent[:, :EDGE_DIM]
        to_obstacles = p[:, None] - obstacles[..., :EDGE_DIM]
        to_obstacles[..., 2] = 0.0  # cylinders span the arena height
        edges = torch.cat(((p - goal[:, :EDGE_DIM])[:, None], to_obstacles, p[:, None] - planes[..., :EDGE_DIM]), 1)
        # Clip the relative position to the radius.
        norm = torch.linalg.vector_norm(edges[..., :3], dim=-1, keepdim=True)
        scale = torch.where(norm > self.radius, self.radius / norm.clamp_min(1e-6), torch.ones_like(norm))
        edges = torch.cat((edges[..., :3] * scale, edges[..., 3:]), dim=-1)

        active = torch.cat((
            torch.ones(E, 1, dtype=torch.bool, device=device),
            torch.cdist(agent[:, None, :2], obstacles[..., :2])[:, 0] < self.radius,
            torch.cdist(agent[:, None, :3], planes[..., :3])[:, 0] < self.radius,
        ), dim=1)
        offset = torch.arange(E, device=device)[:, None] * n_nodes
        senders = torch.arange(1, pad, device=device).expand(E, -1)
        receivers = torch.where(active, 0, pad) + offset
        senders = torch.where(active, senders, pad) + offset
        return GraphData(
            n_graphs=E,
            nodes=nodes.repeat(E, 1),
            edges=edges.reshape(-1, EDGE_DIM),
            receivers=receivers.reshape(-1),
            senders=senders.reshape(-1),
            node_types=node_types.repeat(E),
        )

    def __call__(self, observations: torch.Tensor) -> GraphData:
        return self.build(*self.extract(observations))._replace(observations=observations)
