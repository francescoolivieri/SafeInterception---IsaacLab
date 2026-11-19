"""Gated-attention message-passing GNN (the "attention_aggregation" backbone).

Needs torch_geometric (message passing and grouped softmax), so it is kept apart from
the graph builders in nn/graph.py.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as softmax_pyg

from .graph import GraphData
from .mlp import MLP


class AttentionAggregation(MessagePassing):
    """One message-passing layer with gated-attention aggregation.

      * message: m_ij = Linear_msg( MLP_msg([edge_ij, x_j, x_i]) )
      * gate:    s_ij = Linear_gate( MLP_aggr(m_ij) )  (a scalar score per edge)
      * attn:    softmax of s_ij over the edges sharing a receiver i
      * aggregate: sum_j attn_ij * m_ij
      * update:  x_i' = Linear_out( MLP_upd([x_i, aggr_i]) )  (linear, no final act)
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        msg_dim: int,
        out_dim: int,
        hid_msg: Sequence[int],
        hid_aggr: Sequence[int],
        hid_update: Sequence[int],
    ):
        super().__init__(aggr="add")  # weighted sum: attn weights fold into the message
        self.in_dim = in_dim
        self.msg_dim = msg_dim
        self.out_dim = out_dim

        # upstream MLPs have no LayerNorm and no activation on the last hidden layer.
        self.msg_net = MLP(hid_msg, in_dim=edge_dim + 2 * in_dim, act_final=False, use_layernorm=False)
        self.msg_out = nn.Linear(hid_msg[-1], msg_dim)
        self.gate_net = MLP(hid_aggr, in_dim=msg_dim, act_final=False, use_layernorm=False)
        self.gate_out = nn.Linear(hid_aggr[-1], 1)
        self.update_net = MLP(hid_update, in_dim=in_dim + msg_dim, act_final=False, use_layernorm=False)
        self.update_out = nn.Linear(hid_update[-1], out_dim)
        for layer in (self.msg_out, self.gate_out, self.update_out):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, graph: GraphData) -> GraphData:
        edge_index = torch.stack([graph.senders, graph.receivers], dim=0)
        aggr = self.propagate(edge_index=edge_index, x=graph.nodes, edge_attr=graph.edges)
        out = self.update_out(self.update_net(torch.cat([graph.nodes, aggr], dim=-1)))
        return graph._replace(nodes=out)

    def message(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x_i: torch.Tensor,  # receiver features (pyg convention)
        x_j: torch.Tensor,  # sender features
        edge_attr: torch.Tensor,
        index: torch.Tensor,  # receiver index per edge (for softmax grouping)
    ) -> torch.Tensor:
        msg = self.msg_out(self.msg_net(torch.cat([edge_attr, x_j, x_i], dim=-1)))  # (n_edges, msg_dim)
        gate = self.gate_out(self.gate_net(msg)).squeeze(-1)  # (n_edges,)
        attn = softmax_pyg(gate, index)  # softmax over edges sharing a receiver
        return attn.unsqueeze(-1) * msg


class AttentionAggregationGNN(nn.Module):
    """Stack of AttentionAggregation layers.

    The message projection dim msg_dim is constant across layers; each layer's update
    output is msg_dim except the last, which is out_dim (matching upstream). Edge
    features are not updated between layers.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        msg_dim: int,
        out_dim: int,
        hid_msg: Sequence[int],
        hid_aggr: Sequence[int],
        hid_update: Sequence[int],
        n_layers: int,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.out_dim = out_dim

        layers = []
        cur_dim = in_dim
        for i in range(n_layers):
            layer_out_dim = out_dim if i == n_layers - 1 else msg_dim
            layers.append(
                AttentionAggregation(cur_dim, edge_dim, msg_dim, layer_out_dim, hid_msg, hid_aggr, hid_update)
            )
            cur_dim = layer_out_dim
        self.gnn_layers = nn.ModuleList(layers)

    def forward(self, graph: GraphData) -> torch.Tensor:
        """Pursuer-node embedding of every graph, (n_graphs, out_dim)."""
        for layer in self.gnn_layers:
            graph = layer(graph)
        return graph.agent_nodes(graph.nodes)
