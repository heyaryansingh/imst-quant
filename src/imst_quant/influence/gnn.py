"""2-layer GCN for influence scoring (INF-02)."""

from typing import Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class InfluenceGNN(nn.Module):
    """2-layer GCN for influence scoring."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return torch.sigmoid(x).squeeze(-1)


def prepare_gnn_data(G: nx.DiGraph) -> Data:
    """Convert NetworkX graph to PyG Data."""
    if G.number_of_nodes() == 0:
        return None

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    features = []
    for n in node_list:
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        w_in = sum(G[u][n].get("weight", 1) for u in G.predecessors(n))
        w_out = sum(G[n][v].get("weight", 1) for v in G.successors(n))
        features.append([float(in_d), float(out_d), float(w_in), float(w_out)])

    x = torch.tensor(features, dtype=torch.float32)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    edge_list = list(G.edges())
    edge_index = torch.tensor(
        [
            [node_to_idx[e[0]] for e in edge_list],
            [node_to_idx[e[1]] for e in edge_list],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [G[e[0]][e[1]].get("weight", 1.0) for e in edge_list],
        dtype=torch.float32,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_mapping = idx_to_node
    data.node_list = node_list
    return data
