"""Graph Neural Network for social influence scoring (INF-02).

This module implements a 2-layer Graph Convolutional Network (GCN) for
computing influence scores in social networks. The model learns to
predict author influence based on network topology and node features.

The GNN takes a directed graph where:
    - Nodes represent authors/users
    - Edges represent interactions (replies, mentions, retweets)
    - Edge weights indicate interaction frequency/strength

Features computed per node:
    - In-degree and out-degree
    - Weighted in-degree and out-degree

Classes:
    InfluenceGNN: 2-layer GCN for influence prediction

Functions:
    prepare_gnn_data: Convert NetworkX graph to PyTorch Geometric Data

Example:
    >>> import networkx as nx
    >>> from imst_quant.influence.gnn import InfluenceGNN, prepare_gnn_data
    >>> G = nx.DiGraph()
    >>> G.add_edge("user1", "user2", weight=1.5)
    >>> data = prepare_gnn_data(G)
    >>> model = InfluenceGNN()
    >>> scores = model(data)
"""

from typing import Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class InfluenceGNN(nn.Module):
    """2-layer Graph Convolutional Network for influence scoring.

    Learns node representations that capture structural importance in
    the social network, outputting influence scores in range [0, 1].

    Attributes:
        conv1: First GCN layer (input_dim -> hidden_dim).
        conv2: Second GCN layer (hidden_dim -> output_dim).
        dropout: Dropout probability for regularization.

    Example:
        >>> model = InfluenceGNN(input_dim=4, hidden_dim=32)
        >>> scores = model(data)  # Returns tensor of shape (num_nodes,)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the InfluenceGNN model.

        Args:
            input_dim: Dimension of input node features. Defaults to 4
                (in_degree, out_degree, weighted_in, weighted_out).
            hidden_dim: Hidden layer dimension. Defaults to 64.
            output_dim: Output dimension per node. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the GCN.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features of shape (num_nodes, input_dim)
                - edge_index: Edge indices of shape (2, num_edges)
                - edge_attr (optional): Edge weights of shape (num_edges,)

        Returns:
            Influence scores tensor of shape (num_nodes,) with values
            in range [0, 1] via sigmoid activation.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return torch.sigmoid(x).squeeze(-1)


def prepare_gnn_data(G: nx.DiGraph) -> Data:
    """Convert NetworkX DiGraph to PyTorch Geometric Data object.

    Transforms a NetworkX directed graph into a format suitable for
    GNN processing. Computes node features from graph structure and
    normalizes them using z-score standardization.

    Node features computed (4 dimensions):
        - In-degree: Number of incoming edges
        - Out-degree: Number of outgoing edges
        - Weighted in-degree: Sum of incoming edge weights
        - Weighted out-degree: Sum of outgoing edge weights

    Args:
        G: NetworkX directed graph with optional 'weight' edge attribute.
            Edges without weight attribute default to weight=1.

    Returns:
        PyTorch Geometric Data object with:
            - x: Normalized node features (num_nodes, 4)
            - edge_index: Edge indices (2, num_edges)
            - edge_attr: Edge weights (num_edges,)
            - node_mapping: Dict mapping index to original node ID
            - node_list: List of original node IDs in order
        Returns None if graph has no nodes.

    Example:
        >>> G = nx.DiGraph()
        >>> G.add_weighted_edges_from([("a", "b", 2.0), ("b", "c", 1.0)])
        >>> data = prepare_gnn_data(G)
        >>> data.x.shape  # torch.Size([3, 4])
    """
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
