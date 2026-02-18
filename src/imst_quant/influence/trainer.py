"""Influence GNN training with pseudo-labels (INF-03)."""

from typing import Dict

import networkx as nx
import torch
from torch_geometric.data import Data

from .gnn import InfluenceGNN, prepare_gnn_data


class InfluenceTrainer:
    """Train InfluenceGNN with PageRank pseudo-labels."""

    def __init__(
        self,
        model: InfluenceGNN,
        lr: float = 0.01,
        epochs: int = 100,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, G: nx.DiGraph) -> Dict[str, float]:
        """Train on graph, return dict of author_id -> influence score."""
        data = prepare_gnn_data(G)
        if data is None or data.x.shape[0] < 2:
            return {}

        pr = nx.pagerank(G, weight="weight")
        node_list = data.node_list
        labels = torch.tensor(
            [pr.get(n, 0.0) for n in node_list],
            dtype=torch.float32,
        )
        labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = torch.nn.functional.mse_loss(out, labels)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            scores = self.model(data)
        return {node_list[i]: float(scores[i]) for i in range(len(node_list))}
