"""Tests for influence graph and GCN (Phase 4)."""

import polars as pl
import pytest

from imst_quant.influence import (
    InteractionGraph,
    InfluenceGNN,
    InfluenceTrainer,
    build_monthly_graph,
    prepare_gnn_data,
)


def test_interaction_graph_build():
    """INF-01, INF-04: Graph with min_interactions filter."""
    df = pl.DataFrame({
        "author_id": ["a1", "a2", "a1", "a2", "a3"] * 20,
        "subreddit": ["stocks"] * 100,
        "date": ["2024-01-15"] * 100,
    })
    builder = InteractionGraph(min_interactions=10)
    G = builder.build_from_dataframe(df, 2024, 1)
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1


def test_gnn_forward():
    """INF-02: GCN forward pass."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    data = prepare_gnn_data(G)
    assert data is not None
    model = InfluenceGNN(input_dim=4, hidden_dim=8, output_dim=1)
    out = model(data)
    assert out.shape == (3,)
    assert (out >= 0).all() and (out <= 1).all()


def test_trainer_produces_scores():
    """INF-03: Trainer outputs influence scores."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("a", "c")])
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    model = InfluenceGNN(input_dim=4, hidden_dim=8, output_dim=1)
    trainer = InfluenceTrainer(model, epochs=5)
    scores = trainer.train(G)
    assert len(scores) == 3
    assert all(0 <= v <= 1 for v in scores.values())
