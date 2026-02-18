"""Tests for forecasting models (Phase 7)."""

import tempfile
from pathlib import Path

import polars as pl
import torch
import pytest

from imst_quant.models import (
    LSTMClassifier,
    CNNClassifier,
    TransformerClassifier,
    train_forecaster,
    predict,
)


def test_lstm_forward():
    """LSTM forward pass."""
    model = LSTMClassifier(input_dim=6, hidden_dim=16, num_layers=1)
    x = torch.randn(2, 5, 6)
    out = model(x)
    assert out.shape == (2, 2)


def test_cnn_forward():
    """CNN forward pass."""
    model = CNNClassifier(input_dim=6, num_filters=16)
    x = torch.randn(2, 5, 6)
    out = model(x)
    assert out.shape == (2, 2)


def test_transformer_forward():
    """Transformer forward pass."""
    model = TransformerClassifier(input_dim=6, d_model=32, nhead=2)
    x = torch.randn(2, 5, 6)
    out = model(x)
    assert out.shape == (2, 2)


def test_train_forecaster(tmp_path):
    """FORE-04: Train LSTM on synthetic features."""
    df = pl.DataFrame({
        "date": [f"2024-01-{i:02d}" for i in range(1, 31)] * 2,
        "asset_id": ["AAPL"] * 30 + ["JPM"] * 30,
        "return_1d": [0.01] * 60,
        "return_5d": [0.02] * 60,
        "volatility_30d": [0.01] * 60,
        "related_return_1": [0.0] * 60,
        "related_return_2": [0.0] * 60,
        "related_return_3": [0.0] * 60,
        "sentiment_index": [0.1] * 60,
        "post_count": [5] * 60,
    })
    fp = tmp_path / "features.parquet"
    df.write_parquet(fp)
    model = train_forecaster(fp, model_type="lstm", output_path=tmp_path / "model.pt", epochs=5)
    assert model is not None
    x = torch.randn(1, 5, 8)
    probs = predict(model, x)
    assert probs.shape == (1, 2)
    assert (probs >= 0).all() and (probs <= 1).all()
