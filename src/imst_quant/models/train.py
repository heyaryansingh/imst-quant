"""Training and prediction for forecasters (FORE-04, FORE-05)."""

from pathlib import Path
from typing import List, Literal

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .cnn import CNNClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier

FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "volatility_30d",
    "related_return_1",
    "related_return_2",
    "related_return_3",
    "sentiment_index",
    "post_count",
]


def _prepare_data(
    df: pl.DataFrame,
    window: int = 5,
    feature_cols: List[str] | None = None,
) -> tuple:
    cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    if not cols:
        cols = ["return_1d", "volatility_30d", "sentiment_index"]
    df = df.with_columns(
        (pl.col("return_1d").shift(-1) > 0).cast(pl.Int64).alias("target")
    )
    df = df.drop_nulls("target")
    X, y = [], []
    for asset in df["asset_id"].unique().to_list():
        adf = df.filter(pl.col("asset_id") == asset).sort("date")
        vals = adf.select(cols).fill_null(0).to_numpy()
        tgt = adf["target"].to_numpy()
        for i in range(window, len(vals) - 1):
            X.append(vals[i - window : i])
            y.append(tgt[i])
    if not X:
        return None, None
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_forecaster(
    features_path: Path,
    model_type: Literal["lstm", "cnn", "transformer"] = "lstm",
    output_path: Path | None = None,
    window: int = 5,
    epochs: int = 20,
    lr: float = 0.001,
) -> nn.Module:
    """Train forecaster on gold features."""
    df = pl.read_parquet(features_path)
    X, y = _prepare_data(df, window=window)
    if X is None or len(X) < 10:
        raise ValueError("Insufficient data")
    dim = X.shape[2]
    if model_type == "lstm":
        model = LSTMClassifier(input_dim=dim)
    elif model_type == "cnn":
        model = CNNClassifier(input_dim=dim)
    else:
        model = TransformerClassifier(input_dim=dim)
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=32,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            opt.step()
    model.eval()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
    return model


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Predict class probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return torch.softmax(logits, dim=-1)
