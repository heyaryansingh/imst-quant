"""LightGBM forecaster (FORE-U01) - production upgrade."""

from pathlib import Path
from typing import List

import polars as pl


def train_lightgbm(
    features_path: Path,
    output_path: Path | None = None,
    feature_cols: List[str] | None = None,
) -> object:
    """Train LightGBM. Returns None if lightgbm not installed."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    df = pl.read_parquet(features_path)
    cols = feature_cols or ["return_1d", "volatility_30d", "sentiment_index"]
    cols = [c for c in cols if c in df.columns]
    df = df.with_columns((pl.col("return_1d").shift(-1) > 0).cast(pl.Int64).alias("target"))
    df = df.drop_nulls("target")
    X = df.select(cols).fill_null(0).to_numpy()
    y = df["target"].to_numpy()
    model = lgb.LGBMClassifier(n_estimators=50, verbosity=-1)
    model.fit(X, y)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(output_path))
    return model
