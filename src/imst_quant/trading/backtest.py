"""Event-driven backtest (BACK-01)."""

from pathlib import Path
from typing import Dict

import polars as pl


def run_backtest(
    features_path: Path,
    predictions: pl.DataFrame | None = None,
    transaction_cost: float = 0.001,
) -> Dict:
    """
    Simple backtest: daily PnL = signal * return_1d - cost.
    predictions: date, asset_id, prob_up (or signal).
    """
    df = pl.read_parquet(features_path)
    if predictions is not None:
        df = df.join(
            predictions.select(["date", "asset_id", "prob_up"]),
            on=["date", "asset_id"],
            how="left",
        )
        df = df.with_columns(
            pl.when(pl.col("prob_up") > 0.5)
            .then(1)
            .when(pl.col("prob_up") < 0.5)
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("signal"))

    if "return_1d" not in df.columns:
        return {"total_pnl": 0.0, "sharpe": 0.0, "trades": 0}

    df = df.with_columns(
        (pl.col("signal").cast(pl.Float64) * pl.col("return_1d") - transaction_cost * pl.col("signal").abs()).alias("pnl")
    )
    total = df["pnl"].sum()
    std = df["pnl"].std()
    sharpe = float(total / (std + 1e-8)) if std and std > 0 else 0.0
    trades = int(df.filter(pl.col("signal").abs() > 0).height)
    return {"total_pnl": float(total), "sharpe": sharpe, "trades": trades}
