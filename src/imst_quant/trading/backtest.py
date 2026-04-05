"""Event-driven backtesting engine for trading strategy evaluation.

This module implements a simple but effective backtesting framework that
calculates daily PnL based on position signals and forward returns.

Example:
    >>> from imst_quant.trading.backtest import run_backtest
    >>> results = run_backtest(
    ...     features_path=Path("data/gold/features.parquet"),
    ...     transaction_cost=0.001,
    ... )
    >>> print(f"Sharpe: {results['sharpe']:.2f}")
"""

from pathlib import Path
from typing import Dict

import polars as pl


def run_backtest(
    features_path: Path,
    predictions: pl.DataFrame | None = None,
    transaction_cost: float = 0.001,
) -> Dict[str, float | int]:
    """Run a simple daily backtesting strategy.

    Calculates daily PnL as: signal * return_1d - transaction_cost * |signal|

    Args:
        features_path: Path to Parquet file containing features with
            columns: date, asset_id, return_1d.
        predictions: Optional DataFrame with columns: date, asset_id, prob_up.
            If None, uses a neutral (zero) signal for all days.
        transaction_cost: Trading cost per unit traded (default: 0.1%).

    Returns:
        Dictionary containing:
            - total_pnl: Cumulative profit/loss over the backtest period
            - sharpe: Sharpe ratio (PnL mean / std)
            - trades: Number of non-zero signal days

    Example:
        >>> results = run_backtest(Path("features.parquet"), transaction_cost=0.002)
        >>> print(f"Total PnL: {results['total_pnl']:.4f}")
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
