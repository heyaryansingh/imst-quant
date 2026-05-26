"""Backtest visualization utilities for analyzing trading strategy performance.

This module provides functions to create visual representations of backtest results,
including equity curves, drawdown plots, return distributions, and performance metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import numpy as np


def prepare_equity_data(
    df: pl.DataFrame,
    returns_col: str = "returns",
    timestamp_col: str = "timestamp",
    initial_capital: float = 10000.0,
) -> pl.DataFrame:
    """Prepare equity curve data from returns.

    Args:
        df: DataFrame with returns and timestamps
        returns_col: Column name containing returns
        timestamp_col: Column name containing timestamps
        initial_capital: Starting portfolio value

    Returns:
        DataFrame with equity curve, cumulative returns, and drawdowns
    """
    if df.is_empty():
        raise ValueError("Input DataFrame is empty")

    if returns_col not in df.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame")

    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    # Sort by timestamp
    df = df.sort(timestamp_col)

    # Calculate cumulative returns and equity
    df = df.with_columns([
        (1 + pl.col(returns_col)).cum_prod().alias("cum_returns"),
    ])

    df = df.with_columns([
        (pl.col("cum_returns") * initial_capital).alias("equity"),
        pl.col("cum_returns").cum_max().alias("running_max"),
    ])

    # Calculate drawdown
    df = df.with_columns([
        ((pl.col("equity") / (pl.col("running_max") * initial_capital)) - 1).alias("drawdown"),
    ])

    return df


def calculate_rolling_metrics(
    df: pl.DataFrame,
    returns_col: str = "returns",
    window: int = 252,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Calculate rolling performance metrics.

    Args:
        df: DataFrame with returns
        returns_col: Column name containing returns
        window: Rolling window size in periods
        periods_per_year: Number of periods per year for annualization

    Returns:
        DataFrame with rolling metrics
    """
    if window <= 0:
        raise ValueError("Window must be positive")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    # Rolling mean return
    df = df.with_columns([
        pl.col(returns_col)
        .rolling_mean(window_size=window)
        .alias("rolling_mean"),
    ])

    # Rolling standard deviation (volatility)
    df = df.with_columns([
        pl.col(returns_col)
        .rolling_std(window_size=window)
        .alias("rolling_std"),
    ])

    # Annualized metrics
    df = df.with_columns([
        (pl.col("rolling_mean") * periods_per_year).alias("rolling_annualized_return"),
        (pl.col("rolling_std") * np.sqrt(periods_per_year)).alias("rolling_annualized_vol"),
    ])

    # Rolling Sharpe ratio
    df = df.with_columns([
        (pl.col("rolling_annualized_return") / pl.col("rolling_annualized_vol")).alias("rolling_sharpe"),
    ])

    return df


def get_drawdown_periods(
    df: pl.DataFrame,
    drawdown_col: str = "drawdown",
    threshold: float = -0.05,
) -> List[Dict]:
    """Identify significant drawdown periods.

    Args:
        df: DataFrame with drawdown column
        drawdown_col: Column name containing drawdowns
        threshold: Minimum drawdown to consider (negative value)

    Returns:
        List of dictionaries with drawdown period details
    """
    if drawdown_col not in df.columns:
        raise ValueError(f"Column '{drawdown_col}' not found in DataFrame")

    if threshold >= 0:
        raise ValueError("Threshold must be negative")

    drawdowns = []
    in_drawdown = False
    start_idx = None
    max_dd = 0.0

    for idx, dd in enumerate(df[drawdown_col].to_list()):
        if dd <= threshold and not in_drawdown:
            # Start of drawdown period
            in_drawdown = True
            start_idx = idx
            max_dd = dd
        elif in_drawdown:
            if dd > threshold:
                # End of drawdown period
                drawdowns.append({
                    "start_idx": start_idx,
                    "end_idx": idx - 1,
                    "max_drawdown": max_dd,
                    "duration": idx - start_idx,
                })
                in_drawdown = False
                start_idx = None
                max_dd = 0.0
            else:
                # Continue tracking max drawdown
                max_dd = min(max_dd, dd)

    # Handle case where drawdown extends to end of data
    if in_drawdown:
        drawdowns.append({
            "start_idx": start_idx,
            "end_idx": len(df) - 1,
            "max_drawdown": max_dd,
            "duration": len(df) - start_idx,
        })

    return drawdowns


def calculate_monthly_returns(
    df: pl.DataFrame,
    returns_col: str = "returns",
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Calculate monthly return statistics.

    Args:
        df: DataFrame with returns and timestamps
        returns_col: Column name containing returns
        timestamp_col: Column name containing timestamps

    Returns:
        DataFrame with monthly aggregated returns
    """
    if returns_col not in df.columns or timestamp_col not in df.columns:
        raise ValueError("Required columns not found in DataFrame")

    # Ensure timestamp is datetime
    df = df.with_columns([
        pl.col(timestamp_col).cast(pl.Datetime).alias(timestamp_col),
    ])

    # Extract year and month
    df = df.with_columns([
        pl.col(timestamp_col).dt.year().alias("year"),
        pl.col(timestamp_col).dt.month().alias("month"),
    ])

    # Calculate cumulative returns per month
    monthly = df.group_by(["year", "month"]).agg([
        ((1 + pl.col(returns_col)).prod() - 1).alias("monthly_return"),
        pl.col(returns_col).count().alias("trading_days"),
    ]).sort(["year", "month"])

    return monthly


def analyze_return_distribution(
    df: pl.DataFrame,
    returns_col: str = "returns",
) -> Dict:
    """Analyze the distribution of returns.

    Args:
        df: DataFrame with returns
        returns_col: Column name containing returns

    Returns:
        Dictionary with distribution statistics
    """
    if returns_col not in df.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame")

    returns = df[returns_col].drop_nulls()

    if returns.is_empty():
        raise ValueError("No valid returns data")

    # Basic statistics
    mean = float(returns.mean())
    median = float(returns.median())
    std = float(returns.std())

    # Quantiles
    quantiles = returns.quantile(0.01), returns.quantile(0.05), returns.quantile(0.95), returns.quantile(0.99)

    # Skewness and kurtosis approximations
    returns_np = returns.to_numpy()
    n = len(returns_np)

    # Skewness
    skewness = float(np.sum((returns_np - mean) ** 3) / (n * std ** 3)) if std > 0 else 0.0

    # Excess kurtosis
    kurtosis = float(np.sum((returns_np - mean) ** 4) / (n * std ** 4) - 3) if std > 0 else 0.0

    # Positive/negative return counts
    positive_count = int((returns > 0).sum())
    negative_count = int((returns < 0).sum())

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "percentile_1": float(quantiles[0]),
        "percentile_5": float(quantiles[1]),
        "percentile_95": float(quantiles[2]),
        "percentile_99": float(quantiles[3]),
        "positive_returns": positive_count,
        "negative_returns": negative_count,
        "total_returns": len(returns),
    }


def generate_summary_stats(
    df: pl.DataFrame,
    returns_col: str = "returns",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Generate comprehensive summary statistics for backtest.

    Args:
        df: DataFrame with returns
        returns_col: Column name containing returns
        periods_per_year: Number of periods per year
        risk_free_rate: Daily risk-free rate

    Returns:
        Dictionary with summary statistics
    """
    if returns_col not in df.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame")

    returns = df[returns_col].drop_nulls()

    if returns.is_empty():
        raise ValueError("No valid returns data")

    # Prepare equity curve
    equity_df = prepare_equity_data(df, returns_col=returns_col)

    # Basic metrics
    total_return = float((1 + returns).prod() - 1)
    n_periods = len(returns)
    annualized_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)

    # Volatility
    volatility = float(returns.std() * np.sqrt(periods_per_year))

    # Sharpe ratio
    excess_return = annualized_return - (risk_free_rate * periods_per_year)
    sharpe = float(excess_return / volatility) if volatility > 0 else 0.0

    # Sortino ratio
    downside_returns = returns.filter(pl.col(returns_col) < 0)
    downside_std = float(downside_returns.std() * np.sqrt(periods_per_year)) if len(downside_returns) > 0 else 0.0
    sortino = float(excess_return / downside_std) if downside_std > 0 else 0.0

    # Max drawdown
    max_dd = float(equity_df["drawdown"].min())

    # Calmar ratio
    calmar = float(annualized_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Win rate
    winning_trades = int((returns > 0).sum())
    total_trades = len(returns)
    win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "periods": n_periods,
    }


def export_analysis_data(
    df: pl.DataFrame,
    output_path: Path,
    include_rolling: bool = True,
    rolling_window: int = 252,
) -> None:
    """Export processed analysis data to parquet.

    Args:
        df: DataFrame with backtest results
        output_path: Path to save output file
        include_rolling: Whether to include rolling metrics
        rolling_window: Window size for rolling calculations
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare equity curve
    result_df = prepare_equity_data(df)

    # Add rolling metrics if requested
    if include_rolling:
        result_df = calculate_rolling_metrics(result_df, window=rolling_window)

    # Save to parquet
    result_df.write_parquet(output_path)


def create_performance_report(
    df: pl.DataFrame,
    returns_col: str = "returns",
    timestamp_col: str = "timestamp",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Create comprehensive performance report.

    Args:
        df: DataFrame with backtest results
        returns_col: Column name containing returns
        timestamp_col: Column name containing timestamps
        periods_per_year: Number of periods per year
        risk_free_rate: Daily risk-free rate

    Returns:
        Dictionary with complete performance analysis
    """
    # Summary statistics
    summary = generate_summary_stats(
        df,
        returns_col=returns_col,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    # Return distribution
    distribution = analyze_return_distribution(df, returns_col=returns_col)

    # Monthly returns
    monthly = calculate_monthly_returns(df, returns_col=returns_col, timestamp_col=timestamp_col)

    # Drawdown analysis
    equity_df = prepare_equity_data(df, returns_col=returns_col, timestamp_col=timestamp_col)
    drawdowns = get_drawdown_periods(equity_df, threshold=-0.05)

    return {
        "summary": summary,
        "distribution": distribution,
        "monthly_returns": monthly.to_dicts(),
        "major_drawdowns": drawdowns,
        "total_drawdown_periods": len(drawdowns),
    }
