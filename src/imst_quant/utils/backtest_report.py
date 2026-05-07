"""Comprehensive backtest reporting utilities.

This module generates detailed backtest reports with performance metrics,
visualizations, and analysis.

Functions:
    generate_backtest_report: Create comprehensive backtest summary
    calculate_performance_stats: Calculate all relevant performance metrics
    generate_trade_log: Create detailed trade-by-trade log
    compare_strategies: Compare multiple strategy backtests
"""

from typing import Dict, List, Optional, Tuple
import polars as pl
from datetime import datetime


def generate_backtest_report(
    equity_curve: pl.DataFrame,
    trades: Optional[pl.DataFrame] = None,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, any]:
    """Generate comprehensive backtest report.

    Args:
        equity_curve: DataFrame with columns: date, equity, returns
        trades: Optional DataFrame with trade records
        initial_capital: Starting capital for the backtest
        risk_free_rate: Daily risk-free rate for Sharpe calculation

    Returns:
        Dictionary containing:
        - summary: High-level performance summary
        - returns_stats: Return distribution statistics
        - risk_metrics: Risk-adjusted performance metrics
        - drawdown_stats: Drawdown analysis
        - trade_stats: Trade-level statistics (if trades provided)
    """
    report = {
        "summary": _calculate_summary_stats(equity_curve, initial_capital),
        "returns_stats": _calculate_returns_stats(equity_curve),
        "risk_metrics": _calculate_risk_metrics(equity_curve, risk_free_rate),
        "drawdown_stats": _calculate_drawdown_stats(equity_curve),
    }

    if trades is not None and not trades.is_empty():
        report["trade_stats"] = _calculate_trade_stats(trades)

    return report


def _calculate_summary_stats(
    df: pl.DataFrame,
    initial_capital: float,
) -> Dict[str, float]:
    """Calculate high-level summary statistics."""
    if df.is_empty():
        return {}

    final_equity = df["equity"][-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Calculate days
    start_date = df["date"][0]
    end_date = df["date"][-1]
    total_days = (end_date - start_date).days

    # Annualize return
    years = total_days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "annual_return": annual_return,
        "annual_return_pct": annual_return * 100,
        "total_days": total_days,
        "peak_equity": df["equity"].max(),
        "lowest_equity": df["equity"].min(),
    }


def _calculate_returns_stats(df: pl.DataFrame) -> Dict[str, float]:
    """Calculate return distribution statistics."""
    if df.is_empty() or "returns" not in df.columns:
        return {}

    returns = df["returns"]

    # Filter out null returns
    returns_clean = returns.drop_nulls()

    if returns_clean.is_empty():
        return {}

    positive_returns = returns_clean.filter(pl.col("returns") > 0)
    negative_returns = returns_clean.filter(pl.col("returns") < 0)

    return {
        "mean_return": returns_clean.mean(),
        "median_return": returns_clean.median(),
        "std_return": returns_clean.std(),
        "min_return": returns_clean.min(),
        "max_return": returns_clean.max(),
        "skewness": returns_clean.skew(),
        "kurtosis": returns_clean.kurtosis(),
        "positive_days": len(positive_returns),
        "negative_days": len(negative_returns),
        "positive_pct": len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0.0,
    }


def _calculate_risk_metrics(
    df: pl.DataFrame,
    risk_free_rate: float,
) -> Dict[str, float]:
    """Calculate risk-adjusted performance metrics."""
    if df.is_empty() or "returns" not in df.columns:
        return {}

    returns = df["returns"].drop_nulls()

    if returns.is_empty():
        return {}

    # Sharpe Ratio
    excess_returns = returns - risk_free_rate
    sharpe = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5) if excess_returns.std() > 0 else 0.0

    # Sortino Ratio (downside deviation)
    downside_returns = returns.filter(pl.col("returns") < risk_free_rate)
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (excess_returns.mean() / downside_std) * (252 ** 0.5) if downside_std > 0 else 0.0

    # Calmar Ratio (return / max drawdown)
    max_dd = _calculate_max_drawdown(df)
    annual_return = returns.mean() * 252
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "volatility": returns.std() * (252 ** 0.5),
        "downside_deviation": downside_std * (252 ** 0.5),
    }


def _calculate_drawdown_stats(df: pl.DataFrame) -> Dict[str, float]:
    """Calculate drawdown statistics."""
    if df.is_empty() or "equity" not in df.columns:
        return {}

    # Calculate running maximum
    df_dd = df.with_columns(
        pl.col("equity").cummax().alias("running_max")
    )

    # Calculate drawdown
    df_dd = df_dd.with_columns(
        ((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias("drawdown")
    )

    max_drawdown = df_dd["drawdown"].min()

    # Find max drawdown period
    max_dd_idx = df_dd["drawdown"].arg_min()
    max_dd_date = df_dd["date"][max_dd_idx]

    # Calculate recovery time
    recovery_idx = None
    if max_dd_idx < len(df_dd) - 1:
        recovery_equity = df_dd["running_max"][max_dd_idx]
        future_equity = df_dd[max_dd_idx + 1:]["equity"]

        recovery_mask = future_equity >= recovery_equity
        if recovery_mask.any():
            recovery_idx = max_dd_idx + 1 + recovery_mask.arg_max()

    recovery_days = (df_dd["date"][recovery_idx] - max_dd_date).days if recovery_idx else None

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "max_drawdown_date": max_dd_date,
        "recovery_days": recovery_days,
        "avg_drawdown": df_dd["drawdown"].mean(),
        "current_drawdown": df_dd["drawdown"][-1],
    }


def _calculate_max_drawdown(df: pl.DataFrame) -> float:
    """Calculate maximum drawdown."""
    if df.is_empty() or "equity" not in df.columns:
        return 0.0

    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max

    return drawdown.min()


def _calculate_trade_stats(trades: pl.DataFrame) -> Dict[str, any]:
    """Calculate trade-level statistics."""
    if trades.is_empty():
        return {}

    # Assume trades has: entry_time, exit_time, pnl, asset
    total_trades = len(trades)
    winning_trades = trades.filter(pl.col("pnl") > 0)
    losing_trades = trades.filter(pl.col("pnl") < 0)

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    return {
        "total_trades": total_trades,
        "winning_trades": num_wins,
        "losing_trades": num_losses,
        "win_rate": num_wins / total_trades if total_trades > 0 else 0.0,
        "avg_win": winning_trades["pnl"].mean() if num_wins > 0 else 0.0,
        "avg_loss": losing_trades["pnl"].mean() if num_losses > 0 else 0.0,
        "largest_win": winning_trades["pnl"].max() if num_wins > 0 else 0.0,
        "largest_loss": losing_trades["pnl"].min() if num_losses > 0 else 0.0,
        "total_pnl": trades["pnl"].sum(),
        "avg_pnl": trades["pnl"].mean(),
    }


def calculate_performance_stats(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive performance statistics.

    Args:
        returns: Series of period returns
        risk_free_rate: Risk-free rate for the period
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Dictionary with performance metrics
    """
    if returns.is_empty():
        return {}

    # Basic stats
    mean_return = returns.mean()
    std_return = returns.std()

    # Annualized metrics
    annual_return = mean_return * periods_per_year
    annual_vol = std_return * (periods_per_year ** 0.5)

    # Sharpe ratio
    excess_return = mean_return - risk_free_rate
    sharpe = (excess_return / std_return) * (periods_per_year ** 0.5) if std_return > 0 else 0.0

    # Downside metrics
    downside_returns = returns.filter(pl.col("returns") < 0)
    downside_vol = downside_returns.std() * (periods_per_year ** 0.5) if len(downside_returns) > 0 else 0.0

    return {
        "mean_return": mean_return,
        "annual_return": annual_return,
        "volatility": std_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "downside_volatility": downside_vol,
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def generate_trade_log(
    trades: pl.DataFrame,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """Generate detailed trade-by-trade log with P&L analysis.

    Args:
        trades: DataFrame with trade records
        output_path: Optional path to save CSV

    Returns:
        Enriched DataFrame with cumulative P&L and analysis
    """
    if trades.is_empty():
        return pl.DataFrame()

    # Sort by exit time
    if "exit_time" in trades.columns:
        trades = trades.sort("exit_time")

    # Calculate cumulative P&L
    if "pnl" in trades.columns:
        trades = trades.with_columns(
            pl.col("pnl").cumsum().alias("cumulative_pnl")
        )

        # Calculate running metrics
        trades = trades.with_columns([
            (pl.col("pnl") > 0).cast(pl.Int32).cumsum().alias("total_wins"),
            (pl.col("pnl") <= 0).cast(pl.Int32).cumsum().alias("total_losses"),
        ])

        trades = trades.with_columns(
            (pl.col("total_wins") / (pl.col("total_wins") + pl.col("total_losses"))).alias("running_win_rate")
        )

    if output_path:
        trades.write_csv(output_path)

    return trades


def compare_strategies(
    strategies: Dict[str, pl.DataFrame],
    metric: str = "sharpe_ratio",
) -> pl.DataFrame:
    """Compare multiple strategy backtests.

    Args:
        strategies: Dictionary mapping strategy names to equity curves
        metric: Primary metric for comparison

    Returns:
        DataFrame with comparative statistics
    """
    comparison = []

    for name, df in strategies.items():
        if df.is_empty():
            continue

        report = generate_backtest_report(df)

        comparison.append({
            "strategy": name,
            "total_return": report["summary"].get("total_return_pct", 0),
            "annual_return": report["summary"].get("annual_return_pct", 0),
            "sharpe_ratio": report["risk_metrics"].get("sharpe_ratio", 0),
            "sortino_ratio": report["risk_metrics"].get("sortino_ratio", 0),
            "max_drawdown": report["drawdown_stats"].get("max_drawdown_pct", 0),
            "volatility": report["risk_metrics"].get("volatility", 0),
        })

    comparison_df = pl.DataFrame(comparison)

    if not comparison_df.is_empty():
        comparison_df = comparison_df.sort(metric, descending=True)

    return comparison_df
