"""Trade analytics utilities for comprehensive performance analysis.

This module provides advanced trade-level analytics beyond basic metrics,
including win rate analysis, trade duration statistics, and equity curve
analysis.

Functions:
    calculate_trade_metrics: Comprehensive trade-level metrics
    analyze_trade_distribution: Distribution analysis of trade returns
    calculate_consecutive_metrics: Analyze winning/losing streaks
    calculate_monthly_returns: Aggregate returns by month
"""

from typing import Dict, List, Optional, Tuple
import polars as pl


def calculate_trade_metrics(
    df: pl.DataFrame,
    pnl_col: str = "pnl",
    entry_col: str = "entry_time",
    exit_col: str = "exit_time",
) -> Dict[str, float]:
    """Calculate comprehensive trade-level performance metrics.

    Args:
        df: DataFrame containing trade records with PnL and timing data
        pnl_col: Column name for profit/loss values
        entry_col: Column name for trade entry timestamps
        exit_col: Column name for trade exit timestamps

    Returns:
        Dictionary containing:
        - total_trades: Total number of trades
        - winning_trades: Number of profitable trades
        - losing_trades: Number of losing trades
        - win_rate: Percentage of winning trades
        - avg_win: Average profit per winning trade
        - avg_loss: Average loss per losing trade
        - expectancy: Expected value per trade
        - profit_factor: Ratio of gross profit to gross loss
        - avg_trade_duration_hours: Average time in position
    """
    if df.is_empty():
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration_hours": 0.0,
        }

    total_trades = len(df)
    winning_trades = df.filter(pl.col(pnl_col) > 0)
    losing_trades = df.filter(pl.col(pnl_col) < 0)

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    win_rate = num_wins / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades[pnl_col].mean() if num_wins > 0 else 0.0
    avg_loss = abs(losing_trades[pnl_col].mean()) if num_losses > 0 else 0.0

    gross_profit = winning_trades[pnl_col].sum() if num_wins > 0 else 0.0
    gross_loss = abs(losing_trades[pnl_col].sum()) if num_losses > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Calculate average trade duration
    if entry_col in df.columns and exit_col in df.columns:
        df_with_duration = df.with_columns(
            ((pl.col(exit_col) - pl.col(entry_col)).dt.total_seconds() / 3600).alias("duration_hours")
        )
        avg_duration = df_with_duration["duration_hours"].mean()
    else:
        avg_duration = 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": num_wins,
        "losing_trades": num_losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_trade_duration_hours": avg_duration,
    }


def analyze_trade_distribution(
    df: pl.DataFrame,
    pnl_col: str = "pnl",
    bins: int = 20,
) -> Dict[str, any]:
    """Analyze the distribution of trade returns.

    Args:
        df: DataFrame containing trade PnL data
        pnl_col: Column name for profit/loss values
        bins: Number of bins for histogram analysis

    Returns:
        Dictionary containing distribution statistics:
        - mean: Mean return
        - median: Median return
        - std: Standard deviation
        - skewness: Return skewness
        - kurtosis: Return kurtosis
        - percentiles: 5th, 25th, 75th, 95th percentiles
    """
    if df.is_empty():
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "percentiles": {"5th": 0.0, "25th": 0.0, "75th": 0.0, "95th": 0.0},
        }

    pnl = df[pnl_col]

    return {
        "mean": pnl.mean(),
        "median": pnl.median(),
        "std": pnl.std(),
        "skewness": pnl.skew(),
        "kurtosis": pnl.kurtosis(),
        "percentiles": {
            "5th": pnl.quantile(0.05),
            "25th": pnl.quantile(0.25),
            "75th": pnl.quantile(0.75),
            "95th": pnl.quantile(0.95),
        },
    }


def calculate_consecutive_metrics(
    df: pl.DataFrame,
    pnl_col: str = "pnl",
) -> Dict[str, int]:
    """Analyze winning and losing streaks.

    Args:
        df: DataFrame containing trade PnL data sorted by time
        pnl_col: Column name for profit/loss values

    Returns:
        Dictionary containing:
        - max_win_streak: Longest consecutive winning streak
        - max_loss_streak: Longest consecutive losing streak
        - current_streak: Current streak (positive=wins, negative=losses)
    """
    if df.is_empty():
        return {
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "current_streak": 0,
        }

    # Create win/loss indicator
    df_streaks = df.with_columns(
        (pl.col(pnl_col) > 0).cast(pl.Int32).alias("is_win")
    )

    # Calculate streak changes
    df_streaks = df_streaks.with_columns(
        (pl.col("is_win") != pl.col("is_win").shift(1)).alias("streak_change")
    )

    # Group by streaks
    df_streaks = df_streaks.with_columns(
        pl.col("streak_change").cumsum().alias("streak_id")
    )

    # Calculate streak lengths
    streak_lengths = (
        df_streaks
        .group_by(["streak_id", "is_win"])
        .agg(pl.len().alias("length"))
    )

    win_streaks = streak_lengths.filter(pl.col("is_win") == 1)
    loss_streaks = streak_lengths.filter(pl.col("is_win") == 0)

    max_win_streak = win_streaks["length"].max() if len(win_streaks) > 0 else 0
    max_loss_streak = loss_streaks["length"].max() if len(loss_streaks) > 0 else 0

    # Current streak
    last_streak = streak_lengths.tail(1)
    if len(last_streak) > 0:
        current_length = last_streak["length"][0]
        is_win = last_streak["is_win"][0]
        current_streak = current_length if is_win else -current_length
    else:
        current_streak = 0

    return {
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "current_streak": int(current_streak),
    }


def calculate_monthly_returns(
    df: pl.DataFrame,
    pnl_col: str = "pnl",
    date_col: str = "exit_time",
) -> pl.DataFrame:
    """Aggregate trade returns by month.

    Args:
        df: DataFrame containing trade data with timestamps
        pnl_col: Column name for profit/loss values
        date_col: Column name for exit timestamps

    Returns:
        DataFrame with columns:
        - year_month: YYYY-MM format
        - total_pnl: Total PnL for the month
        - num_trades: Number of trades in the month
        - win_rate: Win rate for the month
        - avg_pnl: Average PnL per trade
    """
    if df.is_empty():
        return pl.DataFrame()

    monthly = (
        df
        .with_columns(
            pl.col(date_col).dt.strftime("%Y-%m").alias("year_month")
        )
        .group_by("year_month")
        .agg([
            pl.col(pnl_col).sum().alias("total_pnl"),
            pl.len().alias("num_trades"),
            (pl.col(pnl_col) > 0).mean().alias("win_rate"),
            pl.col(pnl_col).mean().alias("avg_pnl"),
        ])
        .sort("year_month")
    )

    return monthly


def calculate_drawdown_duration(
    df: pl.DataFrame,
    equity_col: str = "equity",
) -> Dict[str, float]:
    """Calculate drawdown duration statistics.

    Args:
        df: DataFrame with equity curve (sorted by time)
        equity_col: Column name for equity values

    Returns:
        Dictionary containing:
        - max_drawdown_duration_days: Longest time to recover from drawdown
        - avg_drawdown_duration_days: Average recovery time
        - current_drawdown_duration_days: Current time in drawdown
    """
    if df.is_empty() or equity_col not in df.columns:
        return {
            "max_drawdown_duration_days": 0.0,
            "avg_drawdown_duration_days": 0.0,
            "current_drawdown_duration_days": 0.0,
        }

    # Calculate running maximum
    df_dd = df.with_columns(
        pl.col(equity_col).cummax().alias("running_max")
    )

    # Calculate drawdown
    df_dd = df_dd.with_columns(
        (pl.col(equity_col) < pl.col("running_max")).alias("in_drawdown")
    )

    # Identify drawdown periods
    df_dd = df_dd.with_columns(
        (pl.col("in_drawdown") != pl.col("in_drawdown").shift(1)).alias("dd_change")
    )

    df_dd = df_dd.with_columns(
        pl.col("dd_change").cumsum().alias("dd_period_id")
    )

    # Calculate duration of each drawdown period
    dd_periods = (
        df_dd
        .filter(pl.col("in_drawdown"))
        .group_by("dd_period_id")
        .agg(pl.len().alias("duration"))
    )

    if len(dd_periods) == 0:
        return {
            "max_drawdown_duration_days": 0.0,
            "avg_drawdown_duration_days": 0.0,
            "current_drawdown_duration_days": 0.0,
        }

    max_dd_duration = dd_periods["duration"].max()
    avg_dd_duration = dd_periods["duration"].mean()

    # Check if currently in drawdown
    currently_in_dd = df_dd["in_drawdown"][-1]
    if currently_in_dd:
        current_dd_id = df_dd["dd_period_id"][-1]
        current_dd_duration = len(df_dd.filter(pl.col("dd_period_id") == current_dd_id))
    else:
        current_dd_duration = 0

    return {
        "max_drawdown_duration_days": float(max_dd_duration),
        "avg_drawdown_duration_days": float(avg_dd_duration),
        "current_drawdown_duration_days": float(current_dd_duration),
    }
