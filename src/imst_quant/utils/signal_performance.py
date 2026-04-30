"""Signal performance analysis and metrics.

Analyzes the performance of trading signals including win rate, profit factor,
average win/loss, consecutive wins/losses, and signal quality metrics.

Functions:
    analyze_signal_performance: Comprehensive signal analysis
    calculate_win_metrics: Win rate and related statistics
    calculate_profit_factor: Profit factor from wins and losses
    signal_quality_score: Overall quality assessment (0-100)

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.signal_performance import analyze_signal_performance
    >>> df = pl.DataFrame({
    ...     "signal": [1, -1, 1, 0, 1, -1],
    ...     "returns": [0.02, -0.01, 0.03, 0.00, -0.015, 0.025]
    ... })
    >>> metrics = analyze_signal_performance(df)
    >>> print(f"Win Rate: {metrics['win_rate']:.2%}")
"""

from typing import Dict, Optional
import polars as pl


def calculate_win_metrics(
    df: pl.DataFrame,
    signal_col: str = "signal",
    returns_col: str = "returns",
) -> Dict[str, float]:
    """Calculate win rate and related statistics.

    Args:
        df: DataFrame with signal and returns columns
        signal_col: Name of signal column
        returns_col: Name of returns column

    Returns:
        Dictionary with win_rate, avg_win, avg_loss, win_loss_ratio
    """
    # Filter for actual trades (signal != 0)
    trades = df.filter(pl.col(signal_col) != 0)

    if trades.height == 0:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "total_trades": 0,
        }

    # Calculate trade PnL (signal * returns)
    trades = trades.with_columns(
        (pl.col(signal_col) * pl.col(returns_col)).alias("pnl")
    )

    wins = trades.filter(pl.col("pnl") > 0)
    losses = trades.filter(pl.col("pnl") < 0)

    win_rate = wins.height / trades.height if trades.height > 0 else 0.0
    avg_win = wins["pnl"].mean() if wins.height > 0 else 0.0
    avg_loss = abs(losses["pnl"].mean()) if losses.height > 0 else 0.0

    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0

    return {
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_loss_ratio": float(win_loss_ratio),
        "total_trades": trades.height,
    }


def calculate_profit_factor(
    df: pl.DataFrame,
    signal_col: str = "signal",
    returns_col: str = "returns",
) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        df: DataFrame with signal and returns columns
        signal_col: Name of signal column
        returns_col: Name of returns column

    Returns:
        Profit factor (values > 1 indicate profitable strategy)
    """
    trades = df.filter(pl.col(signal_col) != 0)

    if trades.height == 0:
        return 0.0

    trades = trades.with_columns(
        (pl.col(signal_col) * pl.col(returns_col)).alias("pnl")
    )

    gross_profit = trades.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(trades.filter(pl.col("pnl") < 0)["pnl"].sum())

    if gross_loss == 0:
        return 100.0 if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def signal_quality_score(
    df: pl.DataFrame,
    signal_col: str = "signal",
    returns_col: str = "returns",
) -> float:
    """Calculate overall signal quality score (0-100).

    Combines multiple metrics into a single quality score:
    - Win rate (0-50 points)
    - Profit factor (0-25 points)
    - Win/loss ratio (0-25 points)

    Args:
        df: DataFrame with signal and returns columns
        signal_col: Name of signal column
        returns_col: Name of returns column

    Returns:
        Quality score from 0 (worst) to 100 (best)
    """
    metrics = calculate_win_metrics(df, signal_col, returns_col)
    profit_factor = calculate_profit_factor(df, signal_col, returns_col)

    # Win rate score (50 points max)
    win_rate_score = min(metrics["win_rate"] * 100, 50)

    # Profit factor score (25 points max)
    # PF of 2.0 or higher = max points
    pf_score = min(profit_factor / 2.0 * 25, 25)

    # Win/loss ratio score (25 points max)
    # Ratio of 2.0 or higher = max points
    wl_score = min(metrics["win_loss_ratio"] / 2.0 * 25, 25)

    return float(win_rate_score + pf_score + wl_score)


def analyze_signal_performance(
    df: pl.DataFrame,
    signal_col: str = "signal",
    returns_col: str = "returns",
) -> Dict[str, float]:
    """Comprehensive signal performance analysis.

    Calculates all key performance metrics for trading signals including
    win rate, profit factor, quality score, and streak analysis.

    Args:
        df: DataFrame with signal and returns columns
        signal_col: Name of signal column (1 = long, -1 = short, 0 = neutral)
        returns_col: Name of returns column

    Returns:
        Dictionary with comprehensive performance metrics

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1, -1, 1, 0, 1],
        ...     "returns": [0.02, -0.01, 0.03, 0.0, -0.015]
        ... })
        >>> metrics = analyze_signal_performance(df)
    """
    win_metrics = calculate_win_metrics(df, signal_col, returns_col)
    profit_factor = calculate_profit_factor(df, signal_col, returns_col)
    quality_score = signal_quality_score(df, signal_col, returns_col)

    # Calculate consecutive streaks
    trades = df.filter(pl.col(signal_col) != 0)

    if trades.height > 0:
        trades = trades.with_columns(
            (pl.col(signal_col) * pl.col(returns_col)).alias("pnl")
        )

        # Count consecutive wins/losses
        trades = trades.with_columns([
            (pl.col("pnl") > 0).cast(pl.Int32).alias("is_win"),
        ])

        # Calculate streak changes
        trades = trades.with_columns(
            (pl.col("is_win") != pl.col("is_win").shift(1)).cum_sum().alias("streak_id")
        )

        # Get max consecutive wins and losses
        streak_stats = trades.group_by(["is_win", "streak_id"]).agg([
            pl.len().alias("streak_length")
        ])

        max_win_streak = int(
            streak_stats.filter(pl.col("is_win") == 1)["streak_length"].max() or 0
        )
        max_loss_streak = int(
            streak_stats.filter(pl.col("is_win") == 0)["streak_length"].max() or 0
        )
    else:
        max_win_streak = 0
        max_loss_streak = 0

    # Calculate total PnL
    total_pnl = float(
        df.filter(pl.col(signal_col) != 0)
        .with_columns((pl.col(signal_col) * pl.col(returns_col)).alias("pnl"))
        ["pnl"].sum()
    )

    return {
        **win_metrics,
        "profit_factor": profit_factor,
        "quality_score": quality_score,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "total_pnl": total_pnl,
        "avg_trade_pnl": total_pnl / win_metrics["total_trades"] if win_metrics["total_trades"] > 0 else 0.0,
    }
