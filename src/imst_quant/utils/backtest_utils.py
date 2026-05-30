"""Backtesting utility functions for strategy evaluation.

This module provides helper functions commonly used in backtesting workflows
including equity curve generation, drawdown analysis, and performance reporting.

Functions:
    generate_equity_curve: Create cumulative equity from returns
    calculate_underwater_curve: Generate drawdown curve
    find_drawdown_periods: Identify all drawdown periods
    calculate_recovery_time: Time to recover from drawdowns
    generate_monthly_returns: Create monthly returns table
    generate_trade_log: Format trades for analysis
    calculate_holding_periods: Analyze position holding times

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.backtest_utils import generate_equity_curve
    >>> returns = pl.Series([0.01, -0.02, 0.015, 0.02])
    >>> equity = generate_equity_curve(returns, initial_capital=10000)
    >>> print(equity)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import polars as pl


def generate_equity_curve(
    returns: pl.Series,
    initial_capital: float = 10000,
) -> pl.Series:
    """Generate cumulative equity curve from returns.

    Args:
        returns: Series of returns (as decimals).
        initial_capital: Starting capital (default: 10000).

    Returns:
        Series of cumulative equity values.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> equity = generate_equity_curve(returns, initial_capital=10000)
        >>> print(equity.to_list())
        [10100.0, 9898.0, 10046.47, 10247.4]
    """
    if len(returns) == 0:
        return pl.Series([initial_capital], dtype=pl.Float64)

    cumulative_returns = (1 + returns).cum_prod()
    equity = cumulative_returns * initial_capital

    return equity


def calculate_underwater_curve(equity_curve: pl.Series) -> pl.Series:
    """Calculate underwater (drawdown) curve.

    The underwater curve shows how far below the previous peak
    the equity is at each point in time.

    Args:
        equity_curve: Series of cumulative equity values.

    Returns:
        Series of drawdown percentages (negative values).

    Example:
        >>> equity = pl.Series([10000, 10100, 9900, 10200, 9800])
        >>> underwater = calculate_underwater_curve(equity)
        >>> print(underwater.to_list())
        [0.0, 0.0, -0.0198..., 0.0, -0.0392...]
    """
    if len(equity_curve) == 0:
        return pl.Series([], dtype=pl.Float64)

    running_max = equity_curve.cum_max()
    drawdown = (equity_curve - running_max) / running_max

    return drawdown


def find_drawdown_periods(
    equity_curve: pl.Series,
    min_drawdown: float = 0.05,
) -> List[Dict[str, any]]:
    """Identify distinct drawdown periods.

    A drawdown period starts when equity drops below previous peak
    and ends when equity recovers to a new peak.

    Args:
        equity_curve: Series of cumulative equity values.
        min_drawdown: Minimum drawdown to report (default: 5%).

    Returns:
        List of dictionaries with:
            - start_idx: Start index of drawdown
            - trough_idx: Index of maximum drawdown
            - end_idx: Recovery index (or None if not recovered)
            - depth: Maximum drawdown percentage
            - duration: Number of periods from start to recovery

    Example:
        >>> equity = pl.Series([10000, 10500, 9800, 9500, 10200, 11000])
        >>> periods = find_drawdown_periods(equity, min_drawdown=0.05)
        >>> print(f"Found {len(periods)} significant drawdowns")
    """
    if len(equity_curve) < 2:
        return []

    drawdown = calculate_underwater_curve(equity_curve)
    periods = []

    in_drawdown = False
    start_idx = None
    peak_value = equity_curve[0]
    max_dd = 0.0
    trough_idx = 0

    for i in range(len(equity_curve)):
        current_value = equity_curve[i]
        current_dd = drawdown[i]

        if not in_drawdown:
            if current_dd < -min_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_idx = i - 1  # Previous peak
                peak_value = equity_curve[start_idx]
                max_dd = current_dd
                trough_idx = i
        else:
            # Track maximum drawdown
            if current_dd < max_dd:
                max_dd = current_dd
                trough_idx = i

            # Check for recovery
            if current_value >= peak_value:
                # Drawdown ended
                periods.append({
                    "start_idx": start_idx,
                    "trough_idx": trough_idx,
                    "end_idx": i,
                    "depth": abs(max_dd),
                    "duration": i - start_idx,
                })
                in_drawdown = False
                peak_value = current_value

    # If still in drawdown at end
    if in_drawdown:
        periods.append({
            "start_idx": start_idx,
            "trough_idx": trough_idx,
            "end_idx": None,  # Not recovered
            "depth": abs(max_dd),
            "duration": len(equity_curve) - start_idx,
        })

    return periods


def calculate_recovery_time(equity_curve: pl.Series) -> Dict[str, float]:
    """Calculate drawdown recovery statistics.

    Args:
        equity_curve: Series of cumulative equity values.

    Returns:
        Dictionary with:
            - avg_recovery_time: Average periods to recover
            - max_recovery_time: Longest recovery period
            - current_drawdown_duration: Periods in current drawdown (if any)

    Example:
        >>> equity = pl.Series([10000, 10500, 9800, 9500, 10200, 11000])
        >>> recovery = calculate_recovery_time(equity)
        >>> print(f"Avg recovery: {recovery['avg_recovery_time']:.1f} periods")
    """
    periods = find_drawdown_periods(equity_curve, min_drawdown=0.01)

    if not periods:
        return {
            "avg_recovery_time": 0.0,
            "max_recovery_time": 0,
            "current_drawdown_duration": 0,
        }

    # Filter out current (unrecovered) drawdown
    recovered = [p for p in periods if p["end_idx"] is not None]
    current = [p for p in periods if p["end_idx"] is None]

    recovery_times = [p["duration"] for p in recovered]

    return {
        "avg_recovery_time": sum(recovery_times) / len(recovery_times) if recovery_times else 0.0,
        "max_recovery_time": max(recovery_times) if recovery_times else 0,
        "current_drawdown_duration": current[0]["duration"] if current else 0,
    }


def generate_monthly_returns(
    df: pl.DataFrame,
    date_col: str = "date",
    returns_col: str = "returns",
) -> pl.DataFrame:
    """Generate monthly returns table from daily returns.

    Args:
        df: DataFrame with date and returns columns.
        date_col: Name of date column.
        returns_col: Name of returns column.

    Returns:
        DataFrame with year, month, and monthly_return columns.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 2, 1)],
        ...     "returns": [0.01, 0.02, -0.01]
        ... })
        >>> monthly = generate_monthly_returns(df)
        >>> print(monthly)
    """
    return (
        df.with_columns([
            pl.col(date_col).dt.year().alias("year"),
            pl.col(date_col).dt.month().alias("month"),
        ])
        .group_by(["year", "month"])
        .agg([
            ((1 + pl.col(returns_col)).product() - 1).alias("monthly_return")
        ])
        .sort(["year", "month"])
    )


def calculate_holding_periods(
    trades: pl.DataFrame,
    entry_time_col: str = "entry_time",
    exit_time_col: str = "exit_time",
) -> Dict[str, float]:
    """Analyze position holding time statistics.

    Args:
        trades: DataFrame with entry and exit timestamps.
        entry_time_col: Name of entry timestamp column.
        exit_time_col: Name of exit timestamp column.

    Returns:
        Dictionary with:
            - avg_holding_hours: Average holding time in hours
            - median_holding_hours: Median holding time
            - min_holding_hours: Shortest hold
            - max_holding_hours: Longest hold

    Example:
        >>> trades = pl.DataFrame({
        ...     "entry_time": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 2, 10, 0)],
        ...     "exit_time": [datetime(2024, 1, 1, 16, 0), datetime(2024, 1, 3, 10, 0)]
        ... })
        >>> stats = calculate_holding_periods(trades)
        >>> print(f"Avg hold: {stats['avg_holding_hours']:.1f} hours")
    """
    if len(trades) == 0:
        return {
            "avg_holding_hours": 0.0,
            "median_holding_hours": 0.0,
            "min_holding_hours": 0.0,
            "max_holding_hours": 0.0,
        }

    holding_times = (
        pl.col(exit_time_col) - pl.col(entry_time_col)
    ).dt.total_seconds() / 3600  # Convert to hours

    df_with_holding = trades.with_columns(holding_times.alias("holding_hours"))

    return {
        "avg_holding_hours": float(df_with_holding["holding_hours"].mean()),
        "median_holding_hours": float(df_with_holding["holding_hours"].median()),
        "min_holding_hours": float(df_with_holding["holding_hours"].min()),
        "max_holding_hours": float(df_with_holding["holding_hours"].max()),
    }


def calculate_consecutive_outcomes(
    trades: pl.DataFrame,
    pnl_col: str = "pnl",
) -> Dict[str, int]:
    """Calculate consecutive win/loss streaks.

    Args:
        trades: DataFrame with PnL column.
        pnl_col: Name of PnL column.

    Returns:
        Dictionary with:
            - current_streak: Current win/loss streak (positive = wins)
            - max_win_streak: Longest winning streak
            - max_loss_streak: Longest losing streak

    Example:
        >>> trades = pl.DataFrame({"pnl": [100, 50, -30, -20, 150, 200]})
        >>> streaks = calculate_consecutive_outcomes(trades)
        >>> print(f"Current: {streaks['current_streak']} wins")
    """
    if len(trades) == 0:
        return {
            "current_streak": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    pnl = trades[pnl_col]

    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for value in pnl:
        if value > 0:
            current_win_streak += 1
            current_loss_streak = 0
            current_streak = current_win_streak
            max_win_streak = max(max_win_streak, current_win_streak)
        elif value < 0:
            current_loss_streak += 1
            current_win_streak = 0
            current_streak = -current_loss_streak
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0

    return {
        "current_streak": current_streak,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def calculate_mae_mfe(
    trades: pl.DataFrame,
    entry_col: str = "entry_price",
    exit_col: str = "exit_price",
    high_col: str = "high_during_trade",
    low_col: str = "low_during_trade",
) -> pl.DataFrame:
    """Calculate Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).

    MAE = worst price move against position during the trade
    MFE = best price move in favor of position during the trade

    These metrics help evaluate if trades are being exited too early/late.

    Args:
        trades: DataFrame with price data.
        entry_col: Entry price column name.
        exit_col: Exit price column name.
        high_col: Highest price during trade column name.
        low_col: Lowest price during trade column name.

    Returns:
        DataFrame with added mae and mfe columns (as % of entry).

    Example:
        >>> trades = pl.DataFrame({
        ...     "entry_price": [100, 105],
        ...     "exit_price": [110, 103],
        ...     "high_during_trade": [112, 108],
        ...     "low_during_trade": [98, 102]
        ... })
        >>> mae_mfe = calculate_mae_mfe(trades)
        >>> print(mae_mfe.select(["mae", "mfe"]))
    """
    return trades.with_columns([
        # MAE: worst move against position (negative %)
        (
            (pl.col(low_col) - pl.col(entry_col)) / pl.col(entry_col)
        ).alias("mae"),

        # MFE: best move in favor (positive %)
        (
            (pl.col(high_col) - pl.col(entry_col)) / pl.col(entry_col)
        ).alias("mfe"),
    ])


def generate_performance_summary(
    equity_curve: pl.Series,
    trades: Optional[pl.DataFrame] = None,
    initial_capital: float = 10000,
    periods_per_year: int = 252,
) -> Dict[str, any]:
    """Generate comprehensive backtest performance summary.

    Args:
        equity_curve: Series of cumulative equity values.
        trades: Optional DataFrame of trades for trade-level metrics.
        initial_capital: Starting capital.
        periods_per_year: Periods per year for annualization.

    Returns:
        Dictionary with comprehensive performance metrics.

    Example:
        >>> equity = pl.Series([10000, 10100, 10200, 10150, 10300])
        >>> summary = generate_performance_summary(equity)
        >>> print(f"Total Return: {summary['total_return']:.2%}")
    """
    if len(equity_curve) == 0:
        return {}

    # Calculate returns
    returns = equity_curve.pct_change().drop_nulls()

    # Basic metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    final_value = equity_curve[-1]

    # Drawdown analysis
    underwater = calculate_underwater_curve(equity_curve)
    max_dd = abs(float(underwater.min()))

    recovery = calculate_recovery_time(equity_curve)

    summary = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "avg_recovery_time": recovery["avg_recovery_time"],
        "max_recovery_time": recovery["max_recovery_time"],
        "total_periods": len(equity_curve),
    }

    # Add trade-level metrics if trades provided
    if trades is not None and len(trades) > 0:
        if "pnl" in trades.columns:
            pnl = trades["pnl"]
            winning = pnl.filter(pnl > 0)
            losing = pnl.filter(pnl < 0)

            summary.update({
                "total_trades": len(trades),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": len(winning) / len(trades) if len(trades) > 0 else 0.0,
                "avg_win": float(winning.mean()) if len(winning) > 0 else 0.0,
                "avg_loss": abs(float(losing.mean())) if len(losing) > 0 else 0.0,
            })

    return summary
