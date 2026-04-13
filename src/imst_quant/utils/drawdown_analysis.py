"""Drawdown analysis utilities for portfolio performance evaluation.

This module provides detailed drawdown analysis including identification
of drawdown periods, recovery analysis, underwater curves, and
drawdown statistics.

Functions:
    calculate_drawdown_series: Compute drawdown time series
    identify_drawdown_periods: Find discrete drawdown events
    analyze_underwater: Calculate time spent below peak
    drawdown_statistics: Compute comprehensive drawdown metrics
    worst_drawdowns: Get top N worst drawdown periods

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.drawdown_analysis import (
    ...     identify_drawdown_periods,
    ...     drawdown_statistics
    ... )
    >>> returns = pl.Series([0.01, -0.05, -0.03, 0.02, 0.04, -0.02])
    >>> stats = drawdown_statistics(returns)
    >>> print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl


@dataclass
class DrawdownPeriod:
    """Represents a single drawdown event.

    Attributes:
        start_idx: Index where drawdown began
        trough_idx: Index of maximum drawdown point
        end_idx: Index where recovery completed (None if ongoing)
        start_date: Date when drawdown began (if available)
        trough_date: Date of maximum drawdown (if available)
        end_date: Date of recovery (if available)
        peak_value: Value at the start of drawdown
        trough_value: Value at the lowest point
        max_drawdown: Maximum drawdown as decimal (positive)
        duration_to_trough: Periods from start to trough
        recovery_duration: Periods from trough to recovery
        total_duration: Total periods from start to end
        is_recovered: Whether drawdown has fully recovered
    """

    start_idx: int
    trough_idx: int
    end_idx: Optional[int]
    start_date: Optional[datetime] = None
    trough_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    peak_value: float = 0.0
    trough_value: float = 0.0
    max_drawdown: float = 0.0
    duration_to_trough: int = 0
    recovery_duration: Optional[int] = None
    total_duration: Optional[int] = None
    is_recovered: bool = False


def calculate_drawdown_series(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> pl.DataFrame:
    """Calculate drawdown time series from returns.

    Computes cumulative returns, running maximum, and drawdown at each point.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        DataFrame with columns:
        - cumulative_return: Cumulative wealth (starts at 1.0)
        - running_max: Running peak value
        - drawdown: Current drawdown as decimal (positive)
        - drawdown_pct: Current drawdown as percentage

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, -0.05])
        >>> df = calculate_drawdown_series(returns)
        >>> df["drawdown"].max()  # Max drawdown
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Build cumulative return series (wealth curve starting at 1.0)
    cumulative = (1 + ret_series).cum_prod()

    # Running maximum
    running_max = cumulative.cum_max()

    # Drawdown at each point
    drawdown = (running_max - cumulative) / running_max

    return pl.DataFrame({
        "cumulative_return": cumulative,
        "running_max": running_max,
        "drawdown": drawdown,
        "drawdown_pct": drawdown * 100,
    })


def identify_drawdown_periods(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
    min_drawdown: float = 0.01,
    date_col: Optional[str] = None,
) -> List[DrawdownPeriod]:
    """Identify discrete drawdown periods from returns.

    Finds all drawdown events that exceed the minimum threshold,
    including their start, trough, and recovery points.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.
        min_drawdown: Minimum drawdown to consider as an event.
            Defaults to 0.01 (1%).
        date_col: Optional column name for dates.

    Returns:
        List of DrawdownPeriod objects describing each event.

    Example:
        >>> returns = pl.Series([0.05, -0.10, -0.05, 0.08, 0.05, -0.03])
        >>> periods = identify_drawdown_periods(returns, min_drawdown=0.05)
        >>> len(periods)
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
        dates = returns[date_col].to_list() if date_col and date_col in returns.columns else None
    else:
        ret_series = returns
        dates = None

    # Get drawdown series
    dd_df = calculate_drawdown_series(ret_series)
    cumulative = dd_df["cumulative_return"].to_list()
    running_max = dd_df["running_max"].to_list()
    drawdown = dd_df["drawdown"].to_list()

    periods: List[DrawdownPeriod] = []
    n = len(drawdown)

    i = 0
    while i < n:
        # Find start of drawdown (where drawdown becomes > 0)
        if drawdown[i] > 0:
            start_idx = i

            # Find trough (maximum drawdown point in this period)
            trough_idx = start_idx
            max_dd = drawdown[start_idx]

            j = start_idx + 1
            while j < n and drawdown[j] > 0:
                if drawdown[j] > max_dd:
                    max_dd = drawdown[j]
                    trough_idx = j
                j += 1

            # Check if recovered
            if j < n and drawdown[j] == 0:
                end_idx = j
                is_recovered = True
            else:
                end_idx = None
                is_recovered = False

            # Only include if exceeds minimum threshold
            if max_dd >= min_drawdown:
                period = DrawdownPeriod(
                    start_idx=start_idx,
                    trough_idx=trough_idx,
                    end_idx=end_idx,
                    start_date=dates[start_idx] if dates else None,
                    trough_date=dates[trough_idx] if dates else None,
                    end_date=dates[end_idx] if dates and end_idx else None,
                    peak_value=running_max[start_idx],
                    trough_value=cumulative[trough_idx],
                    max_drawdown=max_dd,
                    duration_to_trough=trough_idx - start_idx,
                    recovery_duration=(end_idx - trough_idx) if end_idx else None,
                    total_duration=(end_idx - start_idx) if end_idx else None,
                    is_recovered=is_recovered,
                )
                periods.append(period)

            i = j if j < n else n
        else:
            i += 1

    return periods


def analyze_underwater(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> Dict[str, float]:
    """Analyze time spent underwater (below previous peak).

    Calculates statistics about how long the portfolio spends
    in drawdown states.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with:
        - underwater_ratio: Fraction of time in drawdown
        - avg_underwater_depth: Average drawdown when underwater
        - longest_underwater: Maximum consecutive periods underwater
        - underwater_periods: Total number of periods underwater

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, 0.05, -0.02, 0.03])
        >>> stats = analyze_underwater(returns)
        >>> print(f"Underwater {stats['underwater_ratio']:.1%} of the time")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    dd_df = calculate_drawdown_series(ret_series)
    drawdown = dd_df["drawdown"].to_list()

    n = len(drawdown)
    underwater_periods = sum(1 for d in drawdown if d > 0)

    # Average depth when underwater
    underwater_values = [d for d in drawdown if d > 0]
    avg_depth = np.mean(underwater_values) if underwater_values else 0.0

    # Find longest consecutive underwater period
    longest = 0
    current_streak = 0
    for d in drawdown:
        if d > 0:
            current_streak += 1
            longest = max(longest, current_streak)
        else:
            current_streak = 0

    return {
        "underwater_ratio": underwater_periods / n if n > 0 else 0.0,
        "avg_underwater_depth": float(avg_depth),
        "longest_underwater": longest,
        "underwater_periods": underwater_periods,
    }


def drawdown_statistics(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> Dict[str, float]:
    """Calculate comprehensive drawdown statistics.

    Provides a complete picture of drawdown characteristics including
    maximum, average, and distribution metrics.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with:
        - max_drawdown: Maximum peak-to-trough decline
        - avg_drawdown: Average drawdown (when in drawdown)
        - drawdown_volatility: Std dev of drawdown values
        - calmar_ratio: Annualized return / max drawdown
        - ulcer_index: Square root of mean squared drawdown
        - pain_index: Mean of all drawdown values
        - recovery_factor: Total return / max drawdown
        - num_drawdowns: Count of drawdown events > 1%

    Example:
        >>> returns = pl.Series([0.01, -0.05, 0.02, -0.03, 0.04, 0.01])
        >>> stats = drawdown_statistics(returns)
        >>> print(f"Max DD: {stats['max_drawdown']:.2%}")
        >>> print(f"Ulcer Index: {stats['ulcer_index']:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    dd_df = calculate_drawdown_series(ret_series)
    drawdown = dd_df["drawdown"]
    cumulative = dd_df["cumulative_return"]

    # Basic metrics
    max_dd = float(drawdown.max()) if drawdown.max() is not None else 0.0

    # Underwater metrics
    underwater = drawdown.filter(drawdown > 0)
    avg_dd = float(underwater.mean()) if underwater.len() > 0 else 0.0
    dd_vol = float(underwater.std()) if underwater.len() > 1 else 0.0

    # Ulcer Index: sqrt(mean of squared drawdowns)
    squared_dd = (drawdown ** 2).mean()
    ulcer_index = float(np.sqrt(squared_dd)) if squared_dd is not None else 0.0

    # Pain Index: mean of all drawdowns (including zeros)
    pain_index = float(drawdown.mean()) if drawdown.mean() is not None else 0.0

    # Total return
    total_return = float(cumulative.to_list()[-1] - 1) if cumulative.len() > 0 else 0.0

    # Annualized return (assuming daily)
    n_periods = cumulative.len()
    if n_periods > 0:
        annualized_return = ((1 + total_return) ** (252 / n_periods)) - 1
    else:
        annualized_return = 0.0

    # Calmar ratio
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # Recovery factor
    recovery_factor = total_return / max_dd if max_dd > 0 else 0.0

    # Count significant drawdowns
    periods = identify_drawdown_periods(ret_series, min_drawdown=0.01)

    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "drawdown_volatility": dd_vol,
        "calmar_ratio": calmar,
        "ulcer_index": ulcer_index,
        "pain_index": pain_index,
        "recovery_factor": recovery_factor,
        "num_drawdowns": len(periods),
    }


def worst_drawdowns(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
    n: int = 5,
    date_col: Optional[str] = None,
) -> List[DrawdownPeriod]:
    """Get the N worst drawdown periods by magnitude.

    Identifies and ranks the most severe drawdown events.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.
        n: Number of worst drawdowns to return. Defaults to 5.
        date_col: Optional column name for dates.

    Returns:
        List of DrawdownPeriod objects, sorted by severity (worst first).

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.08, -0.15, 0.12, -0.08])
        >>> worst = worst_drawdowns(returns, n=3)
        >>> for i, dd in enumerate(worst):
        ...     print(f"{i+1}. {dd.max_drawdown:.2%}")
    """
    all_periods = identify_drawdown_periods(
        returns,
        return_col=return_col,
        min_drawdown=0.0,  # Get all drawdowns
        date_col=date_col,
    )

    # Sort by max_drawdown descending
    sorted_periods = sorted(all_periods, key=lambda p: p.max_drawdown, reverse=True)

    return sorted_periods[:n]


def drawdown_duration_analysis(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> Dict[str, float]:
    """Analyze drawdown durations and recovery times.

    Provides statistics about how long drawdowns last and how
    quickly they recover.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with:
        - avg_drawdown_duration: Average duration of drawdown events
        - avg_recovery_time: Average time from trough to recovery
        - max_drawdown_duration: Longest single drawdown
        - max_recovery_time: Longest recovery from a drawdown
        - pct_recovered: Percentage of drawdowns that recovered

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, 0.05, 0.02, -0.08, 0.03])
        >>> duration_stats = drawdown_duration_analysis(returns)
    """
    periods = identify_drawdown_periods(returns, return_col=return_col, min_drawdown=0.01)

    if not periods:
        return {
            "avg_drawdown_duration": 0.0,
            "avg_recovery_time": 0.0,
            "max_drawdown_duration": 0,
            "max_recovery_time": 0,
            "pct_recovered": 0.0,
        }

    # Durations to trough
    durations_to_trough = [p.duration_to_trough for p in periods]

    # Recovery times (only for recovered drawdowns)
    recovery_times = [p.recovery_duration for p in periods if p.recovery_duration is not None]

    # Total durations
    total_durations = [p.total_duration for p in periods if p.total_duration is not None]

    recovered_count = sum(1 for p in periods if p.is_recovered)

    return {
        "avg_drawdown_duration": float(np.mean(total_durations)) if total_durations else 0.0,
        "avg_recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
        "max_drawdown_duration": max(total_durations) if total_durations else 0,
        "max_recovery_time": max(recovery_times) if recovery_times else 0,
        "pct_recovered": recovered_count / len(periods) if periods else 0.0,
    }
