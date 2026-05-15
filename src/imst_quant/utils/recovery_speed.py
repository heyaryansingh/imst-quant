"""Drawdown recovery speed analysis utilities.

This module provides utilities for analyzing how quickly a portfolio
recovers from drawdowns, including recovery rate metrics, velocity
analysis, and recovery pattern classification.

Functions:
    calculate_recovery_rate: Compute recovery speed for each drawdown
    analyze_recovery_velocity: Measure acceleration of recovery
    classify_recovery_pattern: Categorize recovery shapes (V, U, L)
    recovery_efficiency_score: Overall recovery performance metric

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.recovery_speed import (
    ...     calculate_recovery_rate,
    ...     classify_recovery_pattern
    ... )
    >>> returns = pl.Series([0.01, -0.05, -0.03, 0.02, 0.04, -0.02])
    >>> rates = calculate_recovery_rate(returns)
    >>> pattern = classify_recovery_pattern(returns)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl

from imst_quant.utils.drawdown_analysis import (
    DrawdownPeriod,
    calculate_drawdown_series,
    identify_drawdown_periods,
)


class RecoveryPattern(str, Enum):
    """Classification of recovery shape patterns."""

    V_SHAPED = "V-shaped"  # Quick, linear recovery
    U_SHAPED = "U-shaped"  # Slow bottom, then recovery
    L_SHAPED = "L-shaped"  # Extended bottom, minimal recovery
    W_SHAPED = "W-shaped"  # Multiple recovery attempts
    INCOMPLETE = "Incomplete"  # Not yet recovered


@dataclass
class RecoveryMetrics:
    """Metrics describing a single drawdown recovery event.

    Attributes:
        period: The underlying drawdown period
        recovery_rate: Average return per period during recovery
        recovery_velocity: Change in recovery rate (acceleration)
        recovery_efficiency: Actual recovery / theoretical optimal recovery
        pattern: Shape classification of recovery
        time_to_50pct: Periods to recover 50% of drawdown
        time_to_90pct: Periods to recover 90% of drawdown
    """

    period: DrawdownPeriod
    recovery_rate: float
    recovery_velocity: float
    recovery_efficiency: float
    pattern: RecoveryPattern
    time_to_50pct: Optional[int] = None
    time_to_90pct: Optional[int] = None


def calculate_recovery_rate(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
    date_col: Optional[str] = None,
) -> List[RecoveryMetrics]:
    """Calculate recovery rate for each drawdown period.

    Analyzes how quickly the portfolio recovers from each drawdown,
    measuring the average return per period during recovery phase.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.
        date_col: Optional column name for dates.

    Returns:
        List of RecoveryMetrics for each drawdown that has recovered.

    Example:
        >>> returns = pl.Series([0.05, -0.10, -0.05, 0.08, 0.05])
        >>> metrics = calculate_recovery_rate(returns)
        >>> for m in metrics:
        ...     print(f"Recovery rate: {m.recovery_rate:.2%}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Identify all drawdown periods
    periods = identify_drawdown_periods(
        returns, return_col=return_col, min_drawdown=0.01, date_col=date_col
    )

    recovery_metrics = []

    for period in periods:
        if not period.is_recovered or period.recovery_duration is None:
            continue

        # Extract recovery phase returns
        trough_idx = period.trough_idx
        end_idx = period.end_idx
        if end_idx is None:
            continue

        recovery_returns = ret_series[trough_idx + 1 : end_idx + 1]

        # Calculate average recovery rate
        avg_recovery_rate = float(recovery_returns.mean())

        # Calculate recovery velocity (acceleration)
        if recovery_returns.len() > 1:
            first_half = recovery_returns[: len(recovery_returns) // 2].mean()
            second_half = recovery_returns[len(recovery_returns) // 2 :].mean()
            velocity = float(second_half - first_half)
        else:
            velocity = 0.0

        # Calculate recovery efficiency
        # Theoretical optimal: recover in 1 period with single return
        drawdown_amount = period.max_drawdown
        actual_periods = period.recovery_duration
        optimal_periods = 1
        efficiency = optimal_periods / actual_periods if actual_periods > 0 else 0.0

        # Classify pattern
        pattern = _classify_pattern(period, recovery_returns)

        # Calculate milestone times
        time_50, time_90 = _calculate_milestone_times(period, ret_series)

        metrics = RecoveryMetrics(
            period=period,
            recovery_rate=avg_recovery_rate,
            recovery_velocity=velocity,
            recovery_efficiency=efficiency,
            pattern=pattern,
            time_to_50pct=time_50,
            time_to_90pct=time_90,
        )

        recovery_metrics.append(metrics)

    return recovery_metrics


def _classify_pattern(period: DrawdownPeriod, recovery_returns: pl.Series) -> RecoveryPattern:
    """Classify the shape of recovery pattern.

    Args:
        period: Drawdown period to classify
        recovery_returns: Returns during recovery phase

    Returns:
        Recovery pattern classification
    """
    if not period.is_recovered:
        return RecoveryPattern.INCOMPLETE

    if period.recovery_duration is None or period.recovery_duration == 0:
        return RecoveryPattern.INCOMPLETE

    # V-shaped: Fast recovery, high average returns
    if period.recovery_duration <= 3 and recovery_returns.mean() > 0.02:
        return RecoveryPattern.V_SHAPED

    # Check for W-shaped (multiple dips during recovery)
    if recovery_returns.len() > 3:
        negative_count = (recovery_returns < 0).sum()
        if negative_count > recovery_returns.len() * 0.3:
            return RecoveryPattern.W_SHAPED

    # U-shaped: Moderate recovery speed
    if period.recovery_duration > period.duration_to_trough:
        return RecoveryPattern.U_SHAPED

    # L-shaped: Very slow recovery
    if period.recovery_duration > period.duration_to_trough * 2:
        return RecoveryPattern.L_SHAPED

    return RecoveryPattern.U_SHAPED


def _calculate_milestone_times(
    period: DrawdownPeriod, returns: pl.Series
) -> tuple[Optional[int], Optional[int]]:
    """Calculate time to recover 50% and 90% of drawdown.

    Args:
        period: Drawdown period
        returns: Full return series

    Returns:
        Tuple of (time_to_50pct, time_to_90pct) in periods
    """
    if not period.is_recovered or period.end_idx is None:
        return None, None

    # Calculate cumulative wealth from trough
    trough_idx = period.trough_idx
    end_idx = period.end_idx

    recovery_segment = returns[trough_idx : end_idx + 1]
    cumulative_from_trough = (1 + recovery_segment).cum_prod()

    # Target values
    trough_value = period.trough_value
    peak_value = period.peak_value
    target_50pct = trough_value + (peak_value - trough_value) * 0.5
    target_90pct = trough_value + (peak_value - trough_value) * 0.9

    # Find first time each target is reached
    time_50 = None
    time_90 = None

    cumulative_list = cumulative_from_trough.to_list()
    for i, val in enumerate(cumulative_list):
        current_value = trough_value * val

        if time_50 is None and current_value >= target_50pct:
            time_50 = i

        if time_90 is None and current_value >= target_90pct:
            time_90 = i

    return time_50, time_90


def analyze_recovery_velocity(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> Dict[str, float]:
    """Analyze recovery velocity across all drawdowns.

    Measures how quickly recoveries accelerate or decelerate.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with:
        - avg_recovery_rate: Average recovery return per period
        - avg_recovery_velocity: Average acceleration during recovery
        - positive_velocity_pct: % of recoveries that accelerate
        - fastest_recovery_rate: Highest recovery rate observed
        - slowest_recovery_rate: Lowest recovery rate observed

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, 0.05, -0.08, 0.06])
        >>> velocity_stats = analyze_recovery_velocity(returns)
    """
    metrics = calculate_recovery_rate(returns, return_col=return_col)

    if not metrics:
        return {
            "avg_recovery_rate": 0.0,
            "avg_recovery_velocity": 0.0,
            "positive_velocity_pct": 0.0,
            "fastest_recovery_rate": 0.0,
            "slowest_recovery_rate": 0.0,
        }

    rates = [m.recovery_rate for m in metrics]
    velocities = [m.recovery_velocity for m in metrics]

    positive_velocity_count = sum(1 for v in velocities if v > 0)

    return {
        "avg_recovery_rate": float(np.mean(rates)),
        "avg_recovery_velocity": float(np.mean(velocities)),
        "positive_velocity_pct": positive_velocity_count / len(velocities),
        "fastest_recovery_rate": float(np.max(rates)),
        "slowest_recovery_rate": float(np.min(rates)),
    }


def classify_recovery_pattern(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> Dict[str, int]:
    """Classify recovery patterns across all drawdowns.

    Categorizes each recovery by its shape (V, U, L, W).

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with counts for each pattern type:
        - V-shaped: Fast, linear recoveries
        - U-shaped: Moderate recoveries
        - L-shaped: Slow/stagnant recoveries
        - W-shaped: Multiple recovery attempts
        - Incomplete: Not yet recovered

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.08, -0.15, 0.02, 0.05])
        >>> patterns = classify_recovery_pattern(returns)
        >>> print(f"V-shaped: {patterns['V-shaped']}")
    """
    metrics = calculate_recovery_rate(returns, return_col=return_col)

    pattern_counts = {
        RecoveryPattern.V_SHAPED: 0,
        RecoveryPattern.U_SHAPED: 0,
        RecoveryPattern.L_SHAPED: 0,
        RecoveryPattern.W_SHAPED: 0,
        RecoveryPattern.INCOMPLETE: 0,
    }

    for m in metrics:
        pattern_counts[m.pattern] += 1

    return {k.value: v for k, v in pattern_counts.items()}


def recovery_efficiency_score(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> float:
    """Calculate overall recovery efficiency score (0-1 scale).

    Combines recovery rate, velocity, and pattern quality into a
    single score. Higher scores indicate better recovery characteristics.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Efficiency score from 0 (worst) to 1 (best).

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.08, 0.05, -0.05, 0.06])
        >>> score = recovery_efficiency_score(returns)
        >>> print(f"Recovery efficiency: {score:.2%}")
    """
    metrics = calculate_recovery_rate(returns, return_col=return_col)

    if not metrics:
        return 0.0

    # Component scores
    efficiencies = [m.recovery_efficiency for m in metrics]
    avg_efficiency = np.mean(efficiencies)

    # Pattern quality score
    pattern_counts = classify_recovery_pattern(returns, return_col=return_col)
    total_patterns = sum(pattern_counts.values())

    if total_patterns == 0:
        pattern_score = 0.0
    else:
        # V-shaped = 1.0, U-shaped = 0.7, L-shaped = 0.3, W-shaped = 0.5, Incomplete = 0
        pattern_weights = {
            RecoveryPattern.V_SHAPED.value: 1.0,
            RecoveryPattern.U_SHAPED.value: 0.7,
            RecoveryPattern.L_SHAPED.value: 0.3,
            RecoveryPattern.W_SHAPED.value: 0.5,
            RecoveryPattern.INCOMPLETE.value: 0.0,
        }

        weighted_sum = sum(
            pattern_counts.get(p, 0) * w for p, w in pattern_weights.items()
        )
        pattern_score = weighted_sum / total_patterns

    # Velocity score (positive is good)
    velocities = [m.recovery_velocity for m in metrics]
    positive_velocity_pct = sum(1 for v in velocities if v > 0) / len(velocities)

    # Combine scores (weighted average)
    final_score = (
        0.4 * avg_efficiency +
        0.4 * pattern_score +
        0.2 * positive_velocity_pct
    )

    return min(1.0, max(0.0, final_score))
