"""Drawdown recovery analysis for portfolio performance evaluation.

This module provides detailed analysis of drawdown recovery patterns,
including identification of recovery periods, timing analysis, recovery
rate calculations, and probabilistic recovery time estimation.

Features:
- Recovery period identification and analysis
- Recovery timing statistics (duration, rate)
- Historical recovery pattern analysis
- Monte Carlo based recovery time estimation
- Underwater curve analysis with duration tracking

Functions:
    analyze_recovery_periods: Identify and analyze all recovery events
    recovery_statistics: Compute aggregate recovery statistics
    estimate_recovery_time: Estimate expected recovery time from current drawdown
    underwater_analysis: Calculate underwater curve with duration tracking

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.drawdown_recovery import (
    ...     analyze_recovery_periods,
    ...     recovery_statistics,
    ...     estimate_recovery_time
    ... )
    >>> returns = pl.Series([0.01, -0.05, -0.03, 0.02, 0.04, 0.03, -0.02])
    >>> recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
    >>> stats = recovery_statistics(recoveries)
    >>> print(f"Avg recovery duration: {stats['avg_recovery_duration']:.1f} periods")
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl


@dataclass
class RecoveryAnalysis:
    """Container for a single drawdown recovery event.

    Represents a complete drawdown-recovery cycle from the start of a
    drawdown through the lowest point (trough) to full recovery.

    Attributes:
        drawdown_depth: Maximum depth of the drawdown as decimal (positive).
        drawdown_start: Index where drawdown began.
        drawdown_trough: Index of the lowest point (maximum drawdown).
        recovery_end: Index where recovery completed (None if still underwater).
        drawdown_duration: Number of periods from start to trough.
        recovery_duration: Number of periods from trough to recovery (None if ongoing).
        total_duration: Total periods from start to recovery (None if ongoing).
        recovery_rate: Annualized rate of recovery from trough (None if ongoing).

    Example:
        >>> recovery = RecoveryAnalysis(
        ...     drawdown_depth=0.15,
        ...     drawdown_start=10,
        ...     drawdown_trough=25,
        ...     recovery_end=50,
        ...     drawdown_duration=15,
        ...     recovery_duration=25,
        ...     total_duration=40,
        ...     recovery_rate=0.42
        ... )
        >>> print(f"Recovered {recovery.drawdown_depth:.1%} in {recovery.total_duration} periods")
    """

    drawdown_depth: float
    drawdown_start: int
    drawdown_trough: int
    recovery_end: Optional[int]
    drawdown_duration: int
    recovery_duration: Optional[int]
    total_duration: Optional[int]
    recovery_rate: Optional[float]


def _to_numpy(
    returns: Union[pl.Series, np.ndarray, List[float]],
) -> np.ndarray:
    """Convert returns input to numpy array.

    Args:
        returns: Returns as pl.Series, np.ndarray, or list.

    Returns:
        Numpy array of returns.
    """
    if isinstance(returns, pl.Series):
        return returns.drop_nulls().to_numpy()
    elif isinstance(returns, list):
        return np.array(returns)
    else:
        return returns


def analyze_recovery_periods(
    returns: Union[pl.Series, np.ndarray, List[float]],
    min_drawdown: float = 0.05,
    periods_per_year: int = 252,
) -> List[RecoveryAnalysis]:
    """Identify and analyze all drawdown recovery periods.

    Finds all drawdown events that exceed the minimum threshold and
    calculates recovery timing and rates for each. A drawdown period
    starts when cumulative returns decline from a peak and ends when
    they return to that peak level.

    Args:
        returns: Return series (daily returns recommended).
        min_drawdown: Minimum drawdown depth to consider as an event.
            Defaults to 0.05 (5%).
        periods_per_year: Number of trading periods per year for
            annualization. Defaults to 252 (daily).

    Returns:
        List of RecoveryAnalysis objects, one per qualifying drawdown event.

    Example:
        >>> returns = pl.Series([0.02, -0.10, -0.05, 0.03, 0.08, 0.05, -0.03])
        >>> recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        >>> for r in recoveries:
        ...     print(f"Drawdown: {r.drawdown_depth:.1%}, Duration: {r.total_duration}")
    """
    ret_np = _to_numpy(returns)
    n = len(ret_np)

    if n == 0:
        return []

    # Calculate cumulative wealth curve
    cumulative = np.cumprod(1 + ret_np)

    # Calculate running maximum (peak values)
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdown at each point
    drawdown = (running_max - cumulative) / running_max

    recoveries: List[RecoveryAnalysis] = []

    i = 0
    while i < n:
        # Find start of drawdown (where drawdown becomes > 0)
        if drawdown[i] > 0:
            start_idx = i

            # Track back to find true start (last peak)
            # The start is where drawdown first became positive
            while start_idx > 0 and drawdown[start_idx - 1] == 0:
                start_idx -= 1

            # Find trough (maximum drawdown point in this period)
            trough_idx = i
            max_dd = drawdown[i]

            j = i + 1
            while j < n and drawdown[j] > 0:
                if drawdown[j] > max_dd:
                    max_dd = drawdown[j]
                    trough_idx = j
                j += 1

            # Check if recovered (drawdown returns to 0)
            if j < n and drawdown[j] == 0:
                end_idx: Optional[int] = j
                recovery_duration = j - trough_idx
                total_duration = j - start_idx

                # Calculate annualized recovery rate
                # Recovery rate = how fast it climbed from trough to peak
                if recovery_duration > 0:
                    trough_value = cumulative[trough_idx]
                    recovery_value = cumulative[j]
                    recovery_return = (recovery_value / trough_value) - 1

                    # Annualize the recovery rate
                    recovery_rate = (
                        (1 + recovery_return) ** (periods_per_year / recovery_duration)
                    ) - 1
                else:
                    recovery_rate = None
            else:
                end_idx = None
                recovery_duration = None
                total_duration = None
                recovery_rate = None

            # Only include if exceeds minimum threshold
            if max_dd >= min_drawdown:
                recovery = RecoveryAnalysis(
                    drawdown_depth=float(max_dd),
                    drawdown_start=start_idx,
                    drawdown_trough=trough_idx,
                    recovery_end=end_idx,
                    drawdown_duration=trough_idx - start_idx,
                    recovery_duration=recovery_duration,
                    total_duration=total_duration,
                    recovery_rate=recovery_rate,
                )
                recoveries.append(recovery)

            i = j if j < n else n
        else:
            i += 1

    return recoveries


def recovery_statistics(
    recoveries: List[RecoveryAnalysis],
) -> Dict[str, float]:
    """Calculate aggregate statistics from a list of recovery events.

    Computes summary statistics across all recovery events including
    average durations, depths, and recovery rates.

    Args:
        recoveries: List of RecoveryAnalysis objects from analyze_recovery_periods.

    Returns:
        Dictionary with:
        - avg_recovery_duration: Mean periods from trough to recovery.
        - median_recovery_duration: Median periods from trough to recovery.
        - avg_drawdown_depth: Mean maximum drawdown across events.
        - max_drawdown_depth: Worst drawdown observed.
        - pct_recovered: Percentage of drawdowns that fully recovered.
        - avg_recovery_rate: Mean annualized recovery rate.
        - longest_recovery: Maximum periods to recover from a drawdown.

    Example:
        >>> recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        >>> stats = recovery_statistics(recoveries)
        >>> print(f"Recovery rate: {stats['pct_recovered']:.1%}")
        >>> print(f"Avg recovery time: {stats['avg_recovery_duration']:.0f} days")
    """
    if not recoveries:
        return {
            "avg_recovery_duration": 0.0,
            "median_recovery_duration": 0.0,
            "avg_drawdown_depth": 0.0,
            "max_drawdown_depth": 0.0,
            "pct_recovered": 0.0,
            "avg_recovery_rate": 0.0,
            "longest_recovery": 0.0,
        }

    # Extract recovery durations (only for recovered events)
    recovery_durations = [
        r.recovery_duration for r in recoveries if r.recovery_duration is not None
    ]

    # Extract drawdown depths
    drawdown_depths = [r.drawdown_depth for r in recoveries]

    # Extract recovery rates (only for recovered events)
    recovery_rates = [
        r.recovery_rate for r in recoveries if r.recovery_rate is not None
    ]

    # Count recovered vs total
    recovered_count = sum(1 for r in recoveries if r.recovery_end is not None)

    return {
        "avg_recovery_duration": (
            float(np.mean(recovery_durations)) if recovery_durations else 0.0
        ),
        "median_recovery_duration": (
            float(np.median(recovery_durations)) if recovery_durations else 0.0
        ),
        "avg_drawdown_depth": float(np.mean(drawdown_depths)),
        "max_drawdown_depth": float(np.max(drawdown_depths)),
        "pct_recovered": recovered_count / len(recoveries) if recoveries else 0.0,
        "avg_recovery_rate": (
            float(np.mean(recovery_rates)) if recovery_rates else 0.0
        ),
        "longest_recovery": (
            float(max(recovery_durations)) if recovery_durations else 0.0
        ),
    }


def estimate_recovery_time(
    returns: Union[pl.Series, np.ndarray, List[float]],
    current_drawdown: float,
    method: Literal["historical", "monte_carlo"] = "historical",
    n_simulations: int = 10000,
    periods_per_year: int = 252,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate expected recovery time from current drawdown depth.

    Uses historical recovery patterns or Monte Carlo simulation to
    estimate how long it might take to recover from the current
    drawdown level.

    Args:
        returns: Historical return series for estimation.
        current_drawdown: Current drawdown depth as decimal (e.g., 0.15 = 15%).
        method: Estimation method, either "historical" or "monte_carlo".
            Default "historical".
        n_simulations: Number of Monte Carlo simulations if method is
            "monte_carlo". Default 10000.
        periods_per_year: Trading periods per year. Default 252.
        seed: Random seed for reproducibility (Monte Carlo only).

    Returns:
        Dictionary with:
        - expected_days: Expected number of periods to recover.
        - confidence_interval_90: Tuple with (lower, upper) 90% CI bounds.
        - prob_recover_30d: Probability of recovery within 30 periods.
        - prob_recover_90d: Probability of recovery within 90 periods.
        - prob_recover_252d: Probability of recovery within 252 periods.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005] * 100)
        >>> estimate = estimate_recovery_time(returns, current_drawdown=0.10)
        >>> print(f"Expected recovery: {estimate['expected_days']:.0f} days")
        >>> print(f"P(recover in 90d): {estimate['prob_recover_90d']:.1%}")
    """
    ret_np = _to_numpy(returns)
    rng = np.random.default_rng(seed)

    if method == "historical":
        return _estimate_recovery_historical(
            ret_np, current_drawdown, periods_per_year
        )
    else:
        return _estimate_recovery_monte_carlo(
            ret_np, current_drawdown, n_simulations, periods_per_year, rng
        )


def _estimate_recovery_historical(
    returns: np.ndarray,
    current_drawdown: float,
    periods_per_year: int,
) -> Dict[str, float]:
    """Estimate recovery time using historical recovery patterns.

    Finds historical drawdowns of similar depth and uses their recovery
    times to estimate expected recovery from the current drawdown.
    """
    # Analyze historical recovery periods
    recoveries = analyze_recovery_periods(
        returns,
        min_drawdown=current_drawdown * 0.5,  # Look at similar or deeper drawdowns
        periods_per_year=periods_per_year,
    )

    # Filter to recoveries with similar depth (within 50% of current)
    similar_recoveries = [
        r
        for r in recoveries
        if r.recovery_duration is not None
        and r.drawdown_depth >= current_drawdown * 0.5
    ]

    if not similar_recoveries:
        # Fallback: use mean return to estimate recovery
        mean_return = float(np.mean(returns))
        if mean_return <= 0:
            return {
                "expected_days": float("inf"),
                "confidence_interval_90": (float("inf"), float("inf")),
                "prob_recover_30d": 0.0,
                "prob_recover_90d": 0.0,
                "prob_recover_252d": 0.0,
            }

        # Approximate recovery time: ln(1/(1-dd)) / ln(1+r)
        expected_days = np.log(1 / (1 - current_drawdown)) / np.log(1 + mean_return)

        return {
            "expected_days": float(expected_days),
            "confidence_interval_90": (expected_days * 0.5, expected_days * 2.0),
            "prob_recover_30d": 1.0 if expected_days <= 30 else 0.3,
            "prob_recover_90d": 1.0 if expected_days <= 90 else 0.5,
            "prob_recover_252d": 1.0 if expected_days <= 252 else 0.7,
        }

    # Weight by similarity of drawdown depth
    weights = []
    recovery_times = []
    for r in similar_recoveries:
        # Closer depth = higher weight
        depth_diff = abs(r.drawdown_depth - current_drawdown)
        weight = 1.0 / (1.0 + depth_diff * 10)
        weights.append(weight)
        recovery_times.append(r.recovery_duration)

    weights = np.array(weights)
    weights = weights / weights.sum()
    recovery_times = np.array(recovery_times)

    # Weighted average
    expected_days = float(np.average(recovery_times, weights=weights))

    # Confidence interval from distribution
    sorted_times = np.sort(recovery_times)
    ci_lower = float(np.percentile(sorted_times, 5))
    ci_upper = float(np.percentile(sorted_times, 95))

    # Recovery probabilities
    n_total = len(recovery_times)
    prob_30d = float(np.sum(recovery_times <= 30) / n_total)
    prob_90d = float(np.sum(recovery_times <= 90) / n_total)
    prob_252d = float(np.sum(recovery_times <= 252) / n_total)

    return {
        "expected_days": expected_days,
        "confidence_interval_90": (ci_lower, ci_upper),
        "prob_recover_30d": prob_30d,
        "prob_recover_90d": prob_90d,
        "prob_recover_252d": prob_252d,
    }


def _estimate_recovery_monte_carlo(
    returns: np.ndarray,
    current_drawdown: float,
    n_simulations: int,
    periods_per_year: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Estimate recovery time using Monte Carlo simulation.

    Simulates future return paths by sampling from historical returns
    and tracks how long it takes to recover from the current drawdown.
    """
    max_horizon = periods_per_year * 5  # Max 5 years to recover
    recovery_times = []

    # Current wealth is (1 - drawdown) relative to peak
    current_value = 1.0 - current_drawdown
    target_value = 1.0  # Need to get back to peak

    for _ in range(n_simulations):
        value = current_value
        for t in range(1, max_horizon + 1):
            # Sample a random return from history
            sampled_return = rng.choice(returns)
            value = value * (1 + sampled_return)

            if value >= target_value:
                recovery_times.append(t)
                break
        else:
            # Did not recover within max horizon
            recovery_times.append(max_horizon)

    recovery_times = np.array(recovery_times)

    # Statistics
    expected_days = float(np.mean(recovery_times))
    ci_lower = float(np.percentile(recovery_times, 5))
    ci_upper = float(np.percentile(recovery_times, 95))

    # Recovery probabilities
    prob_30d = float(np.mean(recovery_times <= 30))
    prob_90d = float(np.mean(recovery_times <= 90))
    prob_252d = float(np.mean(recovery_times <= 252))

    return {
        "expected_days": expected_days,
        "confidence_interval_90": (ci_lower, ci_upper),
        "prob_recover_30d": prob_30d,
        "prob_recover_90d": prob_90d,
        "prob_recover_252d": prob_252d,
    }


def underwater_analysis(
    returns: Union[pl.Series, np.ndarray, List[float]],
) -> pl.DataFrame:
    """Calculate underwater curve with duration tracking.

    Computes the cumulative return, peak, drawdown, and consecutive
    underwater duration at each period. The underwater curve shows
    how deep below the peak the portfolio is at each point in time.

    Args:
        returns: Return series.

    Returns:
        DataFrame with columns:
        - period: Index (0-based).
        - cumulative_return: Cumulative wealth (starts at 1.0).
        - peak: Running maximum cumulative return.
        - drawdown: Current drawdown as decimal (positive).
        - is_underwater: Boolean indicating if currently in drawdown.
        - underwater_duration: Consecutive periods underwater.

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, 0.05, -0.02, 0.03])
        >>> df = underwater_analysis(returns)
        >>> print(df)
        shape: (6, 6)
        +--------+-------------------+------+----------+--------------+---------------------+
        | period | cumulative_return | peak | drawdown | is_underwater| underwater_duration |
        +--------+-------------------+------+----------+--------------+---------------------+
        | 0      | 1.05              | 1.05 | 0.0      | false        | 0                   |
        | 1      | 0.945             | 1.05 | 0.1      | true         | 1                   |
        | ...    |                   |      |          |              |                     |
        +--------+-------------------+------+----------+--------------+---------------------+
    """
    ret_np = _to_numpy(returns)
    n = len(ret_np)

    if n == 0:
        return pl.DataFrame({
            "period": [],
            "cumulative_return": [],
            "peak": [],
            "drawdown": [],
            "is_underwater": [],
            "underwater_duration": [],
        })

    # Calculate cumulative wealth
    cumulative = np.cumprod(1 + ret_np)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdown at each point
    drawdown = (running_max - cumulative) / running_max

    # Determine underwater status
    is_underwater = drawdown > 0

    # Calculate consecutive underwater duration
    underwater_duration = np.zeros(n, dtype=int)
    current_streak = 0
    for i in range(n):
        if is_underwater[i]:
            current_streak += 1
            underwater_duration[i] = current_streak
        else:
            current_streak = 0
            underwater_duration[i] = 0

    return pl.DataFrame({
        "period": list(range(n)),
        "cumulative_return": cumulative.tolist(),
        "peak": running_max.tolist(),
        "drawdown": drawdown.tolist(),
        "is_underwater": is_underwater.tolist(),
        "underwater_duration": underwater_duration.tolist(),
    })


def recovery_by_depth_bucket(
    recoveries: List[RecoveryAnalysis],
    buckets: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Analyze recovery statistics by drawdown depth buckets.

    Groups recovery events by their drawdown depth and computes
    statistics for each bucket. Useful for understanding how
    recovery characteristics vary with drawdown severity.

    Args:
        recoveries: List of RecoveryAnalysis objects.
        buckets: List of bucket boundaries (e.g., [0.05, 0.10, 0.20, 0.50]).
            Defaults to [0.05, 0.10, 0.15, 0.20, 0.30].

    Returns:
        Dictionary mapping bucket labels to statistics:
        - count: Number of events in bucket.
        - avg_recovery_duration: Mean recovery duration.
        - avg_recovery_rate: Mean annualized recovery rate.
        - pct_recovered: Percentage that fully recovered.

    Example:
        >>> recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        >>> by_depth = recovery_by_depth_bucket(recoveries)
        >>> for bucket, stats in by_depth.items():
        ...     print(f"{bucket}: {stats['avg_recovery_duration']:.0f} days")
    """
    if buckets is None:
        buckets = [0.05, 0.10, 0.15, 0.20, 0.30]

    # Create bucket labels
    bucket_labels = []
    for i, b in enumerate(buckets):
        if i == 0:
            bucket_labels.append(f"<{b:.0%}")
        bucket_labels.append(f"{b:.0%}-{buckets[i+1]:.0%}" if i + 1 < len(buckets) else f">{b:.0%}")

    # Initialize results
    results: Dict[str, Dict[str, float]] = {}

    # Process each bucket
    for i in range(len(buckets) + 1):
        if i == 0:
            lower = 0.0
            upper = buckets[0]
            label = f"<{buckets[0]:.0%}"
        elif i == len(buckets):
            lower = buckets[-1]
            upper = float("inf")
            label = f">{buckets[-1]:.0%}"
        else:
            lower = buckets[i - 1]
            upper = buckets[i]
            label = f"{lower:.0%}-{upper:.0%}"

        # Filter recoveries in this bucket
        bucket_recoveries = [
            r for r in recoveries if lower <= r.drawdown_depth < upper
        ]

        if not bucket_recoveries:
            results[label] = {
                "count": 0,
                "avg_recovery_duration": 0.0,
                "avg_recovery_rate": 0.0,
                "pct_recovered": 0.0,
            }
            continue

        # Calculate statistics
        recovery_durations = [
            r.recovery_duration
            for r in bucket_recoveries
            if r.recovery_duration is not None
        ]
        recovery_rates = [
            r.recovery_rate
            for r in bucket_recoveries
            if r.recovery_rate is not None
        ]
        recovered_count = sum(
            1 for r in bucket_recoveries if r.recovery_end is not None
        )

        results[label] = {
            "count": float(len(bucket_recoveries)),
            "avg_recovery_duration": (
                float(np.mean(recovery_durations)) if recovery_durations else 0.0
            ),
            "avg_recovery_rate": (
                float(np.mean(recovery_rates)) if recovery_rates else 0.0
            ),
            "pct_recovered": recovered_count / len(bucket_recoveries),
        }

    return results


def recovery_velocity(
    recoveries: List[RecoveryAnalysis],
) -> Dict[str, float]:
    """Calculate recovery velocity metrics.

    Analyzes how quickly portfolios recover from drawdowns by
    computing velocity (depth / duration) and acceleration metrics.

    Args:
        recoveries: List of RecoveryAnalysis objects.

    Returns:
        Dictionary with:
        - avg_velocity: Average recovery velocity (depth per period).
        - max_velocity: Fastest observed recovery velocity.
        - min_velocity: Slowest observed recovery velocity.
        - velocity_std: Standard deviation of velocities.

    Example:
        >>> recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        >>> velocity = recovery_velocity(recoveries)
        >>> print(f"Avg recovery speed: {velocity['avg_velocity']:.4f} per day")
    """
    velocities = []

    for r in recoveries:
        if r.recovery_duration is not None and r.recovery_duration > 0:
            # Velocity = drawdown depth / recovery duration
            velocity = r.drawdown_depth / r.recovery_duration
            velocities.append(velocity)

    if not velocities:
        return {
            "avg_velocity": 0.0,
            "max_velocity": 0.0,
            "min_velocity": 0.0,
            "velocity_std": 0.0,
        }

    return {
        "avg_velocity": float(np.mean(velocities)),
        "max_velocity": float(np.max(velocities)),
        "min_velocity": float(np.min(velocities)),
        "velocity_std": float(np.std(velocities)),
    }
