"""Signal consistency and stability analysis utilities.

This module provides utilities for measuring the stability and consistency
of trading signals over time. Helps identify reliable vs noisy signals
and detect signal degradation.

Functions:
    calculate_signal_persistence: Measure how long signals persist
    analyze_signal_flip_rate: Track frequency of signal direction changes
    measure_signal_strength_stability: Assess consistency of signal magnitude
    detect_signal_degradation: Identify periods of declining signal quality
    signal_consistency_score: Overall signal reliability metric

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.signal_consistency import (
    ...     calculate_signal_persistence,
    ...     signal_consistency_score
    ... )
    >>> signals = pl.Series([1, 1, 1, -1, -1, 1, 1, 1, 1])
    >>> persistence = calculate_signal_persistence(signals)
    >>> score = signal_consistency_score(signals)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


@dataclass
class SignalPersistenceMetrics:
    """Metrics describing signal persistence characteristics.

    Attributes:
        avg_run_length: Average consecutive periods with same signal
        max_run_length: Longest consecutive period with same signal
        min_run_length: Shortest run (excluding single periods)
        num_runs: Total number of distinct signal runs
        signal_flip_rate: Frequency of signal direction changes
    """

    avg_run_length: float
    max_run_length: int
    min_run_length: int
    num_runs: int
    signal_flip_rate: float


def calculate_signal_persistence(
    signals: Union[pl.Series, pl.DataFrame],
    signal_col: str = "signal",
) -> SignalPersistenceMetrics:
    """Calculate how long trading signals persist before changing.

    Measures the stability of signals by analyzing consecutive
    periods with the same signal direction.

    Args:
        signals: Series or DataFrame with signal values (1, 0, -1)
        signal_col: Column name if signals is a DataFrame

    Returns:
        SignalPersistenceMetrics with run length statistics

    Example:
        >>> signals = pl.Series([1, 1, 1, -1, -1, 1, 0, 0, 1, 1])
        >>> metrics = calculate_signal_persistence(signals)
        >>> print(f"Avg run: {metrics.avg_run_length:.1f} periods")
    """
    if isinstance(signals, pl.DataFrame):
        signal_series = signals[signal_col]
    else:
        signal_series = signals

    signal_list = signal_series.to_list()

    if len(signal_list) == 0:
        return SignalPersistenceMetrics(
            avg_run_length=0.0,
            max_run_length=0,
            min_run_length=0,
            num_runs=0,
            signal_flip_rate=0.0,
        )

    # Find runs of consecutive identical signals
    runs = []
    current_signal = signal_list[0]
    current_run_length = 1

    for i in range(1, len(signal_list)):
        if signal_list[i] == current_signal:
            current_run_length += 1
        else:
            runs.append((current_signal, current_run_length))
            current_signal = signal_list[i]
            current_run_length = 1

    # Add final run
    runs.append((current_signal, current_run_length))

    # Calculate metrics
    run_lengths = [length for _, length in runs]
    num_flips = len(runs) - 1

    return SignalPersistenceMetrics(
        avg_run_length=float(np.mean(run_lengths)),
        max_run_length=max(run_lengths),
        min_run_length=min(run_lengths),
        num_runs=len(runs),
        signal_flip_rate=num_flips / len(signal_list) if len(signal_list) > 0 else 0.0,
    )


def analyze_signal_flip_rate(
    signals: Union[pl.Series, pl.DataFrame],
    signal_col: str = "signal",
    window_size: int = 20,
) -> Dict[str, float]:
    """Analyze frequency of signal direction changes over time.

    Computes rolling flip rate to detect periods of signal instability.

    Args:
        signals: Series or DataFrame with signal values
        signal_col: Column name if signals is a DataFrame
        window_size: Window for rolling flip rate calculation

    Returns:
        Dictionary with:
        - overall_flip_rate: Total flips / total periods
        - avg_rolling_flip_rate: Average of rolling flip rates
        - max_flip_rate_period: Highest rolling flip rate
        - min_flip_rate_period: Lowest rolling flip rate

    Example:
        >>> signals = pl.Series([1, -1, 1, -1, 1, 1, 1, -1, -1, -1])
        >>> flip_stats = analyze_signal_flip_rate(signals, window_size=3)
    """
    if isinstance(signals, pl.DataFrame):
        signal_series = signals[signal_col]
    else:
        signal_series = signals

    signal_list = signal_series.to_list()
    n = len(signal_list)

    if n < 2:
        return {
            "overall_flip_rate": 0.0,
            "avg_rolling_flip_rate": 0.0,
            "max_flip_rate_period": 0.0,
            "min_flip_rate_period": 0.0,
        }

    # Calculate overall flips
    flips = sum(1 for i in range(1, n) if signal_list[i] != signal_list[i - 1])
    overall_flip_rate = flips / (n - 1)

    # Calculate rolling flip rate
    if n < window_size:
        window_size = n

    rolling_flip_rates = []
    for i in range(window_size, n + 1):
        window = signal_list[i - window_size : i]
        window_flips = sum(1 for j in range(1, len(window)) if window[j] != window[j - 1])
        rolling_rate = window_flips / (window_size - 1)
        rolling_flip_rates.append(rolling_rate)

    if rolling_flip_rates:
        avg_rolling = float(np.mean(rolling_flip_rates))
        max_rolling = float(np.max(rolling_flip_rates))
        min_rolling = float(np.min(rolling_flip_rates))
    else:
        avg_rolling = overall_flip_rate
        max_rolling = overall_flip_rate
        min_rolling = overall_flip_rate

    return {
        "overall_flip_rate": overall_flip_rate,
        "avg_rolling_flip_rate": avg_rolling,
        "max_flip_rate_period": max_rolling,
        "min_flip_rate_period": min_rolling,
    }


def measure_signal_strength_stability(
    signal_strengths: Union[pl.Series, pl.DataFrame],
    strength_col: str = "strength",
    window_size: int = 20,
) -> Dict[str, float]:
    """Measure consistency of signal magnitude over time.

    Analyzes whether signal strengths (confidence levels) are stable
    or highly variable, which impacts reliability.

    Args:
        signal_strengths: Series or DataFrame with signal strength/confidence values
        strength_col: Column name if DataFrame
        window_size: Window for rolling volatility calculation

    Returns:
        Dictionary with:
        - avg_strength: Mean signal strength
        - strength_volatility: Std dev of strengths
        - rolling_stability: Average rolling coefficient of variation
        - strength_range: Max - min strength observed

    Example:
        >>> strengths = pl.Series([0.8, 0.75, 0.9, 0.85, 0.2, 0.3, 0.7])
        >>> stability = measure_signal_strength_stability(strengths)
    """
    if isinstance(signal_strengths, pl.DataFrame):
        strength_series = signal_strengths[strength_col]
    else:
        strength_series = signal_strengths

    strength_list = strength_series.to_list()
    n = len(strength_list)

    if n == 0:
        return {
            "avg_strength": 0.0,
            "strength_volatility": 0.0,
            "rolling_stability": 0.0,
            "strength_range": 0.0,
        }

    # Overall metrics
    avg_strength = float(np.mean(strength_list))
    strength_vol = float(np.std(strength_list))
    strength_range = float(np.max(strength_list) - np.min(strength_list))

    # Rolling coefficient of variation
    if n < window_size:
        window_size = n

    rolling_cvs = []
    for i in range(window_size, n + 1):
        window = strength_list[i - window_size : i]
        window_mean = np.mean(window)
        window_std = np.std(window)
        cv = window_std / window_mean if window_mean != 0 else 0.0
        rolling_cvs.append(cv)

    rolling_stability = float(np.mean(rolling_cvs)) if rolling_cvs else 0.0

    return {
        "avg_strength": avg_strength,
        "strength_volatility": strength_vol,
        "rolling_stability": rolling_stability,
        "strength_range": strength_range,
    }


def detect_signal_degradation(
    signals: Union[pl.Series, pl.DataFrame],
    signal_col: str = "signal",
    returns: Optional[Union[pl.Series, pl.DataFrame]] = None,
    return_col: str = "returns",
    window_size: int = 50,
    lookback_periods: int = 3,
) -> List[Dict[str, any]]:
    """Detect periods where signal quality degrades.

    Identifies windows where signals become less predictive or
    less reliable compared to recent history.

    Args:
        signals: Series or DataFrame with signal values
        signal_col: Column name for signals
        returns: Optional returns data to measure signal effectiveness
        return_col: Column name for returns
        window_size: Window for quality measurement
        lookback_periods: Number of windows to compare against

    Returns:
        List of degradation events with:
        - period_start: Start index of degraded period
        - quality_before: Average quality in lookback
        - quality_current: Quality in current window
        - degradation_pct: Percentage decline in quality

    Example:
        >>> signals = pl.Series([1, 1, -1, -1, 1, 0, 0, -1, 1, -1])
        >>> returns = pl.Series([0.01, 0.02, -0.01, -0.02, 0.01, 0, 0, -0.01, 0.005, -0.005])
        >>> degradation = detect_signal_degradation(signals, returns=returns)
    """
    if isinstance(signals, pl.DataFrame):
        signal_series = signals[signal_col]
    else:
        signal_series = signals

    signal_list = signal_series.to_list()
    n = len(signal_list)

    degradation_events = []

    # If returns provided, use signal-return correlation as quality metric
    if returns is not None:
        if isinstance(returns, pl.DataFrame):
            return_series = returns[return_col]
        else:
            return_series = returns

        return_list = return_series.to_list()

        if len(return_list) != n:
            raise ValueError("Signals and returns must have same length")

        # Calculate rolling signal quality (correlation)
        for i in range(window_size * (lookback_periods + 1), n, window_size // 2):
            # Current window quality
            current_window_signals = signal_list[i - window_size : i]
            current_window_returns = return_list[i - window_size : i]
            current_quality = float(
                np.corrcoef(current_window_signals, current_window_returns)[0, 1]
            )

            # Lookback windows quality
            lookback_qualities = []
            for j in range(1, lookback_periods + 1):
                start = i - window_size * (j + 1)
                end = i - window_size * j
                lb_signals = signal_list[start:end]
                lb_returns = return_list[start:end]
                lb_quality = float(np.corrcoef(lb_signals, lb_returns)[0, 1])
                lookback_qualities.append(lb_quality)

            avg_lookback_quality = float(np.mean(lookback_qualities))

            # Check for degradation
            if avg_lookback_quality > 0.1:  # Only flag if previously had some signal
                degradation_pct = (
                    (avg_lookback_quality - current_quality) / avg_lookback_quality
                )

                if degradation_pct > 0.3:  # 30% degradation threshold
                    degradation_events.append(
                        {
                            "period_start": i - window_size,
                            "quality_before": avg_lookback_quality,
                            "quality_current": current_quality,
                            "degradation_pct": degradation_pct,
                        }
                    )

    else:
        # Without returns, use flip rate as quality proxy (lower is better)
        for i in range(window_size * (lookback_periods + 1), n, window_size // 2):
            # Current window flip rate
            current_window = signal_list[i - window_size : i]
            current_flips = sum(
                1 for j in range(1, len(current_window)) if current_window[j] != current_window[j - 1]
            )
            current_quality = 1.0 - (current_flips / (len(current_window) - 1))

            # Lookback windows flip rate
            lookback_qualities = []
            for j in range(1, lookback_periods + 1):
                start = i - window_size * (j + 1)
                end = i - window_size * j
                lb_window = signal_list[start:end]
                lb_flips = sum(
                    1 for k in range(1, len(lb_window)) if lb_window[k] != lb_window[k - 1]
                )
                lb_quality = 1.0 - (lb_flips / (len(lb_window) - 1))
                lookback_qualities.append(lb_quality)

            avg_lookback_quality = float(np.mean(lookback_qualities))

            # Check for degradation (flip rate increased)
            degradation_pct = (avg_lookback_quality - current_quality) / avg_lookback_quality

            if degradation_pct > 0.3:
                degradation_events.append(
                    {
                        "period_start": i - window_size,
                        "quality_before": avg_lookback_quality,
                        "quality_current": current_quality,
                        "degradation_pct": degradation_pct,
                    }
                )

    return degradation_events


def signal_consistency_score(
    signals: Union[pl.Series, pl.DataFrame],
    signal_col: str = "signal",
    signal_strengths: Optional[Union[pl.Series, pl.DataFrame]] = None,
    strength_col: str = "strength",
) -> float:
    """Calculate overall signal consistency score (0-1 scale).

    Combines persistence, flip rate, and strength stability into
    a single score. Higher scores indicate more reliable signals.

    Args:
        signals: Series or DataFrame with signal values
        signal_col: Column name for signals
        signal_strengths: Optional signal strength/confidence values
        strength_col: Column name for strengths

    Returns:
        Consistency score from 0 (worst) to 1 (best)

    Example:
        >>> signals = pl.Series([1, 1, 1, -1, -1, -1, 1, 1, 1])
        >>> strengths = pl.Series([0.9, 0.85, 0.8, 0.75, 0.8, 0.85, 0.9, 0.88, 0.92])
        >>> score = signal_consistency_score(signals, signal_strengths=strengths)
    """
    # Persistence score (longer runs = better)
    persistence = calculate_signal_persistence(signals, signal_col=signal_col)
    avg_run = persistence.avg_run_length
    persistence_score = min(1.0, avg_run / 10.0)  # Normalize to 10-period target

    # Flip rate score (lower flip rate = better)
    flip_stats = analyze_signal_flip_rate(signals, signal_col=signal_col)
    flip_rate = flip_stats["overall_flip_rate"]
    flip_score = 1.0 - flip_rate  # Invert so lower flip rate = higher score

    # Strength stability score (if provided)
    if signal_strengths is not None:
        strength_stats = measure_signal_strength_stability(
            signal_strengths, strength_col=strength_col
        )
        # Lower rolling stability (CV) = better
        rolling_cv = strength_stats["rolling_stability"]
        strength_score = 1.0 / (1.0 + rolling_cv)
    else:
        strength_score = 0.5  # Neutral if not provided

    # Combine scores (weighted average)
    if signal_strengths is not None:
        final_score = 0.35 * persistence_score + 0.35 * flip_score + 0.30 * strength_score
    else:
        final_score = 0.5 * persistence_score + 0.5 * flip_score

    return min(1.0, max(0.0, final_score))
