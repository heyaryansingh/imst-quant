"""Multi-timeframe signal aggregator for combining trading signals across different time horizons.

This module provides tools to aggregate trading signals from multiple timeframes
(e.g., daily, weekly, monthly) to generate more robust trading decisions.
"""

from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger()


class SignalStrength(Enum):
    """Enum for signal strength levels."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class AggregationMethod(Enum):
    """Enum for signal aggregation methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"


def aggregate_timeframe_signals(
    signals: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
) -> float:
    """Aggregate signals from multiple timeframes into a single signal.

    Args:
        signals: Dictionary mapping timeframe names to signal values (-1 to 1).
            Example: {'1d': 0.5, '1w': 0.3, '1m': -0.2}
        weights: Optional dictionary mapping timeframe names to weights.
            If None, equal weights are used.
        method: Aggregation method to use.

    Returns:
        Aggregated signal value (-1 to 1).

    Example:
        >>> signals = {'1d': 0.5, '1w': 0.3, '1m': -0.2}
        >>> weights = {'1d': 0.5, '1w': 0.3, '1m': 0.2}
        >>> aggregate_timeframe_signals(signals, weights)
        0.27
    """
    if not signals:
        return 0.0

    if weights is None:
        weights = {tf: 1.0 / len(signals) for tf in signals.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {tf: w / total_weight for tf, w in weights.items()}

    if method == AggregationMethod.WEIGHTED_AVERAGE:
        agg_signal = sum(signals[tf] * normalized_weights.get(tf, 0) for tf in signals)

    elif method == AggregationMethod.MAJORITY_VOTE:
        # Convert signals to discrete votes
        votes = [1 if s > 0 else -1 if s < 0 else 0 for s in signals.values()]
        agg_signal = 1.0 if sum(votes) > 0 else -1.0 if sum(votes) < 0 else 0.0

    elif method == AggregationMethod.CONSENSUS:
        # All timeframes must agree in direction
        all_positive = all(s > 0 for s in signals.values())
        all_negative = all(s < 0 for s in signals.values())
        if all_positive:
            agg_signal = min(signals.values())  # Most conservative positive
        elif all_negative:
            agg_signal = max(signals.values())  # Most conservative negative
        else:
            agg_signal = 0.0

    elif method == AggregationMethod.HIERARCHICAL:
        # Longer timeframes override shorter ones
        sorted_tfs = sorted(signals.keys(), reverse=True)  # Longest first
        agg_signal = signals[sorted_tfs[0]]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Clip to valid range
    agg_signal = max(-1.0, min(1.0, agg_signal))

    logger.debug(
        "signals_aggregated",
        method=method.value,
        num_timeframes=len(signals),
        result=agg_signal,
    )

    return agg_signal


def classify_signal_strength(signal: float) -> SignalStrength:
    """Classify a signal value into a strength category.

    Args:
        signal: Signal value (-1 to 1).

    Returns:
        SignalStrength enum value.

    Example:
        >>> classify_signal_strength(0.8)
        <SignalStrength.STRONG_BUY: 2>
        >>> classify_signal_strength(-0.3)
        <SignalStrength.SELL: -1>
    """
    if signal >= 0.6:
        return SignalStrength.STRONG_BUY
    elif signal >= 0.2:
        return SignalStrength.BUY
    elif signal <= -0.6:
        return SignalStrength.STRONG_SELL
    elif signal <= -0.2:
        return SignalStrength.SELL
    else:
        return SignalStrength.NEUTRAL


def calculate_timeframe_confidence(
    signals: Dict[str, float],
    threshold: float = 0.7,
) -> float:
    """Calculate confidence score based on timeframe alignment.

    Args:
        signals: Dictionary mapping timeframe names to signal values.
        threshold: Minimum absolute signal value to consider as "strong".

    Returns:
        Confidence score (0 to 1), where 1 means all timeframes strongly agree.

    Example:
        >>> signals = {'1d': 0.8, '1w': 0.7, '1m': 0.9}
        >>> calculate_timeframe_confidence(signals)
        1.0  # All strong and aligned
    """
    if not signals:
        return 0.0

    # Check if all signals point in the same direction
    all_positive = all(s >= 0 for s in signals.values())
    all_negative = all(s <= 0 for s in signals.values())

    if not (all_positive or all_negative):
        # Mixed signals = low confidence
        return 0.0

    # Calculate what proportion of signals are "strong"
    strong_signals = sum(1 for s in signals.values() if abs(s) >= threshold)
    confidence = strong_signals / len(signals)

    logger.debug(
        "confidence_calculated",
        num_timeframes=len(signals),
        strong_signals=strong_signals,
        confidence=confidence,
    )

    return confidence


def generate_multi_timeframe_signals(
    df: pd.DataFrame,
    signal_col: str = "signal",
    timeframes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate signals across multiple timeframes from a single timeframe dataset.

    Args:
        df: DataFrame with datetime index and a signal column.
        signal_col: Name of the column containing raw signals.
        timeframes: List of timeframe periods (e.g., ['1D', '1W', '1M']).
            If None, defaults to ['1D', '1W', '1M'].

    Returns:
        DataFrame with columns for each timeframe signal, aggregated signal,
        and confidence score.

    Example:
        >>> df = pd.DataFrame({
        ...     'signal': [0.5, 0.3, -0.2, 0.1],
        ... }, index=pd.date_range('2024-01-01', periods=4))
        >>> result = generate_multi_timeframe_signals(df)
    """
    if timeframes is None:
        timeframes = ["1D", "1W", "1M"]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    if signal_col not in df.columns:
        raise ValueError(f"Column '{signal_col}' not found in DataFrame")

    result = df[[signal_col]].copy()

    # Resample to each timeframe
    for tf in timeframes:
        tf_signal = df[signal_col].resample(tf).mean()
        # Forward fill to align with original index
        result[f"signal_{tf}"] = tf_signal.reindex(df.index, method="ffill")

    # Aggregate signals
    def agg_row_signals(row):
        tf_signals = {
            tf: row[f"signal_{tf}"]
            for tf in timeframes
            if not pd.isna(row[f"signal_{tf}"])
        }
        if not tf_signals:
            return 0.0
        return aggregate_timeframe_signals(tf_signals)

    result["aggregated_signal"] = result.apply(agg_row_signals, axis=1)

    # Calculate confidence
    def calc_row_confidence(row):
        tf_signals = {
            tf: row[f"signal_{tf}"]
            for tf in timeframes
            if not pd.isna(row[f"signal_{tf}"])
        }
        if not tf_signals:
            return 0.0
        return calculate_timeframe_confidence(tf_signals)

    result["confidence"] = result.apply(calc_row_confidence, axis=1)

    # Classify strength
    result["strength"] = result["aggregated_signal"].apply(
        lambda x: classify_signal_strength(x).name
    )

    logger.info(
        "multi_timeframe_signals_generated",
        num_rows=len(result),
        timeframes=timeframes,
        avg_confidence=result["confidence"].mean(),
    )

    return result


def filter_high_confidence_signals(
    df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    signal_col: str = "aggregated_signal",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Filter signals to only include high-confidence periods.

    Args:
        df: DataFrame with signal and confidence columns.
        confidence_threshold: Minimum confidence to include (0 to 1).
        signal_col: Name of the signal column.
        confidence_col: Name of the confidence column.

    Returns:
        Filtered DataFrame containing only high-confidence signals.

    Example:
        >>> df = pd.DataFrame({
        ...     'aggregated_signal': [0.5, 0.3, -0.2],
        ...     'confidence': [0.8, 0.5, 0.9],
        ... })
        >>> filter_high_confidence_signals(df, confidence_threshold=0.7)
    """
    if confidence_col not in df.columns:
        raise ValueError(f"Column '{confidence_col}' not found in DataFrame")

    filtered = df[df[confidence_col] >= confidence_threshold].copy()

    logger.info(
        "signals_filtered_by_confidence",
        original_rows=len(df),
        filtered_rows=len(filtered),
        confidence_threshold=confidence_threshold,
        retention_rate=len(filtered) / len(df) if len(df) > 0 else 0,
    )

    return filtered
