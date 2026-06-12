"""Sentiment-to-signal conversion utilities.

Converts aggregated sentiment scores into actionable trading signals with
configurable thresholds, momentum detection, and contrarian logic.

Functions:
    sentiment_to_signal: Convert raw sentiment scores to directional signals
    sentiment_momentum_signal: Generate signals from sentiment rate-of-change
    sentiment_extreme_signal: Detect extreme sentiment for contrarian trades
    sentiment_crossover_signal: Generate signals from sentiment moving average crossovers
    composite_sentiment_signal: Combine multiple sentiment signal types

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.sentiment_signals import sentiment_to_signal
    >>> df = pl.DataFrame({
    ...     "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    ...     "sentiment": [0.3, 0.6, -0.2],
    ... })
    >>> result = sentiment_to_signal(df, sentiment_col="sentiment")
    >>> print(result["sentiment_signal"].to_list())
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl


@dataclass
class SentimentSignalConfig:
    """Configuration for sentiment signal generation.

    Attributes:
        bullish_threshold: Sentiment above this triggers a buy signal.
        bearish_threshold: Sentiment below this triggers a sell signal.
        extreme_high: Upper extreme for contrarian signals.
        extreme_low: Lower extreme for contrarian signals.
        momentum_window: Window for sentiment rate-of-change.
        smoothing_window: Window for sentiment smoothing.
        min_confidence: Minimum confidence to emit a signal.
    """

    bullish_threshold: float = 0.2
    bearish_threshold: float = -0.2
    extreme_high: float = 0.8
    extreme_low: float = -0.8
    momentum_window: int = 5
    smoothing_window: int = 3
    min_confidence: float = 0.0


def sentiment_to_signal(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    confidence_col: Optional[str] = None,
    config: Optional[SentimentSignalConfig] = None,
) -> pl.DataFrame:
    """Convert raw sentiment scores to directional trading signals.

    Maps continuous sentiment scores to discrete signals using threshold-based
    classification. Optionally filters by confidence.

    Args:
        df: DataFrame with sentiment scores.
        sentiment_col: Column containing sentiment values (-1 to 1).
        confidence_col: Optional column with confidence scores (0 to 1).
        config: Signal generation configuration.

    Returns:
        DataFrame with added columns:
        - sentiment_signal: Directional signal (1=buy, -1=sell, 0=hold)
        - sentiment_strength: Absolute sentiment magnitude (0 to 1)
    """
    if config is None:
        config = SentimentSignalConfig()

    result = df.with_columns(
        pl.when(pl.col(sentiment_col) >= config.bullish_threshold)
        .then(1)
        .when(pl.col(sentiment_col) <= config.bearish_threshold)
        .then(-1)
        .otherwise(0)
        .alias("sentiment_signal"),
        pl.col(sentiment_col).abs().alias("sentiment_strength"),
    )

    # Filter by confidence if provided
    if confidence_col and confidence_col in df.columns:
        result = result.with_columns(
            pl.when(pl.col(confidence_col) < config.min_confidence)
            .then(0)
            .otherwise(pl.col("sentiment_signal"))
            .alias("sentiment_signal")
        )

    return result


def sentiment_momentum_signal(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    window: int = 5,
    momentum_threshold: float = 0.1,
) -> pl.DataFrame:
    """Generate signals from sentiment rate-of-change (momentum).

    Detects when sentiment is rapidly improving or deteriorating,
    generating signals on acceleration rather than level.

    Args:
        df: DataFrame with sentiment scores.
        sentiment_col: Column containing sentiment values.
        window: Lookback window for momentum calculation.
        momentum_threshold: Minimum rate-of-change to trigger signal.

    Returns:
        DataFrame with added columns:
        - sentiment_momentum: Rate of change in sentiment
        - sentiment_momentum_signal: 1=accelerating bullish, -1=accelerating bearish, 0=flat
    """
    result = df.with_columns(
        (pl.col(sentiment_col) - pl.col(sentiment_col).shift(window)).alias(
            "sentiment_momentum"
        )
    )

    result = result.with_columns(
        pl.when(pl.col("sentiment_momentum") >= momentum_threshold)
        .then(1)
        .when(pl.col("sentiment_momentum") <= -momentum_threshold)
        .then(-1)
        .otherwise(0)
        .alias("sentiment_momentum_signal")
    )

    return result


def sentiment_extreme_signal(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    extreme_high: float = 0.8,
    extreme_low: float = -0.8,
) -> pl.DataFrame:
    """Detect extreme sentiment levels for contrarian trading signals.

    Generates contrarian signals when sentiment reaches extreme levels,
    based on the mean-reversion tendency of crowd sentiment.

    Args:
        df: DataFrame with sentiment scores.
        sentiment_col: Column containing sentiment values.
        extreme_high: Upper threshold for extreme bullishness (contrarian sell).
        extreme_low: Lower threshold for extreme bearishness (contrarian buy).

    Returns:
        DataFrame with added columns:
        - sentiment_extreme: Boolean flag for extreme sentiment
        - contrarian_signal: -1 at extreme highs (sell), 1 at extreme lows (buy), 0 otherwise
    """
    result = df.with_columns(
        (
            (pl.col(sentiment_col) >= extreme_high)
            | (pl.col(sentiment_col) <= extreme_low)
        ).alias("sentiment_extreme"),
        pl.when(pl.col(sentiment_col) >= extreme_high)
        .then(-1)
        .when(pl.col(sentiment_col) <= extreme_low)
        .then(1)
        .otherwise(0)
        .alias("contrarian_signal"),
    )

    return result


def sentiment_crossover_signal(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    fast_window: int = 3,
    slow_window: int = 10,
) -> pl.DataFrame:
    """Generate signals from sentiment moving average crossovers.

    Similar to price MA crossovers, this detects when short-term sentiment
    crosses above or below long-term sentiment.

    Args:
        df: DataFrame with sentiment scores.
        sentiment_col: Column containing sentiment values.
        fast_window: Short-term smoothing window.
        slow_window: Long-term smoothing window.

    Returns:
        DataFrame with added columns:
        - sentiment_fast_ma: Short-term sentiment moving average
        - sentiment_slow_ma: Long-term sentiment moving average
        - crossover_signal: 1=bullish crossover, -1=bearish crossover, 0=no crossover
    """
    result = df.with_columns(
        pl.col(sentiment_col)
        .rolling_mean(window_size=fast_window)
        .alias("sentiment_fast_ma"),
        pl.col(sentiment_col)
        .rolling_mean(window_size=slow_window)
        .alias("sentiment_slow_ma"),
    )

    # Detect crossovers: fast crosses above slow = bullish
    spread = pl.col("sentiment_fast_ma") - pl.col("sentiment_slow_ma")
    prev_spread = spread.shift(1)

    result = result.with_columns(
        pl.when((spread > 0) & (prev_spread <= 0))
        .then(1)
        .when((spread < 0) & (prev_spread >= 0))
        .then(-1)
        .otherwise(0)
        .alias("crossover_signal")
    )

    return result


def composite_sentiment_signal(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    confidence_col: Optional[str] = None,
    config: Optional[SentimentSignalConfig] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Combine multiple sentiment signal types into a composite signal.

    Applies threshold, momentum, extreme, and crossover signals, then
    combines them with configurable weights into a single composite score.

    Args:
        df: DataFrame with sentiment scores.
        sentiment_col: Column containing sentiment values.
        confidence_col: Optional confidence column.
        config: Signal generation configuration.
        weights: Weights for each signal type. Keys:
            - threshold: Weight for basic threshold signal (default 0.4)
            - momentum: Weight for momentum signal (default 0.3)
            - contrarian: Weight for contrarian extreme signal (default 0.15)
            - crossover: Weight for crossover signal (default 0.15)

    Returns:
        DataFrame with all individual signals plus:
        - composite_sentiment_score: Weighted combination of all signals (-1 to 1)
        - composite_sentiment_signal: Discretized composite (1, -1, 0)
    """
    if config is None:
        config = SentimentSignalConfig()

    if weights is None:
        weights = {
            "threshold": 0.4,
            "momentum": 0.3,
            "contrarian": 0.15,
            "crossover": 0.15,
        }

    # Apply all signal generators
    result = sentiment_to_signal(df, sentiment_col, confidence_col, config)
    result = sentiment_momentum_signal(
        result, sentiment_col, config.momentum_window
    )
    result = sentiment_extreme_signal(
        result, sentiment_col, config.extreme_high, config.extreme_low
    )
    result = sentiment_crossover_signal(
        result, sentiment_col, config.smoothing_window
    )

    # Combine signals with weights
    composite = (
        pl.col("sentiment_signal") * weights.get("threshold", 0.4)
        + pl.col("sentiment_momentum_signal") * weights.get("momentum", 0.3)
        + pl.col("contrarian_signal") * weights.get("contrarian", 0.15)
        + pl.col("crossover_signal") * weights.get("crossover", 0.15)
    )

    result = result.with_columns(
        composite.alias("composite_sentiment_score"),
        pl.when(composite > 0.2)
        .then(1)
        .when(composite < -0.2)
        .then(-1)
        .otherwise(0)
        .alias("composite_sentiment_signal"),
    )

    return result


def calculate_sentiment_divergence(
    df: pl.DataFrame,
    sentiment_col: str = "sentiment",
    price_returns_col: str = "returns",
    window: int = 10,
) -> pl.DataFrame:
    """Detect divergence between sentiment and price action.

    Identifies periods where sentiment direction disagrees with price
    direction, which can indicate upcoming reversals.

    Args:
        df: DataFrame with sentiment and returns columns.
        sentiment_col: Column containing sentiment values.
        price_returns_col: Column containing price returns.
        window: Rolling window for smoothing.

    Returns:
        DataFrame with added columns:
        - sentiment_direction: Rolling sentiment trend (1=up, -1=down)
        - price_direction: Rolling price trend (1=up, -1=down)
        - sentiment_price_divergence: True when directions disagree
        - divergence_signal: 1=bullish divergence (neg sentiment, pos price),
            -1=bearish divergence (pos sentiment, neg price)
    """
    result = df.with_columns(
        pl.when(
            pl.col(sentiment_col).rolling_mean(window_size=window)
            > pl.col(sentiment_col).rolling_mean(window_size=window).shift(1)
        )
        .then(1)
        .otherwise(-1)
        .alias("sentiment_direction"),
        pl.when(
            pl.col(price_returns_col).rolling_sum(window_size=window) > 0
        )
        .then(1)
        .otherwise(-1)
        .alias("price_direction"),
    )

    result = result.with_columns(
        (pl.col("sentiment_direction") != pl.col("price_direction")).alias(
            "sentiment_price_divergence"
        ),
        pl.when(
            (pl.col("sentiment_direction") < 0)
            & (pl.col("price_direction") > 0)
        )
        .then(1)
        .when(
            (pl.col("sentiment_direction") > 0)
            & (pl.col("price_direction") < 0)
        )
        .then(-1)
        .otherwise(0)
        .alias("divergence_signal"),
    )

    return result
