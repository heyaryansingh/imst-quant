"""Signal generation from ML predictions and technical indicators (TRAD-03).

This module converts model probability outputs and price data into actionable
trading signals. Signals are discretized into long (+1), short (-1), and
neutral (0) positions based on configurable thresholds.

Includes both ML-based signal generation and technical analysis signals:
- prediction_to_signal: Convert ML probability predictions
- momentum_signal: Momentum-based trend following
- mean_reversion_signal: Mean reversion (RSI-style)
- crossover_signal: Moving average crossover signals
- composite_signal: Combine multiple signals with weights

Example:
    >>> import polars as pl
    >>> from imst_quant.trading.signals import momentum_signal, composite_signal
    >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 106]})
    >>> df = momentum_signal(df, lookback=3)
    >>> print(df["momentum_signal"].to_list())
"""

from typing import Dict, List, Optional

import polars as pl


def prediction_to_signal(
    df: pl.DataFrame,
    prob_col: str = "prob_up",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Convert model probability predictions to discrete trading signals.

    Transforms continuous probability outputs from classification models
    into actionable trading signals. Uses symmetric thresholds around 0.5
    to determine long, short, or neutral positions.

    Args:
        df: DataFrame containing prediction probabilities. Must have
            the column specified by prob_col.
        prob_col: Name of the column containing upward movement probability.
            Values should be in range [0, 1]. Defaults to "prob_up".
        threshold: Probability threshold for signal generation.
            - prob > threshold: long signal (+1)
            - prob < (1 - threshold): short signal (-1)
            - otherwise: neutral signal (0)
            Defaults to 0.5 (equal threshold).

    Returns:
        DataFrame with new "signal" column containing integer values:
        1 for long, -1 for short, 0 for neutral.

    Example:
        >>> df = pl.DataFrame({"prob_up": [0.8, 0.2, 0.5]})
        >>> result = prediction_to_signal(df, threshold=0.6)
        >>> result["signal"].to_list()  # [1, -1, 0]
    """
    return df.with_columns(
        pl.when(pl.col(prob_col) > threshold)
        .then(1)
        .when(pl.col(prob_col) < (1 - threshold))
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )


def momentum_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    lookback: int = 20,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Generate momentum-based trading signals.

    Calculates return over lookback period and generates signals based
    on momentum direction and magnitude.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        lookback: Number of periods for momentum calculation. Defaults to 20.
        threshold: Minimum absolute return to generate signal. Defaults to 0.

    Returns:
        DataFrame with new columns:
        - momentum: Return over lookback period
        - momentum_signal: 1 (uptrend), -1 (downtrend), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({"close": [100, 105, 110, 108, 115]})
        >>> result = momentum_signal(df, lookback=2)
    """
    df = df.with_columns(
        ((pl.col(price_col) / pl.col(price_col).shift(lookback)) - 1).alias("momentum")
    )

    return df.with_columns(
        pl.when(pl.col("momentum") > threshold)
        .then(1)
        .when(pl.col("momentum") < -threshold)
        .then(-1)
        .otherwise(0)
        .alias("momentum_signal")
    )


def mean_reversion_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    lookback: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pl.DataFrame:
    """Generate mean reversion signals based on RSI-style indicator.

    Calculates a relative strength indicator and generates buy signals
    when oversold and sell signals when overbought.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        lookback: Period for RSI calculation. Defaults to 14.
        oversold: RSI level below which to buy. Defaults to 30.
        overbought: RSI level above which to sell. Defaults to 70.

    Returns:
        DataFrame with new columns:
        - rsi: Relative strength index (0-100)
        - reversion_signal: 1 (buy oversold), -1 (sell overbought), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({"close": [100, 98, 95, 92, 90, 93, 96]})
        >>> result = mean_reversion_signal(df, lookback=3)
    """
    # Calculate price changes
    df = df.with_columns(
        (pl.col(price_col) - pl.col(price_col).shift(1)).alias("_change")
    )

    # Separate gains and losses
    df = df.with_columns([
        pl.when(pl.col("_change") > 0)
        .then(pl.col("_change"))
        .otherwise(0)
        .alias("_gain"),
        pl.when(pl.col("_change") < 0)
        .then(-pl.col("_change"))
        .otherwise(0)
        .alias("_loss"),
    ])

    # Calculate average gain and loss
    df = df.with_columns([
        pl.col("_gain").rolling_mean(window_size=lookback).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(window_size=lookback).alias("_avg_loss"),
    ])

    # Calculate RSI
    df = df.with_columns(
        pl.when(pl.col("_avg_loss") == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))))
        .alias("rsi")
    )

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("rsi") < oversold)
        .then(1)  # Buy when oversold
        .when(pl.col("rsi") > overbought)
        .then(-1)  # Sell when overbought
        .otherwise(0)
        .alias("reversion_signal")
    )

    # Clean up temporary columns
    return df.drop(["_change", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def crossover_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
) -> pl.DataFrame:
    """Generate moving average crossover signals.

    Calculates fast and slow moving averages and generates signals
    when they cross.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        fast_period: Period for fast moving average. Defaults to 10.
        slow_period: Period for slow moving average. Defaults to 30.

    Returns:
        DataFrame with new columns:
        - ma_fast: Fast moving average
        - ma_slow: Slow moving average
        - crossover_signal: 1 (fast > slow), -1 (fast < slow)

    Example:
        >>> df = pl.DataFrame({"close": list(range(50, 100))})
        >>> result = crossover_signal(df, fast_period=5, slow_period=10)
    """
    df = df.with_columns([
        pl.col(price_col).rolling_mean(window_size=fast_period).alias("ma_fast"),
        pl.col(price_col).rolling_mean(window_size=slow_period).alias("ma_slow"),
    ])

    return df.with_columns(
        pl.when(pl.col("ma_fast") > pl.col("ma_slow"))
        .then(1)
        .when(pl.col("ma_fast") < pl.col("ma_slow"))
        .then(-1)
        .otherwise(0)
        .alias("crossover_signal")
    )


def composite_signal(
    df: pl.DataFrame,
    signal_weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.3,
) -> pl.DataFrame:
    """Combine multiple signals into a composite signal.

    Calculates weighted average of specified signal columns and
    discretizes into trading signals.

    Args:
        df: DataFrame containing signal columns.
        signal_weights: Dict mapping signal column names to weights.
            If None, uses equal weights for all *_signal columns.
        threshold: Minimum weighted average to generate signal.
            Defaults to 0.3.

    Returns:
        DataFrame with new columns:
        - composite_score: Weighted average of signals (-1 to 1)
        - composite_signal: Discretized final signal

    Example:
        >>> df = pl.DataFrame({
        ...     "momentum_signal": [1, 1, 0, -1],
        ...     "reversion_signal": [0, 1, 1, -1],
        ... })
        >>> result = composite_signal(df, {"momentum_signal": 0.6, "reversion_signal": 0.4})
    """
    if signal_weights is None:
        # Auto-detect signal columns
        signal_cols = [c for c in df.columns if c.endswith("_signal")]
        if not signal_cols:
            df = df.with_columns([
                pl.lit(0.0).alias("composite_score"),
                pl.lit(0).alias("composite_signal"),
            ])
            return df
        signal_weights = {c: 1.0 / len(signal_cols) for c in signal_cols}

    # Calculate weighted sum
    weighted_sum = sum(
        pl.col(col) * weight for col, weight in signal_weights.items()
    )

    df = df.with_columns(weighted_sum.alias("composite_score"))

    return df.with_columns(
        pl.when(pl.col("composite_score") > threshold)
        .then(1)
        .when(pl.col("composite_score") < -threshold)
        .then(-1)
        .otherwise(0)
        .alias("composite_signal")
    )
