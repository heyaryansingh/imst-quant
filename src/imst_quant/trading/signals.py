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


def breakout_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    lookback: int = 20,
    breakout_threshold: float = 0.0,
) -> pl.DataFrame:
    """Generate breakout trading signals based on Donchian channel.

    Identifies breakouts above the highest high or below the lowest low
    over the lookback period. Useful for trend-following strategies.

    Args:
        df: DataFrame containing price data with high/low columns.
        price_col: Name of the closing price column. Defaults to "close".
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        lookback: Number of periods for channel calculation. Defaults to 20.
        breakout_threshold: Minimum % above/below channel for signal.
            Defaults to 0 (any breakout generates signal).

    Returns:
        DataFrame with new columns:
        - channel_high: Highest high over lookback period
        - channel_low: Lowest low over lookback period
        - breakout_signal: 1 (upside breakout), -1 (downside breakout), 0 (no breakout)

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 105, 103, 110],
        ...     "high": [101, 103, 106, 104, 112],
        ...     "low": [99, 101, 104, 102, 108],
        ... })
        >>> result = breakout_signal(df, lookback=3)
    """
    # Calculate Donchian channel
    df = df.with_columns([
        pl.col(high_col).rolling_max(window_size=lookback).shift(1).alias("channel_high"),
        pl.col(low_col).rolling_min(window_size=lookback).shift(1).alias("channel_low"),
    ])

    # Calculate breakout percentage
    df = df.with_columns([
        ((pl.col(price_col) - pl.col("channel_high")) / pl.col("channel_high")).alias("_up_break"),
        ((pl.col("channel_low") - pl.col(price_col)) / pl.col("channel_low")).alias("_down_break"),
    ])

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("_up_break") > breakout_threshold)
        .then(1)
        .when(pl.col("_down_break") > breakout_threshold)
        .then(-1)
        .otherwise(0)
        .alias("breakout_signal")
    )

    return df.drop(["_up_break", "_down_break"])


def volatility_adjusted_signal(
    df: pl.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    vol_lookback: int = 20,
    vol_threshold: float = 1.5,
) -> pl.DataFrame:
    """Adjust trading signals based on volatility regime.

    Filters or modifies signals based on current volatility relative to
    historical average. Can reduce position sizes in high-volatility
    environments or filter signals entirely.

    Args:
        df: DataFrame containing signal and price data.
        signal_col: Name of the input signal column. Defaults to "signal".
        price_col: Name of the price column for volatility calculation.
        vol_lookback: Period for volatility calculation. Defaults to 20.
        vol_threshold: Multiple of average vol above which to filter signals.
            Defaults to 1.5 (signals filtered when vol > 1.5x average).

    Returns:
        DataFrame with new columns:
        - realized_vol: Rolling realized volatility
        - vol_ratio: Current vol / average vol
        - vol_adjusted_signal: Signal adjusted for volatility regime

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 98, 105, 95, 108, 92],
        ...     "signal": [1, 1, 0, 1, -1, 1, -1],
        ... })
        >>> result = volatility_adjusted_signal(df, vol_lookback=3)
    """
    # Calculate returns
    df = df.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1) - 1).alias("_ret")
    )

    # Calculate rolling volatility (annualized)
    df = df.with_columns(
        (pl.col("_ret").rolling_std(window_size=vol_lookback) * (252 ** 0.5))
        .alias("realized_vol")
    )

    # Calculate average volatility over longer period
    df = df.with_columns(
        pl.col("realized_vol").rolling_mean(window_size=vol_lookback * 3)
        .alias("_avg_vol")
    )

    # Calculate volatility ratio
    df = df.with_columns(
        (pl.col("realized_vol") / pl.col("_avg_vol")).alias("vol_ratio")
    )

    # Adjust signal based on volatility regime
    df = df.with_columns(
        pl.when(pl.col("vol_ratio") > vol_threshold)
        .then(0)  # Filter signals in high-vol regime
        .otherwise(pl.col(signal_col))
        .alias("vol_adjusted_signal")
    )

    return df.drop(["_ret", "_avg_vol"])


def macd_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """Generate MACD crossover trading signals.

    Calculates MACD (Moving Average Convergence Divergence) and generates
    signals based on MACD line crossing the signal line.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        fast_period: Period for fast EMA. Defaults to 12.
        slow_period: Period for slow EMA. Defaults to 26.
        signal_period: Period for signal line EMA. Defaults to 9.

    Returns:
        DataFrame with new columns:
        - macd_line: MACD line (fast EMA - slow EMA)
        - macd_signal_line: Signal line (EMA of MACD line)
        - macd_histogram: Difference between MACD and signal line
        - macd_signal: 1 (bullish crossover), -1 (bearish crossover), 0 (no cross)

    Example:
        >>> df = pl.DataFrame({"close": list(range(50, 100))})
        >>> result = macd_signal(df, fast_period=5, slow_period=10, signal_period=3)
    """
    # Calculate EMAs using Polars' ewm_mean
    df = df.with_columns([
        pl.col(price_col).ewm_mean(span=fast_period).alias("_ema_fast"),
        pl.col(price_col).ewm_mean(span=slow_period).alias("_ema_slow"),
    ])

    # Calculate MACD line
    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
    )

    # Calculate signal line
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=signal_period).alias("macd_signal_line")
    )

    # Calculate histogram
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal_line")).alias("macd_histogram")
    )

    # Generate crossover signals
    df = df.with_columns(
        pl.col("macd_histogram").shift(1).alias("_prev_hist")
    )

    df = df.with_columns(
        pl.when((pl.col("macd_histogram") > 0) & (pl.col("_prev_hist") <= 0))
        .then(1)  # Bullish crossover
        .when((pl.col("macd_histogram") < 0) & (pl.col("_prev_hist") >= 0))
        .then(-1)  # Bearish crossover
        .otherwise(0)
        .alias("macd_signal")
    )

    return df.drop(["_ema_fast", "_ema_slow", "_prev_hist"])


def signal_strength(
    df: pl.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    volume_col: Optional[str] = None,
    lookback: int = 20,
) -> pl.DataFrame:
    """Calculate confidence/strength indicator for trading signals.

    Combines multiple factors (trend strength, volume confirmation, momentum)
    to provide a confidence score for the current signal.

    Args:
        df: DataFrame containing signal and price data.
        signal_col: Name of the signal column. Defaults to "signal".
        price_col: Name of the price column. Defaults to "close".
        volume_col: Name of the volume column. If None, volume factor excluded.
        lookback: Period for strength calculations. Defaults to 20.

    Returns:
        DataFrame with new columns:
        - trend_strength: Absolute value of price momentum (0-100)
        - volume_strength: Volume relative to average (0-100), if volume provided
        - signal_strength: Combined confidence score (0-100)
        - strong_signal: Signal only when strength > 50

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 105, 108, 112, 110],
        ...     "signal": [0, 1, 1, 1, 1, -1],
        ...     "volume": [1000, 1200, 1500, 1100, 1800, 900],
        ... })
        >>> result = signal_strength(df, volume_col="volume", lookback=3)
    """
    # Calculate trend strength based on directional movement
    df = df.with_columns(
        ((pl.col(price_col) / pl.col(price_col).shift(lookback)) - 1).alias("_momentum")
    )

    # Normalize momentum to 0-100 scale using rolling percentile
    df = df.with_columns(
        (pl.col("_momentum").abs().rolling_quantile(quantile=0.5, window_size=lookback * 5))
        .alias("_median_mom")
    )

    df = df.with_columns(
        pl.when(pl.col("_median_mom") > 0)
        .then((pl.col("_momentum").abs() / (pl.col("_median_mom") * 2) * 100).clip(0, 100))
        .otherwise(50.0)
        .alias("trend_strength")
    )

    factors = [pl.col("trend_strength")]

    # Add volume strength if volume column provided
    if volume_col and volume_col in df.columns:
        df = df.with_columns(
            pl.col(volume_col).rolling_mean(window_size=lookback).alias("_avg_vol")
        )

        df = df.with_columns(
            pl.when(pl.col("_avg_vol") > 0)
            .then((pl.col(volume_col) / pl.col("_avg_vol") * 50).clip(0, 100))
            .otherwise(50.0)
            .alias("volume_strength")
        )
        factors.append(pl.col("volume_strength"))
        df = df.drop("_avg_vol")

    # Calculate combined signal strength as average of factors
    combined = sum(factors) / len(factors)
    df = df.with_columns(combined.alias("signal_strength"))

    # Generate strong signal (only when strength > 50)
    df = df.with_columns(
        pl.when(pl.col("signal_strength") > 50)
        .then(pl.col(signal_col))
        .otherwise(0)
        .alias("strong_signal")
    )

    return df.drop(["_momentum", "_median_mom"])


def bollinger_band_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    lookback: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Generate trading signals based on Bollinger Band position.

    Signals are generated when price touches or crosses the bands:
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        lookback: Period for moving average and std calculation. Defaults to 20.
        num_std: Number of standard deviations for bands. Defaults to 2.0.

    Returns:
        DataFrame with new columns:
        - bb_middle: Middle band (SMA)
        - bb_upper: Upper band
        - bb_lower: Lower band
        - bb_percent: Price position within bands (0-100)
        - bb_signal: 1 (at lower band), -1 (at upper band), 0 (middle)

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 98, 95, 105, 110, 108]})
        >>> result = bollinger_band_signal(df, lookback=3)
    """
    # Calculate middle band (SMA)
    df = df.with_columns(
        pl.col(price_col).rolling_mean(window_size=lookback).alias("bb_middle")
    )

    # Calculate rolling standard deviation
    df = df.with_columns(
        pl.col(price_col).rolling_std(window_size=lookback).alias("_std")
    )

    # Calculate upper and lower bands
    df = df.with_columns([
        (pl.col("bb_middle") + num_std * pl.col("_std")).alias("bb_upper"),
        (pl.col("bb_middle") - num_std * pl.col("_std")).alias("bb_lower"),
    ])

    # Calculate percent B (position within bands)
    df = df.with_columns(
        pl.when((pl.col("bb_upper") - pl.col("bb_lower")) > 0)
        .then(
            ((pl.col(price_col) - pl.col("bb_lower")) /
             (pl.col("bb_upper") - pl.col("bb_lower")) * 100)
        )
        .otherwise(50.0)
        .alias("bb_percent")
    )

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("bb_percent") < 5)  # Near lower band
        .then(1)
        .when(pl.col("bb_percent") > 95)  # Near upper band
        .then(-1)
        .otherwise(0)
        .alias("bb_signal")
    )

    return df.drop("_std")


def roc_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    lookback: int = 12,
    threshold: float = 5.0,
) -> pl.DataFrame:
    """Generate Rate of Change (ROC) momentum signals.

    ROC measures the percentage change in price over a lookback period.
    Positive ROC indicates upward momentum, negative indicates downward.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        lookback: Period for ROC calculation. Defaults to 12.
        threshold: ROC threshold in percent for signal generation. Defaults to 5.0.

    Returns:
        DataFrame with new columns:
        - roc: Rate of change (percentage)
        - roc_signal: 1 (strong upward momentum), -1 (strong downward), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({"close": [100, 105, 110, 108, 115, 120]})
        >>> result = roc_signal(df, lookback=3, threshold=3.0)
    """
    df = df.with_columns(
        (
            (pl.col(price_col) - pl.col(price_col).shift(lookback))
            / pl.col(price_col).shift(lookback)
            * 100
        ).alias("roc")
    )

    return df.with_columns(
        pl.when(pl.col("roc") > threshold)
        .then(1)
        .when(pl.col("roc") < -threshold)
        .then(-1)
        .otherwise(0)
        .alias("roc_signal")
    )


def stochastic_signal(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> pl.DataFrame:
    """Generate Stochastic Oscillator signals.

    Calculates %K and %D lines and generates signals based on overbought/oversold
    levels and crossovers.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        k_period: Period for %K calculation. Defaults to 14.
        d_period: Period for %D smoothing. Defaults to 3.
        oversold: Oversold threshold. Defaults to 20.
        overbought: Overbought threshold. Defaults to 80.

    Returns:
        DataFrame with new columns:
        - stoch_k: Fast stochastic (%K)
        - stoch_d: Slow stochastic (%D, smoothed %K)
        - stoch_signal: 1 (oversold or bullish cross), -1 (overbought or bearish cross), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 110, 108, 112, 115],
        ...     "low": [100, 105, 103, 107, 110],
        ...     "close": [102, 108, 106, 110, 113],
        ... })
        >>> result = stochastic_signal(df, k_period=3, d_period=2)
    """
    # Calculate %K
    df = df.with_columns([
        pl.col(high_col).rolling_max(window_size=k_period).alias("_highest_high"),
        pl.col(low_col).rolling_min(window_size=k_period).alias("_lowest_low"),
    ])

    df = df.with_columns(
        pl.when((pl.col("_highest_high") - pl.col("_lowest_low")) > 0)
        .then(
            (pl.col(close_col) - pl.col("_lowest_low"))
            / (pl.col("_highest_high") - pl.col("_lowest_low"))
            * 100
        )
        .otherwise(50.0)
        .alias("stoch_k")
    )

    # Calculate %D (smoothed %K)
    df = df.with_columns(
        pl.col("stoch_k").rolling_mean(window_size=d_period).alias("stoch_d")
    )

    # Generate signals based on levels and crossovers
    df = df.with_columns([
        pl.col("stoch_k").shift(1).alias("_prev_k"),
        pl.col("stoch_d").shift(1).alias("_prev_d"),
    ])

    df = df.with_columns(
        pl.when(
            (pl.col("stoch_k") < oversold)
            | ((pl.col("stoch_k") > pl.col("stoch_d")) & (pl.col("_prev_k") <= pl.col("_prev_d")))
        )
        .then(1)  # Oversold or bullish crossover
        .when(
            (pl.col("stoch_k") > overbought)
            | ((pl.col("stoch_k") < pl.col("stoch_d")) & (pl.col("_prev_k") >= pl.col("_prev_d")))
        )
        .then(-1)  # Overbought or bearish crossover
        .otherwise(0)
        .alias("stoch_signal")
    )

    return df.drop(["_highest_high", "_lowest_low", "_prev_k", "_prev_d"])


def adx_signal(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
    adx_threshold: float = 25.0,
) -> pl.DataFrame:
    """Generate Average Directional Index (ADX) trend strength signals.

    ADX measures trend strength (not direction). High ADX indicates strong trend,
    low ADX indicates weak trend or ranging market.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        period: Period for ADX calculation. Defaults to 14.
        adx_threshold: ADX level above which trend is considered strong. Defaults to 25.

    Returns:
        DataFrame with new columns:
        - adx: Average Directional Index (0-100)
        - di_plus: Positive Directional Indicator
        - di_minus: Negative Directional Indicator
        - adx_signal: 1 (strong uptrend), -1 (strong downtrend), 0 (weak/no trend)

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 110, 108, 112, 115, 118],
        ...     "low": [100, 105, 103, 107, 110, 113],
        ...     "close": [102, 108, 106, 110, 113, 116],
        ... })
        >>> result = adx_signal(df, period=3)
    """
    # Calculate True Range and Directional Movement
    df = df.with_columns([
        (pl.col(high_col) - pl.col(low_col)).alias("_tr1"),
        (pl.col(high_col) - pl.col(close_col).shift(1)).abs().alias("_tr2"),
        (pl.col(low_col) - pl.col(close_col).shift(1)).abs().alias("_tr3"),
        (pl.col(high_col) - pl.col(high_col).shift(1)).alias("_plus_dm_raw"),
        (pl.col(low_col).shift(1) - pl.col(low_col)).alias("_minus_dm_raw"),
    ])

    # True Range (max of three values)
    df = df.with_columns(
        pl.max_horizontal("_tr1", "_tr2", "_tr3").alias("_tr")
    )

    # Directional Movement
    df = df.with_columns([
        pl.when((pl.col("_plus_dm_raw") > pl.col("_minus_dm_raw")) & (pl.col("_plus_dm_raw") > 0))
        .then(pl.col("_plus_dm_raw"))
        .otherwise(0)
        .alias("_plus_dm"),
        pl.when((pl.col("_minus_dm_raw") > pl.col("_plus_dm_raw")) & (pl.col("_minus_dm_raw") > 0))
        .then(pl.col("_minus_dm_raw"))
        .otherwise(0)
        .alias("_minus_dm"),
    ])

    # Smoothed TR and DM
    df = df.with_columns([
        pl.col("_tr").ewm_mean(span=period).alias("_atr"),
        pl.col("_plus_dm").ewm_mean(span=period).alias("_smoothed_plus_dm"),
        pl.col("_minus_dm").ewm_mean(span=period).alias("_smoothed_minus_dm"),
    ])

    # Directional Indicators
    df = df.with_columns([
        pl.when(pl.col("_atr") > 0)
        .then((pl.col("_smoothed_plus_dm") / pl.col("_atr")) * 100)
        .otherwise(0)
        .alias("di_plus"),
        pl.when(pl.col("_atr") > 0)
        .then((pl.col("_smoothed_minus_dm") / pl.col("_atr")) * 100)
        .otherwise(0)
        .alias("di_minus"),
    ])

    # Directional Index
    df = df.with_columns(
        pl.when((pl.col("di_plus") + pl.col("di_minus")) > 0)
        .then(
            ((pl.col("di_plus") - pl.col("di_minus")).abs()
             / (pl.col("di_plus") + pl.col("di_minus")))
            * 100
        )
        .otherwise(0)
        .alias("_dx")
    )

    # ADX (smoothed DX)
    df = df.with_columns(
        pl.col("_dx").ewm_mean(span=period).alias("adx")
    )

    # Generate signals
    df = df.with_columns(
        pl.when((pl.col("adx") > adx_threshold) & (pl.col("di_plus") > pl.col("di_minus")))
        .then(1)  # Strong uptrend
        .when((pl.col("adx") > adx_threshold) & (pl.col("di_minus") > pl.col("di_plus")))
        .then(-1)  # Strong downtrend
        .otherwise(0)  # Weak trend or ranging
        .alias("adx_signal")
    )

    return df.drop([
        "_tr1", "_tr2", "_tr3", "_plus_dm_raw", "_minus_dm_raw",
        "_tr", "_plus_dm", "_minus_dm", "_atr",
        "_smoothed_plus_dm", "_smoothed_minus_dm", "_dx"
    ])
