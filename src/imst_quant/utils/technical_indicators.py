"""Technical indicators for quantitative trading analysis.

This module provides common technical analysis indicators implemented
with Polars for efficient computation on large datasets.

Indicators included:
    - MACD: Moving Average Convergence Divergence
    - Bollinger Bands: Volatility bands around moving average
    - ATR: Average True Range (volatility measure)
    - ADX: Average Directional Index (trend strength)
    - Stochastic Oscillator: Momentum oscillator
    - OBV: On Balance Volume

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.technical_indicators import macd, bollinger_bands
    >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 110]})
    >>> df = macd(df, fast=3, slow=5, signal=2)
    >>> df = bollinger_bands(df, window=3, num_std=2.0)
"""

from typing import Union

import numpy as np
import polars as pl


def _safe_ratio(
    numerator: pl.Expr,
    denominator: pl.Expr,
    fallback: float,
) -> pl.Expr:
    """Divide two expressions, substituting ``fallback`` on a zero denominator.

    Flat bars (identical high/low/close, or zero rolling standard deviation)
    make several indicator denominators exactly zero, which otherwise yields
    ``inf`` or ``NaN`` and silently poisons every downstream feature. Nulls in
    the denominator (rolling-window warmup) propagate as nulls, unchanged.

    Args:
        numerator: Expression for the dividend.
        denominator: Expression for the divisor.
        fallback: Value to use where the denominator is exactly zero.

    Returns:
        Expression evaluating to the ratio, or ``fallback`` where undefined.
    """
    return (
        pl.when(denominator == 0)
        .then(pl.lit(fallback, dtype=pl.Float64))
        .otherwise(numerator / denominator)
    )


def macd(
    df: pl.DataFrame,
    price_col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD).

    MACD is a trend-following momentum indicator showing the relationship
    between two exponential moving averages (EMAs) of prices.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        fast: Period for fast EMA. Defaults to 12.
        slow: Period for slow EMA. Defaults to 26.
        signal: Period for signal line EMA. Defaults to 9.

    Returns:
        DataFrame with new columns:
        - macd_line: Fast EMA minus slow EMA
        - macd_signal: Signal line (EMA of MACD line)
        - macd_histogram: MACD line minus signal line

    Example:
        >>> df = pl.DataFrame({"close": list(range(100, 150))})
        >>> result = macd(df, fast=12, slow=26, signal=9)
        >>> "macd_line" in result.columns
        True
    """
    # Calculate EMAs
    df = df.with_columns([
        pl.col(price_col).ewm_mean(span=fast, adjust=False).alias("_ema_fast"),
        pl.col(price_col).ewm_mean(span=slow, adjust=False).alias("_ema_slow"),
    ])

    # MACD line = fast EMA - slow EMA
    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
    )

    # Signal line = EMA of MACD line
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=signal, adjust=False).alias("macd_signal")
    )

    # Histogram = MACD line - signal line
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
    )

    # Clean up temporary columns
    return df.drop(["_ema_fast", "_ema_slow"])


def bollinger_bands(
    df: pl.DataFrame,
    price_col: str = "close",
    window: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Calculate Bollinger Bands volatility indicator.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        window: Period for moving average and standard deviation.
            Defaults to 20.
        num_std: Number of standard deviations for bands. Defaults to 2.0.

    Returns:
        DataFrame with new columns:
        - bb_middle: Simple moving average (middle band)
        - bb_upper: Upper band (middle + num_std * std)
        - bb_lower: Lower band (middle - num_std * std)
        - bb_width: Band width as percentage of middle
        - bb_percent: Price position within bands (0-1 scale)

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 110, 109]})
        >>> result = bollinger_bands(df, window=3)
        >>> "bb_upper" in result.columns
        True
    """
    # Calculate middle band (SMA)
    df = df.with_columns(
        pl.col(price_col).rolling_mean(window_size=window).alias("bb_middle")
    )

    # Calculate standard deviation
    df = df.with_columns(
        pl.col(price_col).rolling_std(window_size=window).alias("_bb_std")
    )

    # Calculate upper and lower bands
    df = df.with_columns([
        (pl.col("bb_middle") + num_std * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - num_std * pl.col("_bb_std")).alias("bb_lower"),
    ])

    # Calculate band width as percentage
    df = df.with_columns(
        _safe_ratio(
            pl.col("bb_upper") - pl.col("bb_lower"),
            pl.col("bb_middle"),
            0.0,
        ).alias("bb_width")
    )

    # Calculate %B (where price is relative to bands)
    # Zero-width bands (flat prices) put price exactly at the midpoint.
    df = df.with_columns(
        _safe_ratio(
            pl.col(price_col) - pl.col("bb_lower"),
            pl.col("bb_upper") - pl.col("bb_lower"),
            0.5,
        ).alias("bb_percent")
    )

    return df.drop(["_bb_std"])


def atr(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14,
) -> pl.DataFrame:
    """Calculate Average True Range (ATR) volatility indicator.

    ATR measures market volatility by taking the average of true ranges
    over a specified period. True range considers gaps between periods.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        window: Period for ATR averaging. Defaults to 14.

    Returns:
        DataFrame with new columns:
        - true_range: Current period's true range
        - atr: Average true range over window

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107, 110],
        ...     "low": [100, 102, 103, 106],
        ...     "close": [103, 106, 105, 109]
        ... })
        >>> result = atr(df, window=3)
    """
    # Calculate true range components
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    df = df.with_columns([
        (pl.col(high_col) - pl.col(low_col)).alias("_hl"),
        (pl.col(high_col) - pl.col(close_col).shift(1)).abs().alias("_hpc"),
        (pl.col(low_col) - pl.col(close_col).shift(1)).abs().alias("_lpc"),
    ])

    # True range is max of the three components
    df = df.with_columns(
        pl.max_horizontal(["_hl", "_hpc", "_lpc"]).alias("true_range")
    )

    # Wilder smoothing (alpha = 1/window), as in rsi(). An EMA with
    # span=window (alpha = 2/(window+1)) nearly doubles the reaction speed
    # and is not the ATR that stop/ADX thresholds are calibrated on.
    df = df.with_columns(
        pl.col("true_range").ewm_mean(alpha=1.0 / window, adjust=False).alias("atr")
    )

    return df.drop(["_hl", "_hpc", "_lpc"])


def adx(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14,
) -> pl.DataFrame:
    """Calculate Average Directional Index (ADX) trend strength indicator.

    ADX quantifies trend strength regardless of direction. Values above 25
    indicate a strong trend; below 20 indicates a weak or ranging market.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        window: Period for smoothing. Defaults to 14.

    Returns:
        DataFrame with new columns:
        - plus_di: Positive directional indicator
        - minus_di: Negative directional indicator
        - adx: Average directional index

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107, 110, 112],
        ...     "low": [100, 102, 103, 106, 108],
        ...     "close": [103, 106, 105, 109, 111]
        ... })
        >>> result = adx(df, window=3)
    """
    # First ensure we have ATR
    if "atr" not in df.columns:
        df = atr(df, high_col, low_col, close_col, window)

    # Calculate directional movement
    df = df.with_columns([
        (pl.col(high_col) - pl.col(high_col).shift(1)).alias("_up_move"),
        (pl.col(low_col).shift(1) - pl.col(low_col)).alias("_down_move"),
    ])

    # +DM and -DM
    df = df.with_columns([
        pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0))
        .then(pl.col("_up_move"))
        .otherwise(0)
        .alias("_plus_dm"),
        pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0))
        .then(pl.col("_down_move"))
        .otherwise(0)
        .alias("_minus_dm"),
    ])

    # Smooth DM values (Wilder, matching atr)
    df = df.with_columns([
        pl.col("_plus_dm").ewm_mean(alpha=1.0 / window, adjust=False).alias("_plus_dm_smooth"),
        pl.col("_minus_dm").ewm_mean(alpha=1.0 / window, adjust=False).alias("_minus_dm_smooth"),
    ])

    # Calculate +DI and -DI
    # A zero ATR means no range at all, so there is no directional movement.
    df = df.with_columns([
        _safe_ratio(100 * pl.col("_plus_dm_smooth"), pl.col("atr"), 0.0).alias("plus_di"),
        _safe_ratio(100 * pl.col("_minus_dm_smooth"), pl.col("atr"), 0.0).alias("minus_di"),
    ])

    # Calculate DX
    df = df.with_columns(
        _safe_ratio(
            100 * (pl.col("plus_di") - pl.col("minus_di")).abs(),
            pl.col("plus_di") + pl.col("minus_di"),
            0.0,
        ).alias("_dx")
    )

    # ADX is smoothed DX
    df = df.with_columns(
        pl.col("_dx").ewm_mean(alpha=1.0 / window, adjust=False).alias("adx")
    )

    return df.drop([
        "_up_move", "_down_move", "_plus_dm", "_minus_dm",
        "_plus_dm_smooth", "_minus_dm_smooth", "_dx"
    ])


def stochastic_oscillator(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k_period: int = 14,
    d_period: int = 3,
) -> pl.DataFrame:
    """Calculate Stochastic Oscillator momentum indicator.

    The stochastic oscillator compares closing price to the price range
    over a period. Values above 80 suggest overbought; below 20 oversold.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        k_period: Lookback period for %K. Defaults to 14.
        d_period: Smoothing period for %D. Defaults to 3.

    Returns:
        DataFrame with new columns:
        - stoch_k: Fast stochastic (%K)
        - stoch_d: Slow stochastic (%D, smoothed %K)

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107, 110, 112],
        ...     "low": [100, 102, 103, 106, 108],
        ...     "close": [103, 106, 105, 109, 111]
        ... })
        >>> result = stochastic_oscillator(df, k_period=3)
    """
    # Calculate highest high and lowest low over period
    df = df.with_columns([
        pl.col(high_col).rolling_max(window_size=k_period).alias("_highest"),
        pl.col(low_col).rolling_min(window_size=k_period).alias("_lowest"),
    ])

    # Calculate %K
    # A zero-width range (flat bars) is neither overbought nor oversold.
    df = df.with_columns(
        _safe_ratio(
            100 * (pl.col(close_col) - pl.col("_lowest")),
            pl.col("_highest") - pl.col("_lowest"),
            50.0,
        ).alias("stoch_k")
    )

    # %D is smoothed %K
    df = df.with_columns(
        pl.col("stoch_k").rolling_mean(window_size=d_period).alias("stoch_d")
    )

    return df.drop(["_highest", "_lowest"])


def obv(
    df: pl.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Calculate On Balance Volume (OBV) indicator.

    OBV uses volume flow to predict changes in stock price. Rising OBV
    indicates buying pressure; falling OBV indicates selling pressure.

    Args:
        df: DataFrame containing price and volume data.
        close_col: Name of the close price column. Defaults to "close".
        volume_col: Name of the volume column. Defaults to "volume".

    Returns:
        DataFrame with new column:
        - obv: Cumulative on balance volume

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 101, 105, 108],
        ...     "volume": [1000, 1500, 1200, 1800, 2000]
        ... })
        >>> result = obv(df)
    """
    # Determine direction: +volume if close > prev close, -volume if lower
    # Bars with a null/NaN/inf close or volume add nothing, and the next bar
    # compares against the last good close. Otherwise one NaN volume turns
    # every later OBV value into NaN, and polars ranks a NaN close above any
    # price, booking a phantom up-bar then a phantom down-bar.
    valid = (pl.col(close_col).is_finite() & pl.col(volume_col).is_finite()).fill_null(False)
    prev = pl.when(valid).then(pl.col(close_col)).forward_fill().shift(1)
    df = df.with_columns(
        pl.when(valid & (pl.col(close_col) > prev))
        .then(pl.col(volume_col))
        .when(valid & (pl.col(close_col) < prev))
        .then(-pl.col(volume_col))
        .otherwise(0)
        .alias("_obv_change")
    )

    # Cumulative sum
    df = df.with_columns(
        pl.col("_obv_change").cum_sum().alias("obv")
    )

    return df.drop(["_obv_change"])


def vwap(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Calculate Volume Weighted Average Price (VWAP).

    VWAP gives the average price weighted by volume. It's commonly used
    as a benchmark for intraday trading quality.

    Args:
        df: DataFrame containing OHLCV data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        volume_col: Name of the volume column. Defaults to "volume".

    Returns:
        DataFrame with new column:
        - vwap: Volume weighted average price

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107],
        ...     "low": [100, 102, 103],
        ...     "close": [103, 106, 105],
        ...     "volume": [1000, 1500, 1200]
        ... })
        >>> result = vwap(df)
    """
    # Typical price = (high + low + close) / 3
    df = df.with_columns(
        ((pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3).alias("_typical")
    )

    # Cumulative sums for VWAP calculation
    # Skip bars with a null/NaN/inf price or volume: a single NaN would
    # otherwise poison both running sums for the rest of the series.
    valid = (pl.col("_typical").is_finite() & pl.col(volume_col).is_finite()).fill_null(False)
    df = df.with_columns([
        pl.when(valid).then(pl.col("_typical") * pl.col(volume_col)).otherwise(0.0)
        .cum_sum().alias("_tpv_sum"),
        pl.when(valid).then(pl.col(volume_col)).otherwise(0.0).cum_sum().alias("_vol_sum"),
    ])

    # VWAP = cumulative (typical * volume) / cumulative volume; undefined
    # (null) until some volume has traded.
    df = df.with_columns(
        pl.when(pl.col("_vol_sum") > 0)
        .then(pl.col("_tpv_sum") / pl.col("_vol_sum"))
        .alias("vwap")
    )

    return df.drop(["_typical", "_tpv_sum", "_vol_sum"])


def williams_r(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Calculate Williams %R momentum indicator.

    Williams %R measures overbought/oversold levels on a -100 to 0 scale.
    Values above -20 indicate overbought; below -80 indicate oversold.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        period: Lookback period. Defaults to 14.

    Returns:
        DataFrame with new column:
        - williams_r: Williams %R value (-100 to 0)

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107, 110, 112],
        ...     "low": [100, 102, 103, 106, 108],
        ...     "close": [103, 106, 105, 109, 111]
        ... })
        >>> result = williams_r(df, period=3)
    """
    # Calculate highest high and lowest low
    df = df.with_columns([
        pl.col(high_col).rolling_max(window_size=period).alias("_hh"),
        pl.col(low_col).rolling_min(window_size=period).alias("_ll"),
    ])

    # Williams %R = (highest - close) / (highest - lowest) * -100
    # A zero-width range maps to the midpoint of the -100..0 scale.
    df = df.with_columns(
        _safe_ratio(
            -100 * (pl.col("_hh") - pl.col(close_col)),
            pl.col("_hh") - pl.col("_ll"),
            -50.0,
        ).alias("williams_r")
    )

    return df.drop(["_hh", "_ll"])


def cci(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 20,
) -> pl.DataFrame:
    """Calculate Commodity Channel Index (CCI).

    CCI measures the current price level relative to an average price level
    over a given period. Values above +100 suggest overbought; below -100
    suggest oversold.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of the high price column. Defaults to "high".
        low_col: Name of the low price column. Defaults to "low".
        close_col: Name of the close price column. Defaults to "close".
        period: Lookback period. Defaults to 20.

    Returns:
        DataFrame with new column:
        - cci: Commodity Channel Index

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 107, 110, 112],
        ...     "low": [100, 102, 103, 106, 108],
        ...     "close": [103, 106, 105, 109, 111]
        ... })
        >>> result = cci(df, period=3)
    """
    # Typical price
    df = df.with_columns(
        ((pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3).alias("_tp")
    )

    # SMA of typical price
    df = df.with_columns(
        pl.col("_tp").rolling_mean(window_size=period).alias("_tp_sma")
    )

    # Mean absolute deviation of typical price around its window mean.
    # Standard deviation is not a substitute: for normal data it runs ~1.25x
    # the MAD, shrinking CCI so the +/-100 thresholds fire far less often.
    df = df.with_columns(
        pl.col("_tp")
        .rolling_map(lambda w: (w - w.mean()).abs().mean(), window_size=period)
        .alias("_mad")
    )

    # CCI = (TP - SMA) / (0.015 * MAD)
    # Using 0.015 as Lambert's constant. Zero deviation means price sits on
    # its own average, i.e. no deviation from the mean at all.
    df = df.with_columns(
        _safe_ratio(
            pl.col("_tp") - pl.col("_tp_sma"),
            0.015 * pl.col("_mad"),
            0.0,
        ).alias("cci")
    )

    return df.drop(["_tp", "_tp_sma", "_mad"])


def rsi(
    df: pl.DataFrame,
    price_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Calculate Relative Strength Index (RSI).

    RSI is a momentum oscillator measuring the speed and magnitude of price changes.
    Values range from 0 to 100, with readings above 70 indicating overbought conditions
    and below 30 indicating oversold.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column. Defaults to "close".
        period: Lookback period. Defaults to 14.

    Returns:
        DataFrame with new column:
        - rsi: Relative Strength Index

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 110, 107, 109]})
        >>> result = rsi(df, period=6)
    """
    # Calculate price changes
    df = df.with_columns(
        pl.col(price_col).diff().alias("_price_change")
    )

    # Separate gains and losses
    df = df.with_columns([
        pl.when(pl.col("_price_change") > 0)
          .then(pl.col("_price_change"))
          .otherwise(0)
          .alias("_gain"),
        pl.when(pl.col("_price_change") < 0)
          .then(pl.col("_price_change").abs())
          .otherwise(0)
          .alias("_loss")
    ])

    # Calculate average gain and loss using EMA
    alpha = 1.0 / period
    df = df.with_columns([
        pl.col("_gain").ewm_mean(alpha=alpha, adjust=False).alias("_avg_gain"),
        pl.col("_loss").ewm_mean(alpha=alpha, adjust=False).alias("_avg_loss")
    ])

    # RSI = 100 - 100 / (1 + avg_gain / avg_loss), rewritten as
    # 100 * avg_gain / (avg_gain + avg_loss) so a zero average loss gives 100
    # instead of dividing by zero. Flat prices (no gains and no losses) are
    # neutral at 50.
    df = df.with_columns(
        _safe_ratio(
            100 * pl.col("_avg_gain"),
            pl.col("_avg_gain") + pl.col("_avg_loss"),
            50.0,
        ).alias("rsi")
    )

    return df.drop(["_price_change", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def detect_rsi_divergence(
    df: pl.DataFrame,
    price_col: str = "close",
    rsi_col: str = "rsi",
    lookback: int = 14,
    threshold: float = 0.02,
) -> pl.DataFrame:
    """Detect bullish and bearish RSI divergences.

    Divergence occurs when price and RSI move in opposite directions, often
    signaling potential reversals.

    Bullish divergence: Price makes lower low, but RSI makes higher low
    Bearish divergence: Price makes higher high, but RSI makes lower high

    Args:
        df: DataFrame containing price and RSI data.
        price_col: Name of the price column. Defaults to "close".
        rsi_col: Name of the RSI column. Defaults to "rsi".
        lookback: Period to look back for highs/lows. Defaults to 14.
        threshold: Minimum percentage change to consider a divergence. Defaults to 0.02 (2%).

    Returns:
        DataFrame with new columns:
        - bullish_divergence: 1 if bullish divergence detected, 0 otherwise
        - bearish_divergence: 1 if bearish divergence detected, 0 otherwise

    Example:
        >>> df = pl.DataFrame({"close": [100.0 + (i % 7) + 0.5 * i for i in range(40)]})
        >>> df = rsi(df, period=14)
        >>> df = detect_rsi_divergence(df, lookback=10)
        >>> {"bullish_divergence", "bearish_divergence"} <= set(df.columns)
        True
    """
    # Calculate rolling min/max for price and RSI
    df = df.with_columns([
        pl.col(price_col).rolling_min(window_size=lookback).alias("_price_min"),
        pl.col(price_col).rolling_max(window_size=lookback).alias("_price_max"),
        pl.col(rsi_col).rolling_min(window_size=lookback).alias("_rsi_min"),
        pl.col(rsi_col).rolling_max(window_size=lookback).alias("_rsi_max"),
    ])

    # Detect price making new lows/highs
    df = df.with_columns([
        (pl.col(price_col) == pl.col("_price_min")).alias("_price_at_min"),
        (pl.col(price_col) == pl.col("_price_max")).alias("_price_at_max"),
    ])

    # Calculate percentage changes from previous extremes
    df = df.with_columns([
        (pl.col(price_col) / pl.col("_price_min").shift(lookback) - 1).alias("_price_pct_from_min"),
        (pl.col(rsi_col) / pl.col("_rsi_min").shift(lookback) - 1).alias("_rsi_pct_from_min"),
        (pl.col(price_col) / pl.col("_price_max").shift(lookback) - 1).alias("_price_pct_from_max"),
        (pl.col(rsi_col) / pl.col("_rsi_max").shift(lookback) - 1).alias("_rsi_pct_from_max"),
    ])

    # Bullish divergence: price lower, RSI higher (at price minimum)
    df = df.with_columns(
        pl.when(
            pl.col("_price_at_min") &
            (pl.col("_price_pct_from_min") < -threshold) &
            (pl.col("_rsi_pct_from_min") > 0)
        )
        .then(1)
        .otherwise(0)
        .alias("bullish_divergence")
    )

    # Bearish divergence: price higher, RSI lower (at price maximum)
    df = df.with_columns(
        pl.when(
            pl.col("_price_at_max") &
            (pl.col("_price_pct_from_max") > threshold) &
            (pl.col("_rsi_pct_from_max") < 0)
        )
        .then(1)
        .otherwise(0)
        .alias("bearish_divergence")
    )

    return df.drop([
        "_price_min", "_price_max", "_rsi_min", "_rsi_max",
        "_price_at_min", "_price_at_max",
        "_price_pct_from_min", "_rsi_pct_from_min",
        "_price_pct_from_max", "_rsi_pct_from_max"
    ])


def supertrend(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 10,
    multiplier: float = 3.0,
) -> pl.DataFrame:
    """Supertrend: ATR band around the bar midpoint that ratchets with the trend.

    Basic bands are ``(high + low) / 2 -/+ multiplier * ATR(period)`` (Wilder
    ATR). The lower band may only rise while the prior close holds above it,
    and the upper band may only fall while the prior close holds below it.
    The trend flips up when the close breaks above the upper band and down
    when it breaks below the lower band. The first ``period - 1`` rows are
    null; the trend starts up on the first full bar.

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col, close_col: Column names.
        period: ATR look-back (>= 1).
        multiplier: ATR multiple (> 0).

    Returns:
        DataFrame with added ``atr``, ``supertrend`` (the active band) and
        ``st_direction`` (1 uptrend, -1 downtrend) columns.

    Example:
        >>> df = pl.DataFrame({"high": [11.0, 12, 13], "low": [9.0, 10, 11],
        ...                    "close": [10.0, 11, 12]})
        >>> supertrend(df, period=2, multiplier=1.0)["st_direction"].to_list()
        [None, 1, 1]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if not multiplier > 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    out = atr(df, high_col=high_col, low_col=low_col, close_col=close_col, window=period)
    out = out.drop("true_range")
    hl2 = ((out[high_col] + out[low_col]) / 2).to_list()
    atr_v = out["atr"].to_list()
    close = out[close_col].to_list()

    n = out.height
    line: list = [None] * n
    direction: list = [None] * n
    upper = lower = None
    for i in range(period - 1, n):
        bu = hl2[i] + multiplier * atr_v[i]
        bl = hl2[i] - multiplier * atr_v[i]
        if upper is None:
            upper, lower, d = bu, bl, 1
        else:
            prev_close = close[i - 1]
            upper = bu if (bu < upper or prev_close > upper) else upper
            lower = bl if (bl > lower or prev_close < lower) else lower
            d = direction[i - 1]
            if d == 1 and close[i] < lower:
                d = -1
            elif d == -1 and close[i] > upper:
                d = 1
        direction[i] = d
        line[i] = lower if d == 1 else upper

    return out.with_columns(
        pl.Series("supertrend", line, dtype=pl.Float64),
        pl.Series("st_direction", direction, dtype=pl.Int8),
    )


def keltner_channels(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 20,
    multiplier: float = 1.5,
) -> pl.DataFrame:
    """Keltner channels: EMA(close) -/+ multiplier * Wilder ATR.

    The first ``window - 1`` rows are null so the EMA and ATR have warmed up.

    Args:
        df: OHLC DataFrame, oldest first.
        high_col, low_col, close_col: Column names.
        window: EMA span and ATR period (>= 1).
        multiplier: ATR multiple (> 0).

    Returns:
        DataFrame with added ``kc_middle``, ``kc_upper`` and ``kc_lower``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not multiplier > 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    out = atr(df, high_col=high_col, low_col=low_col, close_col=close_col, window=window)
    full = pl.int_range(pl.len()) >= window - 1
    mid = pl.col(close_col).ewm_mean(span=window, adjust=False)
    out = out.with_columns(pl.when(full).then(mid).alias("kc_middle"))
    return out.with_columns(
        (pl.col("kc_middle") + multiplier * pl.col("atr")).alias("kc_upper"),
        (pl.col("kc_middle") - multiplier * pl.col("atr")).alias("kc_lower"),
    ).drop(["true_range", "atr"])


def bollinger_squeeze(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 20,
    bb_std: float = 2.0,
    kc_mult: float = 1.5,
) -> pl.DataFrame:
    """Volatility squeeze: Bollinger Bands fully inside the Keltner channel.

    A squeeze marks unusually compressed volatility; the bar where it ends
    ("fires") often precedes a directional move. Direction is read from the
    close versus the Keltner midline.

    Args:
        df: OHLC DataFrame, oldest first.
        high_col, low_col, close_col: Column names.
        window: Shared look-back for both band sets (>= 2).
        bb_std: Bollinger standard-deviation multiple (> 0).
        kc_mult: Keltner ATR multiple (> 0).

    Returns:
        DataFrame with Bollinger and Keltner columns plus ``squeeze_on``
        (bool, null during warm-up) and ``squeeze_fired`` (True on the first
        bar after a squeeze ends).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if not bb_std > 0:
        raise ValueError(f"bb_std must be > 0, got {bb_std}")

    out = bollinger_bands(df, price_col=close_col, window=window, num_std=bb_std)
    out = keltner_channels(
        out, high_col=high_col, low_col=low_col, close_col=close_col,
        window=window, multiplier=kc_mult,
    )
    on = (pl.col("bb_upper") < pl.col("kc_upper")) & (pl.col("bb_lower") > pl.col("kc_lower"))
    out = out.with_columns(on.alias("squeeze_on"))
    return out.with_columns(
        (pl.col("squeeze_on").shift(1) & ~pl.col("squeeze_on")).fill_null(False).alias("squeeze_fired")
    )


def parabolic_sar(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    step: float = 0.02,
    max_step: float = 0.2,
) -> pl.DataFrame:
    """Wilder's Parabolic SAR (stop and reverse).

    The SAR trails price and accelerates toward the extreme point (EP) by an
    acceleration factor that grows by ``step`` on each new extreme, capped at
    ``max_step``. It never moves inside the prior two bars' range; when price
    crosses it the trend reverses and the SAR jumps to the old EP. The trend
    starts up on the first bar (SAR = first low), whose row is null.

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col: Column names.
        step: Acceleration factor start and increment (> 0).
        max_step: Acceleration factor cap (>= step).

    Returns:
        DataFrame with added ``psar``, ``psar_direction`` (1 up, -1 down) and
        ``psar_af`` columns. The SAR on each row is the stop for that bar.

    Example:
        >>> df = pl.DataFrame({"high": [10.0, 11, 12], "low": [9.0, 10, 11]})
        >>> parabolic_sar(df)["psar_direction"].to_list()
        [None, 1, 1]
    """
    if not step > 0:
        raise ValueError(f"step must be > 0, got {step}")
    if not max_step >= step:
        raise ValueError(f"max_step must be >= step, got {max_step}")

    high = df[high_col].to_list()
    low = df[low_col].to_list()
    n = df.height
    sar_out: list = [None] * n
    dir_out: list = [None] * n
    af_out: list = [None] * n
    if n:
        up, sar, ep, af = True, low[0], high[0], step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                up, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                up, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar_out[i], dir_out[i], af_out[i] = sar, 1 if up else -1, af

    return df.with_columns(
        pl.Series("psar", sar_out, dtype=pl.Float64),
        pl.Series("psar_direction", dir_out, dtype=pl.Int8),
        pl.Series("psar_af", af_out, dtype=pl.Float64),
    )


def donchian_channels(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 20,
) -> pl.DataFrame:
    """Donchian channels: rolling highest high / lowest low over ``window`` bars.

    A breakout is a close beyond the *prior* bar's channel (the current bar is
    always inside its own channel, so comparing to it would never fire).

    Args:
        df: OHLC DataFrame, oldest first.
        high_col, low_col, close_col: Column names.
        window: Look-back bars (>= 1).

    Returns:
        DataFrame with added ``dc_upper``, ``dc_lower``, ``dc_middle`` (null
        during warm-up) and ``dc_breakout`` (1 above prior upper, -1 below
        prior lower, 0 otherwise; null until a prior channel exists).

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 3, 5], "low": [1.0, 2, 4], "close": [1.5, 2.5, 4.5]})
        >>> donchian_channels(df, window=2)["dc_breakout"].to_list()
        [None, None, 1]
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    out = df.with_columns(
        pl.col(high_col).rolling_max(window_size=window).alias("dc_upper"),
        pl.col(low_col).rolling_min(window_size=window).alias("dc_lower"),
    ).with_columns(((pl.col("dc_upper") + pl.col("dc_lower")) / 2).alias("dc_middle"))
    prev_up, prev_lo = pl.col("dc_upper").shift(1), pl.col("dc_lower").shift(1)
    return out.with_columns(
        pl.when(prev_up.is_null()).then(None)
        .when(pl.col(close_col) > prev_up).then(1)
        .when(pl.col(close_col) < prev_lo).then(-1)
        .otherwise(0).cast(pl.Int8).alias("dc_breakout")
    )


def aroon(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    period: int = 25,
) -> pl.DataFrame:
    """Aroon up/down: how recently the period's high and low were set.

    ``aroon_up = 100 * (period - bars_since_highest_high) / period`` over the
    last ``period + 1`` bars (likewise ``aroon_down`` with the low). Ties go to
    the most recent bar, so a flat market reads 100/100 (oscillator 0).

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col: Column names.
        period: Look-back bars (>= 1).

    Returns:
        DataFrame with added ``aroon_up``, ``aroon_down`` and
        ``aroon_osc`` (up - down, -100..100); the first ``period`` rows are null.

    Example:
        >>> df = pl.DataFrame({"high": [1.0, 2, 3], "low": [0.5, 1.5, 2.5]})
        >>> aroon(df, period=2)["aroon_up"].to_list()
        [None, None, 100.0]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    n = df.height
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    if n > period:
        win = period + 1
        hi = np.lib.stride_tricks.sliding_window_view(df[high_col].to_numpy().astype(float), win)
        lo = np.lib.stride_tricks.sliding_window_view(df[low_col].to_numpy().astype(float), win)
        # argmax on the reversed window = bars since the most recent extreme.
        up[period:] = 100.0 * (period - np.argmax(hi[:, ::-1], axis=1)) / period
        down[period:] = 100.0 * (period - np.argmin(lo[:, ::-1], axis=1)) / period
    return df.with_columns(
        pl.Series("aroon_up", up).fill_nan(None),
        pl.Series("aroon_down", down).fill_nan(None),
    ).with_columns((pl.col("aroon_up") - pl.col("aroon_down")).alias("aroon_osc"))


def vortex(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Vortex indicator (VI+ / VI-): directional movement normalised by true range.

    ``VM+ = |high - prior low|``, ``VM- = |low - prior high|``,
    ``TR = max(high - low, |high - prior close|, |low - prior close|)``;
    ``VI± = sum(VM±, period) / sum(TR, period)``. VI+ above VI- reads as an uptrend.

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col, close_col: Column names.
        period: Summation window (>= 1).

    Returns:
        DataFrame with added ``vi_plus``, ``vi_minus`` and ``vi_diff``
        (plus - minus); the first ``period`` rows are null (bar 0 has no prior
        bar). Zero total true range (flat window) gives null, not inf.

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 3, 4], "low": [1.0, 2, 3], "close": [1.5, 2.5, 3.5]})
        >>> vortex(df, period=2)["vi_minus"].to_list()
        [None, None, 0.0]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    h, lo, c = pl.col(high_col), pl.col(low_col), pl.col(close_col)
    tr = pl.max_horizontal(h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs())
    tr = pl.when(c.shift(1).is_null()).then(None).otherwise(tr)
    tr_sum = pl.when(tr.rolling_sum(window_size=period) > 0).then(tr.rolling_sum(window_size=period))
    out = df.with_columns(
        ((h - lo.shift(1)).abs().rolling_sum(window_size=period) / tr_sum).alias("vi_plus"),
        ((lo - h.shift(1)).abs().rolling_sum(window_size=period) / tr_sum).alias("vi_minus"),
    )
    return out.with_columns((pl.col("vi_plus") - pl.col("vi_minus")).alias("vi_diff"))


def money_flow_index(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    period: int = 14,
) -> pl.DataFrame:
    """Money Flow Index: a volume-weighted RSI on typical price.

    Typical price ``tp = (high + low + close) / 3``; raw flow ``tp * volume``
    counts as positive when ``tp`` rose versus the prior bar and negative when
    it fell (unchanged bars count as neither).
    ``MFI = 100 * pos_sum / (pos_sum + neg_sum)`` over ``period`` bars.

    Args:
        df: OHLCV DataFrame, oldest first, no null/NaN values.
        high_col, low_col, close_col, volume_col: Column names.
        period: Summation window (>= 1).

    Returns:
        DataFrame with added ``mfi`` (0..100); the first ``period`` rows are
        null. A window with no flow at all (flat price or zero volume) reads 50.

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 3, 4], "low": [1.0, 2, 3],
        ...                    "close": [1.5, 2.5, 3.5], "volume": [10.0, 10, 10]})
        >>> money_flow_index(df, period=2)["mfi"].to_list()
        [None, None, 100.0]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    tp = (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
    flow = tp * pl.col(volume_col)
    chg = tp - tp.shift(1)
    pos = pl.when(chg > 0).then(flow).otherwise(0.0).rolling_sum(window_size=period)
    neg = pl.when(chg < 0).then(flow).otherwise(0.0).rolling_sum(window_size=period)
    # Row 0 has no prior bar; mask it so the window starts at row `period`.
    valid = chg.is_not_null().cast(pl.Int32).rolling_sum(window_size=period) == period
    mfi = pl.when(~valid).then(None).when(pos + neg == 0).then(50.0).otherwise(100 * pos / (pos + neg))
    return df.with_columns(mfi.cast(pl.Float64).alias("mfi"))


def ichimoku(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    displacement: int = 26,
) -> pl.DataFrame:
    """Ichimoku Kinko Hyo lines, with the cloud aligned to the bar it applies to.

    Each line is a rolling ``(highest high + lowest low) / 2`` midpoint.
    ``lead_a = (tenkan + kijun) / 2`` and ``lead_b`` (the ``senkou`` midpoint)
    are plotted ``displacement`` bars ahead, so the cloud *at* bar t is the lead
    value from bar ``t - displacement``: no look-ahead. The chikou span is just
    the close shifted back, so it is left to callers (compare close to the close
    ``displacement`` bars ago).

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col: Column names.
        tenkan, kijun, senkou: Midpoint windows (>= 1).
        displacement: Bars the cloud is projected forward (>= 1).

    Returns:
        DataFrame with added ``ichi_tenkan``, ``ichi_kijun``, ``ichi_lead_a``,
        ``ichi_lead_b`` (the forward cloud computed at this bar) and
        ``ichi_senkou_a``, ``ichi_senkou_b`` (the cloud in force at this bar);
        null during warm-up.

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 4, 6, 8], "low": [1.0, 3, 5, 7]})
        >>> ichimoku(df, tenkan=1, kijun=2, senkou=3, displacement=1)["ichi_senkou_b"].to_list()
        [None, None, None, 3.5]
    """
    for name, val in (("tenkan", tenkan), ("kijun", kijun), ("senkou", senkou), ("displacement", displacement)):
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")

    def mid(w: int) -> pl.Expr:
        return (pl.col(high_col).rolling_max(window_size=w) + pl.col(low_col).rolling_min(window_size=w)) / 2

    out = df.with_columns(
        mid(tenkan).alias("ichi_tenkan"),
        mid(kijun).alias("ichi_kijun"),
        mid(senkou).alias("ichi_lead_b"),
    ).with_columns(((pl.col("ichi_tenkan") + pl.col("ichi_kijun")) / 2).alias("ichi_lead_a"))
    return out.with_columns(
        pl.col("ichi_lead_a").shift(displacement).alias("ichi_senkou_a"),
        pl.col("ichi_lead_b").shift(displacement).alias("ichi_senkou_b"),
    )


def chaikin_money_flow(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    period: int = 20,
) -> pl.DataFrame:
    """Chaikin Money Flow: volume-weighted location of the close in each bar's range.

    Money-flow multiplier ``mfm = ((close - low) - (high - close)) / (high - low)``
    (+1 = closed at the high, -1 = at the low, 0 for a zero-range bar);
    ``CMF = sum(mfm * volume, period) / sum(volume, period)``, in [-1, 1].

    Args:
        df: OHLCV DataFrame, oldest first, no null/NaN values.
        high_col, low_col, close_col, volume_col: Column names.
        period: Summation window (>= 1).

    Returns:
        DataFrame with added ``cmf``; the first ``period - 1`` rows are null,
        and a window with zero total volume is null rather than NaN.

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 2], "low": [1.0, 1],
        ...                    "close": [2.0, 1.0], "volume": [30.0, 10]})
        >>> chaikin_money_flow(df, period=2)["cmf"].to_list()
        [None, 0.5]
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    h, lo, c, v = pl.col(high_col), pl.col(low_col), pl.col(close_col), pl.col(volume_col)
    mfm = pl.when(h > lo).then(((c - lo) - (h - c)) / (h - lo)).otherwise(0.0)
    vol_sum = v.rolling_sum(window_size=period)
    cmf = pl.when(vol_sum > 0).then((mfm * v).rolling_sum(window_size=period) / vol_sum)
    return df.with_columns(cmf.cast(pl.Float64).alias("cmf"))


def choppiness_index(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Choppiness Index: how much of the window's travel went nowhere.

    ``CHOP = 100 * log10(sum(TR, period) / (max(high, period) - min(low, period))) / log10(period)``.
    Near 100 = sideways chop (lots of true range, little net range); near 0 =
    a clean trend. Common reads: > 61.8 choppy, < 38.2 trending. Bar 0's true
    range is ``high - low`` (no prior close).

    Args:
        df: OHLC DataFrame, oldest first, no null/NaN prices.
        high_col, low_col, close_col: Column names.
        period: Window (>= 2; log10(1) = 0).

    Returns:
        DataFrame with added ``chop``; the first ``period - 1`` rows are null,
        and a flat window (zero high-low range) is null rather than inf.

    Example:
        >>> df = pl.DataFrame({"high": [2.0, 3.0], "low": [1.0, 2.0], "close": [2.0, 3.0]})
        >>> chop = choppiness_index(df, period=2)["chop"].to_list()
        >>> chop[0] is None, round(chop[1], 6)
        (True, 0.0)
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    h, lo, pc = pl.col(high_col), pl.col(low_col), pl.col(close_col).shift(1)
    tr = pl.max_horizontal(h - lo, (h - pc).abs(), (lo - pc).abs())  # null pc -> h - lo
    rng = h.rolling_max(window_size=period) - lo.rolling_min(window_size=period)
    chop = pl.when(rng > 0).then(
        100 * (tr.rolling_sum(window_size=period) / rng).log10() / np.log10(period)
    )
    return df.with_columns(chop.cast(pl.Float64).alias("chop"))
