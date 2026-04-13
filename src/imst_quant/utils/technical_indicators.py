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
        ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias("bb_width")
    )

    # Calculate %B (where price is relative to bands)
    df = df.with_columns(
        ((pl.col(price_col) - pl.col("bb_lower")) /
         (pl.col("bb_upper") - pl.col("bb_lower"))).alias("bb_percent")
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

    # Average true range using EMA for smoothing
    df = df.with_columns(
        pl.col("true_range").ewm_mean(span=window, adjust=False).alias("atr")
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

    # Smooth DM values
    df = df.with_columns([
        pl.col("_plus_dm").ewm_mean(span=window, adjust=False).alias("_plus_dm_smooth"),
        pl.col("_minus_dm").ewm_mean(span=window, adjust=False).alias("_minus_dm_smooth"),
    ])

    # Calculate +DI and -DI
    df = df.with_columns([
        (100 * pl.col("_plus_dm_smooth") / pl.col("atr")).alias("plus_di"),
        (100 * pl.col("_minus_dm_smooth") / pl.col("atr")).alias("minus_di"),
    ])

    # Calculate DX
    df = df.with_columns(
        (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() /
         (pl.col("plus_di") + pl.col("minus_di"))).alias("_dx")
    )

    # ADX is smoothed DX
    df = df.with_columns(
        pl.col("_dx").ewm_mean(span=window, adjust=False).alias("adx")
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
    df = df.with_columns(
        (100 * (pl.col(close_col) - pl.col("_lowest")) /
         (pl.col("_highest") - pl.col("_lowest"))).alias("stoch_k")
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
    df = df.with_columns(
        pl.when(pl.col(close_col) > pl.col(close_col).shift(1))
        .then(pl.col(volume_col))
        .when(pl.col(close_col) < pl.col(close_col).shift(1))
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
    df = df.with_columns([
        (pl.col("_typical") * pl.col(volume_col)).cum_sum().alias("_tpv_sum"),
        pl.col(volume_col).cum_sum().alias("_vol_sum"),
    ])

    # VWAP = cumulative (typical * volume) / cumulative volume
    df = df.with_columns(
        (pl.col("_tpv_sum") / pl.col("_vol_sum")).alias("vwap")
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
    df = df.with_columns(
        ((pl.col("_hh") - pl.col(close_col)) /
         (pl.col("_hh") - pl.col("_ll")) * -100).alias("williams_r")
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

    # Mean deviation (approximation using std for efficiency)
    df = df.with_columns(
        pl.col("_tp").rolling_std(window_size=period).alias("_mad")
    )

    # CCI = (TP - SMA) / (0.015 * MAD)
    # Using 0.015 as Lambert's constant
    df = df.with_columns(
        ((pl.col("_tp") - pl.col("_tp_sma")) / (0.015 * pl.col("_mad"))).alias("cci")
    )

    return df.drop(["_tp", "_tp_sma", "_mad"])
