"""Advanced signal generation with MACD, Stochastic, and ADX indicators.

This module provides advanced technical indicators for signal generation:
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- ADX (Average Directional Index)
- Bollinger Bands
- ATR (Average True Range) for volatility

Example:
    >>> import polars as pl
    >>> from imst_quant.trading.advanced_signals import macd_signal
    >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 106, 110]})
    >>> df = macd_signal(df)
    >>> print(df["macd_signal"].to_list())
"""

from typing import Tuple

import polars as pl


def macd_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """Generate MACD (Moving Average Convergence Divergence) signals.

    MACD is a momentum indicator that shows the relationship between
    two exponential moving averages. Generates signals on histogram crossovers.

    Args:
        df: DataFrame containing price data.
        price_col: Name of the price column (default: "close").
        fast_period: Period for fast EMA (default: 12).
        slow_period: Period for slow EMA (default: 26).
        signal_period: Period for signal line EMA (default: 9).

    Returns:
        DataFrame with new columns:
        - macd_line: MACD line (fast EMA - slow EMA)
        - signal_line: Signal line (EMA of MACD line)
        - macd_histogram: MACD line - signal line
        - macd_signal: 1 (bullish crossover), -1 (bearish crossover), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({"close": list(range(100, 150))})
        >>> result = macd_signal(df)
        >>> print(result.select(["close", "macd_signal"]))
    """
    # Calculate EMAs
    fast_ema = df[price_col].ewm_mean(span=fast_period, adjust=False)
    slow_ema = df[price_col].ewm_mean(span=slow_period, adjust=False)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm_mean(span=signal_period, adjust=False)

    # MACD histogram
    macd_histogram = macd_line - signal_line

    # Generate signals based on histogram crossover
    df = df.with_columns([
        macd_line.alias("macd_line"),
        signal_line.alias("signal_line"),
        macd_histogram.alias("macd_histogram"),
    ])

    # Bullish: histogram crosses above 0
    # Bearish: histogram crosses below 0
    df = df.with_columns(
        pl.when(
            (pl.col("macd_histogram") > 0) & (pl.col("macd_histogram").shift(1) <= 0)
        )
        .then(1)  # Bullish crossover
        .when(
            (pl.col("macd_histogram") < 0) & (pl.col("macd_histogram").shift(1) >= 0)
        )
        .then(-1)  # Bearish crossover
        .otherwise(0)
        .alias("macd_signal")
    )

    return df


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

    The Stochastic Oscillator compares closing price to the price range
    over a period. Signals are generated when crossing oversold/overbought levels.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of high price column (default: "high").
        low_col: Name of low price column (default: "low").
        close_col: Name of close price column (default: "close").
        k_period: Period for %K calculation (default: 14).
        d_period: Period for %D smoothing (default: 3).
        oversold: Oversold threshold (default: 20).
        overbought: Overbought threshold (default: 80).

    Returns:
        DataFrame with new columns:
        - stoch_k: %K line (fast stochastic)
        - stoch_d: %D line (slow stochastic)
        - stoch_signal: 1 (oversold cross), -1 (overbought cross), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 110, 107, 112],
        ...     "low": [100, 102, 105, 103, 108],
        ...     "close": [103, 106, 108, 105, 111],
        ... })
        >>> result = stochastic_signal(df)
    """
    # Calculate highest high and lowest low over k_period
    highest_high = df[high_col].rolling_max(window_size=k_period)
    lowest_low = df[low_col].rolling_min(window_size=k_period)

    # Calculate %K
    stoch_k = 100 * (df[close_col] - lowest_low) / (highest_high - lowest_low)

    # Calculate %D (moving average of %K)
    stoch_d = stoch_k.rolling_mean(window_size=d_period)

    df = df.with_columns([
        stoch_k.alias("stoch_k"),
        stoch_d.alias("stoch_d"),
    ])

    # Generate signals
    # Buy when %K crosses above oversold from below
    # Sell when %K crosses below overbought from above
    df = df.with_columns(
        pl.when(
            (pl.col("stoch_k") > oversold) & (pl.col("stoch_k").shift(1) <= oversold)
        )
        .then(1)  # Oversold bounce
        .when(
            (pl.col("stoch_k") < overbought) & (pl.col("stoch_k").shift(1) >= overbought)
        )
        .then(-1)  # Overbought reversal
        .otherwise(0)
        .alias("stoch_signal")
    )

    return df


def bollinger_bands_signal(
    df: pl.DataFrame,
    price_col: str = "close",
    period: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Generate Bollinger Bands signals.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at N standard deviations. Signals when price crosses bands.

    Args:
        df: DataFrame containing price data.
        price_col: Name of price column (default: "close").
        period: Period for SMA calculation (default: 20).
        num_std: Number of standard deviations for bands (default: 2.0).

    Returns:
        DataFrame with new columns:
        - bb_middle: Middle band (SMA)
        - bb_upper: Upper band (middle + num_std * std)
        - bb_lower: Lower band (middle - num_std * std)
        - bb_width: Band width (normalized volatility)
        - bb_signal: 1 (bounce off lower), -1 (rejection at upper), 0 (neutral)

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 98, 105, 110, 108]})
        >>> result = bollinger_bands_signal(df, period=5)
    """
    # Calculate middle band (SMA)
    bb_middle = df[price_col].rolling_mean(window_size=period)

    # Calculate standard deviation
    bb_std = df[price_col].rolling_std(window_size=period)

    # Calculate upper and lower bands
    bb_upper = bb_middle + (num_std * bb_std)
    bb_lower = bb_middle - (num_std * bb_std)

    # Calculate band width (volatility indicator)
    bb_width = (bb_upper - bb_lower) / bb_middle

    df = df.with_columns([
        bb_middle.alias("bb_middle"),
        bb_upper.alias("bb_upper"),
        bb_lower.alias("bb_lower"),
        bb_width.alias("bb_width"),
    ])

    # Generate signals
    # Buy when price touches/crosses lower band
    # Sell when price touches/crosses upper band
    df = df.with_columns(
        pl.when(pl.col(price_col) <= pl.col("bb_lower"))
        .then(1)  # Oversold (bounce expected)
        .when(pl.col(price_col) >= pl.col("bb_upper"))
        .then(-1)  # Overbought (reversal expected)
        .otherwise(0)
        .alias("bb_signal")
    )

    return df


def atr_volatility(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Calculate Average True Range (ATR) for volatility measurement.

    ATR measures market volatility by calculating the average of true ranges
    over a specified period. Used for position sizing and stop-loss placement.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of high price column (default: "high").
        low_col: Name of low price column (default: "low").
        close_col: Name of close price column (default: "close").
        period: Period for ATR calculation (default: 14).

    Returns:
        DataFrame with new columns:
        - true_range: True range for each period
        - atr: Average True Range
        - atr_percent: ATR as percentage of close price

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 110],
        ...     "low": [100, 102, 105],
        ...     "close": [103, 106, 108],
        ... })
        >>> result = atr_volatility(df)
    """
    # Calculate true range components
    high_low = df[high_col] - df[low_col]
    high_close = (df[high_col] - df[close_col].shift(1)).abs()
    low_close = (df[low_col] - df[close_col].shift(1)).abs()

    # True range is the maximum of the three
    true_range = pl.max_horizontal(high_low, high_close, low_close)

    # Calculate ATR (EMA of true range)
    atr = true_range.ewm_mean(span=period, adjust=False)

    # Calculate ATR as percentage of price
    atr_percent = (atr / df[close_col]) * 100

    return df.with_columns([
        true_range.alias("true_range"),
        atr.alias("atr"),
        atr_percent.alias("atr_percent"),
    ])


def adx_trend_strength(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Calculate Average Directional Index (ADX) for trend strength.

    ADX measures the strength of a trend (not direction). Values above 25
    indicate strong trend, below 20 indicate weak/no trend.

    Args:
        df: DataFrame containing OHLC data.
        high_col: Name of high price column (default: "high").
        low_col: Name of low price column (default: "low").
        close_col: Name of close price column (default: "close").
        period: Period for ADX calculation (default: 14).

    Returns:
        DataFrame with new columns:
        - plus_di: Positive directional indicator
        - minus_di: Negative directional indicator
        - adx: Average directional index (trend strength)
        - trend_strength: "strong" if ADX > 25, "weak" if ADX < 20

    Example:
        >>> df = pl.DataFrame({
        ...     "high": [105, 108, 110, 107],
        ...     "low": [100, 102, 105, 103],
        ...     "close": [103, 106, 108, 105],
        ... })
        >>> result = adx_trend_strength(df)
    """
    # Calculate directional movement
    plus_dm = (df[high_col] - df[high_col].shift(1)).clip(lower_bound=0)
    minus_dm = (df[low_col].shift(1) - df[low_col]).clip(lower_bound=0)

    # Zero out when opposite movement is larger
    plus_dm = pl.when(plus_dm > minus_dm).then(plus_dm).otherwise(0)
    minus_dm = pl.when(minus_dm > plus_dm).then(minus_dm).otherwise(0)

    # Calculate ATR for normalization
    df_temp = atr_volatility(df, high_col, low_col, close_col, period)
    atr = df_temp["atr"]

    # Calculate directional indicators
    plus_di = 100 * (plus_dm.ewm_mean(span=period, adjust=False) / atr)
    minus_di = 100 * (minus_dm.ewm_mean(span=period, adjust=False) / atr)

    # Calculate DX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))

    # Calculate ADX (EMA of DX)
    adx = dx.ewm_mean(span=period, adjust=False)

    df = df.with_columns([
        plus_di.alias("plus_di"),
        minus_di.alias("minus_di"),
        adx.alias("adx"),
    ])

    # Classify trend strength
    df = df.with_columns(
        pl.when(pl.col("adx") > 25)
        .then(pl.lit("strong"))
        .when(pl.col("adx") < 20)
        .then(pl.lit("weak"))
        .otherwise(pl.lit("moderate"))
        .alias("trend_strength")
    )

    return df
