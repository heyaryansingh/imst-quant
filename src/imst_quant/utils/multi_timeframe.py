"""Multi-timeframe analysis utility for robust signal generation.

This module provides tools for analyzing trading signals across multiple
timeframes to improve signal quality and reduce false positives.

The multi-timeframe approach:
1. Analyzes price action across different time periods (e.g., 5min, 15min, 1h, 4h, 1d)
2. Aggregates signals to identify confluence zones
3. Provides weighted voting mechanisms for signal strength
4. Detects trend alignment across timeframes

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.multi_timeframe import analyze_multi_timeframe
    >>> df = pl.read_parquet("data/gold/features.parquet")
    >>> result = analyze_multi_timeframe(df, timeframes=[5, 15, 60])
    >>> print(result.select(["date", "mtf_signal", "mtf_strength"]))
"""

from typing import List, Literal

import polars as pl


def resample_to_timeframe(
    df: pl.DataFrame,
    timeframe_minutes: int,
    date_col: str = "date",
    agg_dict: dict | None = None,
) -> pl.DataFrame:
    """Resample data to a specific timeframe.

    Args:
        df: Input DataFrame with time-series data.
        timeframe_minutes: Target timeframe in minutes (e.g., 5, 15, 60, 240, 1440).
        date_col: Name of the datetime column (default: "date").
        agg_dict: Custom aggregation dictionary. If None, uses OHLC defaults.

    Returns:
        Resampled DataFrame with the specified timeframe.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": pl.date_range(start="2024-01-01", end="2024-01-02", interval="1h"),
        ...     "close": [100, 101, 102, 103, 104, 105, 106, 107],
        ...     "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
        ... })
        >>> result = resample_to_timeframe(df, timeframe_minutes=240)
        >>> print(result)
    """
    if agg_dict is None:
        # Default OHLC aggregation
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

    # Convert to DataFrame operations
    df = df.sort(date_col)

    # Group by time intervals
    interval = f"{timeframe_minutes}m"
    df = df.group_by_dynamic(date_col, every=interval).agg(
        [
            pl.col(col).first().alias(col) if agg == "first"
            else pl.col(col).last().alias(col) if agg == "last"
            else pl.col(col).max().alias(col) if agg == "max"
            else pl.col(col).min().alias(col) if agg == "min"
            else pl.col(col).sum().alias(col) if agg == "sum"
            else pl.col(col).mean().alias(col)
            for col, agg in agg_dict.items()
            if col in df.columns
        ]
    )

    return df


def calculate_timeframe_signal(
    df: pl.DataFrame,
    method: Literal["trend", "momentum", "mean_reversion"] = "trend",
    lookback: int = 20,
) -> pl.DataFrame:
    """Calculate signal for a single timeframe.

    Args:
        df: DataFrame with OHLC data.
        method: Signal calculation method:
            - "trend": Uses price vs MA and slope
            - "momentum": Uses rate of change and RSI
            - "mean_reversion": Uses Bollinger Bands and Z-score
        lookback: Lookback period for calculations (default: 20).

    Returns:
        DataFrame with additional signal column.

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 101, 105, 108, 106, 110, 112, 109, 115]
        ... })
        >>> result = calculate_timeframe_signal(df, method="trend")
        >>> print(result["signal"])
    """
    if method == "trend":
        # Price relative to moving average
        ma = df["close"].rolling_mean(window_size=lookback)
        slope = (df["close"] - df["close"].shift(lookback)) / lookback

        df = df.with_columns([
            ma.alias("ma"),
            slope.alias("slope"),
        ])

        # Bullish: price > MA and positive slope
        # Bearish: price < MA and negative slope
        df = df.with_columns(
            pl.when((pl.col("close") > pl.col("ma")) & (pl.col("slope") > 0))
            .then(1)
            .when((pl.col("close") < pl.col("ma")) & (pl.col("slope") < 0))
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

    elif method == "momentum":
        # Rate of change
        roc = (df["close"] - df["close"].shift(lookback)) / df["close"].shift(lookback) * 100

        # Simple RSI approximation
        delta = df["close"].diff()
        gain = delta.clip_min(0).rolling_mean(window_size=lookback)
        loss = (-delta).clip_min(0).rolling_mean(window_size=lookback)
        rs = gain / loss.replace(0, 1)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        df = df.with_columns([
            roc.alias("roc"),
            rsi.alias("rsi"),
        ])

        # Bullish: positive ROC and RSI < 70
        # Bearish: negative ROC and RSI > 30
        df = df.with_columns(
            pl.when((pl.col("roc") > 0) & (pl.col("rsi") < 70))
            .then(1)
            .when((pl.col("roc") < 0) & (pl.col("rsi") > 30))
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

    elif method == "mean_reversion":
        # Z-score based on rolling mean and std
        rolling_mean = df["close"].rolling_mean(window_size=lookback)
        rolling_std = df["close"].rolling_std(window_size=lookback)
        z_score = (df["close"] - rolling_mean) / rolling_std.replace(0, 1)

        df = df.with_columns([
            rolling_mean.alias("rolling_mean"),
            z_score.alias("z_score"),
        ])

        # Mean reversion: buy when oversold, sell when overbought
        df = df.with_columns(
            pl.when(pl.col("z_score") < -1.5)
            .then(1)  # Oversold, expect reversion up
            .when(pl.col("z_score") > 1.5)
            .then(-1)  # Overbought, expect reversion down
            .otherwise(0)
            .alias("signal")
        )

    return df


def analyze_multi_timeframe(
    df: pl.DataFrame,
    timeframes: List[int] = [5, 15, 60, 240],
    method: Literal["trend", "momentum", "mean_reversion"] = "trend",
    lookback: int = 20,
    date_col: str = "date",
    weights: List[float] | None = None,
) -> pl.DataFrame:
    """Analyze signals across multiple timeframes and aggregate results.

    Args:
        df: Input DataFrame with minute-level or higher frequency data.
        timeframes: List of timeframes in minutes (default: [5, 15, 60, 240]).
        method: Signal calculation method (default: "trend").
        lookback: Lookback period for signal calculations (default: 20).
        date_col: Name of the datetime column (default: "date").
        weights: Optional weights for each timeframe (default: equal weights).
            Weights should sum to 1.0. Higher timeframes typically get higher weights.

    Returns:
        DataFrame with multi-timeframe analysis columns:
        - mtf_signal: Aggregated signal (-1, 0, 1)
        - mtf_strength: Signal strength (0-100)
        - mtf_agreement: Percentage of timeframes agreeing
        - tf_X_signal: Individual signal for each timeframe X

    Example:
        >>> df = pl.read_parquet("data/gold/features.parquet")
        >>> result = analyze_multi_timeframe(
        ...     df,
        ...     timeframes=[15, 60, 240],
        ...     method="trend",
        ...     weights=[0.2, 0.3, 0.5]
        ... )
        >>> print(result.select(["date", "mtf_signal", "mtf_strength"]))
    """
    if weights is None:
        # Equal weights if not specified
        weights = [1.0 / len(timeframes)] * len(timeframes)
    elif len(weights) != len(timeframes):
        raise ValueError(f"Weights length ({len(weights)}) must match timeframes length ({len(timeframes)})")
    elif abs(sum(weights) - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    # Store original timestamps
    original_df = df.select([date_col]).sort(date_col)

    timeframe_signals = []

    # Calculate signals for each timeframe
    for tf_minutes, weight in zip(timeframes, weights):
        # Resample to timeframe
        tf_df = resample_to_timeframe(df, tf_minutes, date_col)

        # Calculate signal
        tf_df = calculate_timeframe_signal(tf_df, method=method, lookback=lookback)

        # Keep only date and signal
        tf_df = tf_df.select([
            pl.col(date_col),
            pl.col("signal").alias(f"tf_{tf_minutes}_signal"),
        ])

        timeframe_signals.append((tf_df, weight))

    # Join all timeframe signals back to original timestamps
    result = original_df
    for tf_df, _ in timeframe_signals:
        # Use asof join to forward-fill signals
        result = result.join_asof(tf_df, on=date_col, strategy="backward")

    # Calculate aggregated signal
    signal_cols = [f"tf_{tf}_signal" for tf in timeframes]

    # Weighted average signal
    weighted_sum_expr = sum(
        pl.col(col).fill_null(0) * weight
        for col, weight in zip(signal_cols, weights)
    )

    # Agreement percentage
    non_zero_count = sum(pl.col(col).fill_null(0) != 0 for col in signal_cols)
    positive_count = sum(pl.col(col).fill_null(0) == 1 for col in signal_cols)
    negative_count = sum(pl.col(col).fill_null(0) == -1 for col in signal_cols)

    result = result.with_columns([
        weighted_sum_expr.alias("mtf_weighted_signal"),
        non_zero_count.alias("mtf_active_signals"),
        positive_count.alias("mtf_bullish_count"),
        negative_count.alias("mtf_bearish_count"),
    ])

    # Final aggregated signal
    result = result.with_columns([
        pl.when(pl.col("mtf_weighted_signal") > 0.3)
        .then(1)
        .when(pl.col("mtf_weighted_signal") < -0.3)
        .then(-1)
        .otherwise(0)
        .alias("mtf_signal"),
    ])

    # Signal strength (0-100)
    result = result.with_columns([
        (pl.col("mtf_weighted_signal").abs() * 100).alias("mtf_strength"),
    ])

    # Agreement percentage
    result = result.with_columns([
        pl.when(pl.col("mtf_signal") == 1)
        .then(pl.col("mtf_bullish_count") * 100 / len(timeframes))
        .when(pl.col("mtf_signal") == -1)
        .then(pl.col("mtf_bearish_count") * 100 / len(timeframes))
        .otherwise(0.0)
        .alias("mtf_agreement"),
    ])

    return result


def detect_trend_alignment(
    df: pl.DataFrame,
    timeframes: List[int] = [15, 60, 240, 1440],
    date_col: str = "date",
    alignment_threshold: float = 0.75,
) -> pl.DataFrame:
    """Detect when trends are aligned across multiple timeframes.

    Strong trends occur when multiple timeframes agree on direction.
    This function identifies these high-probability setups.

    Args:
        df: Input DataFrame with price data.
        timeframes: List of timeframes in minutes (default: [15, 60, 240, 1440]).
        date_col: Name of the datetime column (default: "date").
        alignment_threshold: Minimum agreement ratio for alignment (default: 0.75).

    Returns:
        DataFrame with trend alignment columns:
        - trend_aligned: Boolean indicating alignment
        - trend_direction: 1 (bullish), -1 (bearish), 0 (no alignment)
        - alignment_score: Percentage of timeframes agreeing (0-100)

    Example:
        >>> df = pl.read_parquet("data/gold/features.parquet")
        >>> result = detect_trend_alignment(df, timeframes=[60, 240, 1440])
        >>> aligned = result.filter(pl.col("trend_aligned") == True)
        >>> print(aligned.select(["date", "trend_direction", "alignment_score"]))
    """
    mtf_df = analyze_multi_timeframe(
        df,
        timeframes=timeframes,
        method="trend",
        date_col=date_col,
    )

    # Calculate alignment
    signal_cols = [f"tf_{tf}_signal" for tf in timeframes]
    total_timeframes = len(timeframes)

    bullish_aligned = sum(pl.col(col).fill_null(0) == 1 for col in signal_cols)
    bearish_aligned = sum(pl.col(col).fill_null(0) == -1 for col in signal_cols)

    mtf_df = mtf_df.with_columns([
        bullish_aligned.alias("bullish_aligned_count"),
        bearish_aligned.alias("bearish_aligned_count"),
    ])

    # Check if alignment threshold is met
    mtf_df = mtf_df.with_columns([
        pl.when(pl.col("bullish_aligned_count") >= alignment_threshold * total_timeframes)
        .then(1)
        .when(pl.col("bearish_aligned_count") >= alignment_threshold * total_timeframes)
        .then(-1)
        .otherwise(0)
        .alias("trend_direction"),
    ])

    mtf_df = mtf_df.with_columns([
        (pl.col("trend_direction") != 0).alias("trend_aligned"),
        pl.when(pl.col("trend_direction") == 1)
        .then(pl.col("bullish_aligned_count") * 100 / total_timeframes)
        .when(pl.col("trend_direction") == -1)
        .then(pl.col("bearish_aligned_count") * 100 / total_timeframes)
        .otherwise(0.0)
        .alias("alignment_score"),
    ])

    return mtf_df
