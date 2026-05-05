"""Signal quality scoring and filtering utilities.

This module provides utilities to score and filter trading signals based on
multiple quality metrics including conviction, volatility regime, trend strength,
and market conditions.

Functions:
    calculate_signal_conviction: Score signal strength based on multiple indicators
    filter_signals_by_quality: Filter signals based on quality thresholds
    score_signal_quality: Comprehensive quality score for a signal
    get_signal_thresholds: Get recommended thresholds for different risk levels

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.signal_quality import calculate_signal_conviction
    >>> df = pl.DataFrame({
    ...     "macd_signal": [1, 0, -1],
    ...     "stoch_signal": [1, 0, 0],
    ...     "bb_signal": [1, 0, -1],
    ...     "adx": [30, 15, 25],
    ... })
    >>> df = calculate_signal_conviction(df)
    >>> print(df["signal_conviction"].to_list())
"""

from typing import Dict, List, Optional

import polars as pl


def calculate_signal_conviction(
    df: pl.DataFrame,
    signal_cols: Optional[List[str]] = None,
    adx_col: str = "adx",
    atr_col: Optional[str] = None,
) -> pl.DataFrame:
    """Calculate signal conviction score based on indicator agreement.

    Signal conviction measures the strength of a trading signal by checking
    how many technical indicators agree on the direction.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names (default: auto-detect *_signal columns).
        adx_col: ADX column name for trend strength weighting.
        atr_col: ATR column name for volatility weighting (optional).

    Returns:
        DataFrame with added columns:
        - signal_conviction: Raw conviction score (sum of agreeing signals)
        - signal_conviction_norm: Normalized conviction score (0-1)
        - signal_direction: Consensus direction (1=bullish, -1=bearish, 0=neutral)
        - signal_quality: Quality score incorporating trend and volatility

    Example:
        >>> df = pl.DataFrame({
        ...     "macd_signal": [1, 0, -1, 1],
        ...     "stoch_signal": [1, 0, -1, 0],
        ...     "bb_signal": [1, 1, -1, -1],
        ...     "adx": [30, 15, 25, 20],
        ... })
        >>> result = calculate_signal_conviction(df)
        >>> print(result[["signal_conviction", "signal_direction"]])
    """
    # Auto-detect signal columns if not provided
    if signal_cols is None:
        signal_cols = [col for col in df.columns if col.endswith("_signal")]

    if not signal_cols:
        raise ValueError("No signal columns found. Provide signal_cols parameter.")

    # Calculate raw conviction (sum of all signals)
    signal_conviction = pl.lit(0)
    for col in signal_cols:
        signal_conviction = signal_conviction + df[col]

    df = df.with_columns(signal_conviction.alias("signal_conviction"))

    # Normalize conviction score
    num_signals = len(signal_cols)
    df = df.with_columns(
        (pl.col("signal_conviction").abs() / num_signals).alias("signal_conviction_norm")
    )

    # Determine consensus direction
    df = df.with_columns(
        pl.when(pl.col("signal_conviction") > 0)
        .then(1)
        .when(pl.col("signal_conviction") < 0)
        .then(-1)
        .otherwise(0)
        .alias("signal_direction")
    )

    # Calculate quality score incorporating trend strength
    if adx_col in df.columns:
        # Weight conviction by trend strength (ADX > 25 = strong trend)
        adx_weight = pl.when(pl.col(adx_col) > 25).then(1.0).otherwise(0.5)
        df = df.with_columns((pl.col("signal_conviction_norm") * adx_weight).alias("signal_quality"))
    else:
        df = df.with_columns(pl.col("signal_conviction_norm").alias("signal_quality"))

    # Further weight by volatility if ATR is available
    if atr_col and atr_col in df.columns:
        # Lower quality in extreme volatility (top 10% ATR)
        atr_quantile_90 = df[atr_col].quantile(0.9)
        vol_weight = pl.when(pl.col(atr_col) > atr_quantile_90).then(0.7).otherwise(1.0)
        df = df.with_columns((pl.col("signal_quality") * vol_weight).alias("signal_quality"))

    return df


def filter_signals_by_quality(
    df: pl.DataFrame,
    min_conviction: float = 0.5,
    min_quality: float = 0.4,
    min_adx: Optional[float] = None,
    max_atr_percentile: Optional[float] = None,
) -> pl.DataFrame:
    """Filter trading signals based on quality thresholds.

    Args:
        df: DataFrame with signal_conviction_norm and signal_quality columns.
        min_conviction: Minimum normalized conviction score (0-1).
        min_quality: Minimum quality score (0-1).
        min_adx: Minimum ADX value (trend strength filter).
        max_atr_percentile: Maximum ATR percentile (volatility filter, 0-1).

    Returns:
        Filtered DataFrame containing only high-quality signals.

    Example:
        >>> df = calculate_signal_conviction(df)
        >>> filtered = filter_signals_by_quality(
        ...     df, min_conviction=0.6, min_quality=0.5
        ... )
    """
    # Start with conviction and quality filters
    mask = (pl.col("signal_conviction_norm") >= min_conviction) & (
        pl.col("signal_quality") >= min_quality
    )

    # Add ADX filter if specified
    if min_adx is not None and "adx" in df.columns:
        mask = mask & (pl.col("adx") >= min_adx)

    # Add ATR filter if specified
    if max_atr_percentile is not None and "atr" in df.columns:
        atr_threshold = df["atr"].quantile(max_atr_percentile)
        mask = mask & (pl.col("atr") <= atr_threshold)

    return df.filter(mask)


def score_signal_quality(
    df: pl.DataFrame,
    signal_cols: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Calculate comprehensive signal quality score with custom weights.

    Combines multiple signal sources with custom weights to produce a
    weighted quality score.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names (default: auto-detect).
        weights: Dictionary mapping signal column names to weights.
            Default: equal weighting.

    Returns:
        DataFrame with added weighted_signal_score column.

    Example:
        >>> df = pl.DataFrame({
        ...     "macd_signal": [1, 0, -1],
        ...     "stoch_signal": [1, 0, 0],
        ...     "bb_signal": [1, 0, -1],
        ... })
        >>> weights = {"macd_signal": 0.5, "stoch_signal": 0.3, "bb_signal": 0.2}
        >>> result = score_signal_quality(df, weights=weights)
    """
    # Auto-detect signal columns if not provided
    if signal_cols is None:
        signal_cols = [col for col in df.columns if col.endswith("_signal")]

    if not signal_cols:
        raise ValueError("No signal columns found.")

    # Default to equal weights
    if weights is None:
        weight_value = 1.0 / len(signal_cols)
        weights = {col: weight_value for col in signal_cols}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted score
    weighted_score = pl.lit(0.0)
    for col, weight in normalized_weights.items():
        if col in df.columns:
            weighted_score = weighted_score + (df[col] * weight)

    df = df.with_columns(weighted_score.alias("weighted_signal_score"))

    # Add binary high-quality flag
    df = df.with_columns(
        pl.when(pl.col("weighted_signal_score").abs() > 0.5)
        .then(True)
        .otherwise(False)
        .alias("is_high_quality_signal")
    )

    return df


def get_signal_thresholds(risk_level: str = "medium") -> Dict[str, float]:
    """Get recommended signal quality thresholds for different risk levels.

    Args:
        risk_level: Risk tolerance level ("low", "medium", "high").

    Returns:
        Dictionary with recommended thresholds for:
        - min_conviction: Minimum normalized conviction score
        - min_quality: Minimum quality score
        - min_adx: Minimum trend strength (ADX)
        - max_atr_percentile: Maximum volatility (ATR percentile)

    Example:
        >>> thresholds = get_signal_thresholds("low")
        >>> print(f"Conservative conviction: {thresholds['min_conviction']}")
    """
    thresholds = {
        "low": {  # Conservative: fewer, higher-quality signals
            "min_conviction": 0.7,
            "min_quality": 0.6,
            "min_adx": 25.0,
            "max_atr_percentile": 0.75,
        },
        "medium": {  # Balanced approach
            "min_conviction": 0.5,
            "min_quality": 0.4,
            "min_adx": 20.0,
            "max_atr_percentile": 0.85,
        },
        "high": {  # Aggressive: more signals, lower quality bar
            "min_conviction": 0.3,
            "min_quality": 0.25,
            "min_adx": 15.0,
            "max_atr_percentile": 0.95,
        },
    }

    if risk_level not in thresholds:
        raise ValueError(f"Unknown risk_level: {risk_level}. Use 'low', 'medium', or 'high'.")

    return thresholds[risk_level]


def calculate_signal_win_rate(
    df: pl.DataFrame,
    signal_col: str = "signal_direction",
    returns_col: str = "returns",
    lookahead: int = 1,
) -> pl.DataFrame:
    """Calculate historical win rate for signals.

    Analyzes past signal performance by checking if returns in the
    next N periods matched the signal direction.

    Args:
        df: DataFrame with signals and returns.
        signal_col: Name of signal direction column (1, -1, or 0).
        returns_col: Name of returns column.
        lookahead: Number of periods ahead to check returns.

    Returns:
        DataFrame with added columns:
        - signal_outcome: 1 if correct, -1 if wrong, 0 if neutral
        - signal_hit: Boolean indicating if signal was correct
        - rolling_win_rate: Rolling win rate over last 20 signals

    Example:
        >>> df = pl.DataFrame({
        ...     "signal_direction": [1, -1, 0, 1],
        ...     "returns": [0.01, -0.02, 0.005, 0.015],
        ... })
        >>> result = calculate_signal_win_rate(df)
    """
    # Calculate future returns
    future_returns = df[returns_col].shift(-lookahead)

    # Determine if signal was correct
    # Bullish signal (1) correct if future return > 0
    # Bearish signal (-1) correct if future return < 0
    signal_outcome = pl.when(df[signal_col] == 0).then(0).otherwise(
        pl.when(
            ((df[signal_col] > 0) & (future_returns > 0))
            | ((df[signal_col] < 0) & (future_returns < 0))
        )
        .then(1)
        .otherwise(-1)
    )

    df = df.with_columns([
        signal_outcome.alias("signal_outcome"),
        (signal_outcome == 1).alias("signal_hit"),
    ])

    # Calculate rolling win rate (only for non-zero signals)
    win_rate_window = 20
    df = df.with_columns(
        pl.when(pl.col(signal_col) != 0)
        .then(pl.col("signal_hit").cast(pl.Float64).rolling_mean(window_size=win_rate_window))
        .otherwise(None)
        .alias("rolling_win_rate")
    )

    return df
