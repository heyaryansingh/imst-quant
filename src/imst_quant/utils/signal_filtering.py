"""Adaptive signal filtering utilities for dynamic threshold adjustment.

This module provides advanced signal filtering capabilities that adapt to
changing market conditions, helping to reduce false signals and improve
trade quality.

Functions:
    calculate_adaptive_threshold: Compute dynamic thresholds based on volatility
    filter_by_market_regime: Filter signals based on detected market regime
    apply_confidence_filter: Filter signals requiring minimum confidence level
    combine_signal_filters: Chain multiple filters with configurable rules

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.signal_filtering import calculate_adaptive_threshold
    >>> df = pl.DataFrame({
    ...     "signal": [1.5, 0.3, -1.2, 0.8],
    ...     "volatility": [0.02, 0.01, 0.03, 0.015],
    ... })
    >>> df = calculate_adaptive_threshold(df)
    >>> filtered = df.filter(pl.col("signal").abs() > pl.col("threshold"))
"""

from typing import Dict, List, Literal, Optional

import polars as pl


def calculate_adaptive_threshold(
    df: pl.DataFrame,
    signal_col: str = "signal",
    volatility_col: str = "volatility",
    base_threshold: float = 0.5,
    vol_scaling: float = 2.0,
    lookback: int = 20,
) -> pl.DataFrame:
    """Calculate adaptive signal thresholds based on rolling volatility.

    Adjusts signal thresholds dynamically to account for changing market conditions.
    Higher volatility leads to higher thresholds to reduce noise.

    Args:
        df: DataFrame containing signal and volatility columns.
        signal_col: Name of the signal column.
        volatility_col: Name of the volatility column.
        base_threshold: Baseline threshold value (default: 0.5).
        vol_scaling: Volatility scaling factor (default: 2.0).
        lookback: Rolling window for volatility normalization (default: 20).

    Returns:
        DataFrame with added columns:
        - threshold: Adaptive threshold value
        - signal_strength: Normalized signal strength relative to threshold
        - above_threshold: Boolean flag for signals exceeding threshold

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1.0, 0.5, -1.5],
        ...     "volatility": [0.02, 0.01, 0.03],
        ... })
        >>> result = calculate_adaptive_threshold(df)
        >>> print(result["threshold"].to_list())
    """
    # Calculate rolling volatility stats
    vol_mean = df.select(
        pl.col(volatility_col).rolling_mean(lookback).alias("vol_mean")
    )
    vol_std = df.select(
        pl.col(volatility_col).rolling_std(lookback).alias("vol_std")
    )

    # Compute adaptive threshold: base + (current_vol - mean_vol) * scaling
    df = df.with_columns(
        [
            vol_mean.to_series().alias("vol_mean"),
            vol_std.to_series().alias("vol_std"),
        ]
    )

    df = df.with_columns(
        [
            (
                base_threshold
                + ((pl.col(volatility_col) - pl.col("vol_mean")) / pl.col("vol_std"))
                * vol_scaling
                * base_threshold
            )
            .fill_null(base_threshold)
            .alias("threshold"),
        ]
    )

    # Calculate signal strength relative to threshold
    df = df.with_columns(
        [
            (pl.col(signal_col).abs() / pl.col("threshold")).alias("signal_strength"),
            (pl.col(signal_col).abs() > pl.col("threshold")).alias("above_threshold"),
        ]
    )

    return df.drop(["vol_mean", "vol_std"])


def filter_by_market_regime(
    df: pl.DataFrame,
    signal_col: str = "signal",
    regime_col: str = "regime",
    allowed_regimes: Optional[List[str]] = None,
    regime_map: Optional[Dict[str, List[int]]] = None,
) -> pl.DataFrame:
    """Filter signals based on detected market regime.

    Args:
        df: DataFrame with signal and regime columns.
        signal_col: Name of signal column.
        regime_col: Name of regime column (e.g., "trending", "mean_reverting").
        allowed_regimes: List of regime values to allow (default: ["trending"]).
        regime_map: Optional mapping of regime names to signal directions.
            Example: {"trending": [1], "mean_reverting": [-1]} only allows
            long signals in trending and short signals in mean reverting.

    Returns:
        Filtered DataFrame containing only signals from allowed regimes.

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1, -1, 1, 0],
        ...     "regime": ["trending", "choppy", "trending", "mean_reverting"],
        ... })
        >>> filtered = filter_by_market_regime(df, allowed_regimes=["trending"])
        >>> print(len(filtered))
    """
    if allowed_regimes is None:
        allowed_regimes = ["trending"]

    # Basic regime filtering
    filtered_df = df.filter(pl.col(regime_col).is_in(allowed_regimes))

    # Apply directional regime mapping if provided
    if regime_map:
        conditions = []
        for regime, directions in regime_map.items():
            regime_condition = pl.col(regime_col) == regime
            if 1 in directions and -1 in directions:
                # Allow both directions
                conditions.append(regime_condition)
            elif 1 in directions:
                # Only long signals
                conditions.append(regime_condition & (pl.col(signal_col) > 0))
            elif -1 in directions:
                # Only short signals
                conditions.append(regime_condition & (pl.col(signal_col) < 0))

        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            filtered_df = filtered_df.filter(combined_condition)

    return filtered_df


def apply_confidence_filter(
    df: pl.DataFrame,
    signal_col: str = "signal",
    confidence_col: str = "confidence",
    min_confidence: float = 0.6,
    require_direction_match: bool = True,
) -> pl.DataFrame:
    """Filter signals based on confidence level.

    Args:
        df: DataFrame with signal and confidence columns.
        signal_col: Name of signal column.
        confidence_col: Name of confidence column (0-1 scale).
        min_confidence: Minimum confidence threshold (default: 0.6).
        require_direction_match: If True, confidence sign must match signal
            direction (default: True).

    Returns:
        Filtered DataFrame with high-confidence signals only.

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1, -1, 1, -1],
        ...     "confidence": [0.8, 0.5, 0.7, 0.9],
        ... })
        >>> filtered = apply_confidence_filter(df, min_confidence=0.6)
        >>> print(len(filtered))
    """
    # Basic confidence filtering
    filtered_df = df.filter(pl.col(confidence_col).abs() >= min_confidence)

    # Ensure confidence direction matches signal direction
    if require_direction_match:
        filtered_df = filtered_df.filter(
            (pl.col(signal_col) * pl.col(confidence_col)) > 0
        )

    return filtered_df


def combine_signal_filters(
    df: pl.DataFrame,
    filters: List[Dict],
    combination_rule: Literal["all", "any", "majority"] = "all",
    signal_col: str = "signal",
) -> pl.DataFrame:
    """Apply multiple filters to signals with configurable combination logic.

    Args:
        df: Input DataFrame with signals.
        filters: List of filter configurations, each a dict with:
            - type: Filter type ("threshold", "regime", "confidence")
            - params: Dict of parameters for that filter
        combination_rule: How to combine filters:
            - "all": Signal must pass ALL filters
            - "any": Signal must pass ANY filter
            - "majority": Signal must pass majority of filters
        signal_col: Name of signal column.

    Returns:
        Filtered DataFrame based on combination rule.

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1.0, 0.3, -1.2],
        ...     "volatility": [0.02, 0.01, 0.03],
        ...     "confidence": [0.8, 0.5, 0.9],
        ...     "regime": ["trending", "choppy", "trending"],
        ... })
        >>> filters = [
        ...     {"type": "threshold", "params": {"base_threshold": 0.5}},
        ...     {"type": "confidence", "params": {"min_confidence": 0.6}},
        ... ]
        >>> result = combine_signal_filters(df, filters, combination_rule="all")
    """
    if not filters:
        return df

    # Create filter result columns
    filter_results = []

    for i, filter_config in enumerate(filters):
        filter_type = filter_config["type"]
        params = filter_config.get("params", {})
        col_name = f"filter_{i}_pass"

        if filter_type == "threshold":
            temp_df = calculate_adaptive_threshold(df, signal_col=signal_col, **params)
            df = df.with_columns(temp_df["above_threshold"].alias(col_name))
            filter_results.append(col_name)

        elif filter_type == "regime":
            allowed_regimes = params.get("allowed_regimes", ["trending"])
            regime_col = params.get("regime_col", "regime")
            df = df.with_columns(
                pl.col(regime_col).is_in(allowed_regimes).alias(col_name)
            )
            filter_results.append(col_name)

        elif filter_type == "confidence":
            min_confidence = params.get("min_confidence", 0.6)
            confidence_col = params.get("confidence_col", "confidence")
            df = df.with_columns(
                (pl.col(confidence_col).abs() >= min_confidence).alias(col_name)
            )
            filter_results.append(col_name)

    # Apply combination rule
    if combination_rule == "all":
        # All filters must pass
        combined = pl.lit(True)
        for col in filter_results:
            combined = combined & pl.col(col)
        result_df = df.filter(combined)

    elif combination_rule == "any":
        # Any filter must pass
        combined = pl.lit(False)
        for col in filter_results:
            combined = combined | pl.col(col)
        result_df = df.filter(combined)

    elif combination_rule == "majority":
        # Majority of filters must pass
        threshold = len(filter_results) // 2 + 1
        sum_expr = pl.lit(0)
        for col in filter_results:
            sum_expr = sum_expr + pl.col(col).cast(pl.Int32)
        result_df = df.filter(sum_expr >= threshold)

    else:
        raise ValueError(
            f"Invalid combination_rule: {combination_rule}. "
            "Must be 'all', 'any', or 'majority'"
        )

    # Drop temporary filter columns
    result_df = result_df.drop([col for col in filter_results if col in result_df.columns])

    return result_df


def calculate_signal_decay(
    df: pl.DataFrame,
    signal_col: str = "signal",
    half_life: int = 5,
    min_threshold: float = 0.1,
) -> pl.DataFrame:
    """Apply exponential decay to signals over time.

    Useful for reducing position sizes as signals age or for implementing
    time-based exit strategies.

    Args:
        df: DataFrame with signals (must have datetime index or sorted by time).
        signal_col: Name of signal column.
        half_life: Number of periods for signal to decay to 50% (default: 5).
        min_threshold: Minimum signal value before setting to zero (default: 0.1).

    Returns:
        DataFrame with decayed signal values in new column 'signal_decayed'.

    Example:
        >>> df = pl.DataFrame({
        ...     "signal": [1.0, 1.0, 1.0, 1.0, 1.0],
        ... })
        >>> result = calculate_signal_decay(df, half_life=2)
        >>> print(result["signal_decayed"].to_list())
    """
    # Calculate decay factor
    decay_factor = 0.5 ** (1 / half_life)

    # Apply exponential decay using cumulative product
    df = df.with_columns(
        [
            pl.col(signal_col)
            .cum_prod()
            .pow(pl.lit(decay_factor))
            .alias("signal_decayed"),
        ]
    )

    # Set values below threshold to zero
    df = df.with_columns(
        [
            pl.when(pl.col("signal_decayed").abs() < min_threshold)
            .then(0.0)
            .otherwise(pl.col("signal_decayed"))
            .alias("signal_decayed"),
        ]
    )

    return df
