"""Market regime detection for adaptive trading strategies.

This module provides methods to identify market regimes (trending,
mean-reverting, high/low volatility) to enable strategy adaptation.

Regime Types:
    - Volatility regimes: Low, Normal, High volatility states
    - Trend regimes: Uptrend, Downtrend, Ranging states
    - Combined regimes: Bull-Low-Vol, Bear-High-Vol, etc.

Methods:
    - Rolling statistics based detection
    - Hidden Markov Model inspired state estimation
    - Threshold-based classification

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.regime_detection import (
    ...     detect_volatility_regime,
    ...     detect_trend_regime,
    ... )
    >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 110]})
    >>> df = detect_volatility_regime(df)
    >>> df = detect_trend_regime(df)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


class VolatilityRegime(Enum):
    """Volatility regime states."""

    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"


class TrendRegime(Enum):
    """Trend regime states."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"


class MarketRegime(Enum):
    """Combined market regime states."""

    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"
    RANGING_QUIET = "ranging_quiet"
    RANGING_VOLATILE = "ranging_volatile"


def detect_volatility_regime(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    window: int = 20,
    low_threshold: float = 0.5,
    high_threshold: float = 1.5,
) -> pl.DataFrame:
    """Detect volatility regime based on rolling volatility percentile.

    Classifies periods as low, normal, or high volatility based on
    how current volatility compares to its historical distribution.

    Args:
        df: DataFrame containing return data.
        return_col: Name of return column. Defaults to "return_1d".
        window: Lookback window for volatility calculation. Defaults to 20.
        low_threshold: Multiplier for low volatility cutoff.
            Volatility below (mean - low_threshold * std) is "low".
        high_threshold: Multiplier for high volatility cutoff.
            Volatility above (mean + high_threshold * std) is "high".

    Returns:
        DataFrame with new columns:
        - rolling_volatility: Rolling standard deviation of returns
        - vol_regime: Categorical regime label
        - vol_regime_numeric: Numeric regime (-1=low, 0=normal, 1=high)

    Example:
        >>> df = pl.DataFrame({"return_1d": np.random.randn(100) * 0.02})
        >>> result = detect_volatility_regime(df)
    """
    # Calculate rolling volatility
    df = df.with_columns(
        pl.col(return_col).rolling_std(window_size=window).alias("rolling_volatility")
    )

    # Calculate mean and std of volatility for thresholding
    vol_series = df["rolling_volatility"].drop_nulls()
    if vol_series.len() > 0:
        vol_mean = float(vol_series.mean())
        vol_std = float(vol_series.std()) if vol_series.len() > 1 else vol_mean * 0.5
    else:
        vol_mean = 0.01
        vol_std = 0.005

    low_cutoff = vol_mean - low_threshold * vol_std
    high_cutoff = vol_mean + high_threshold * vol_std

    # Classify regime
    df = df.with_columns([
        pl.when(pl.col("rolling_volatility") < low_cutoff)
        .then(pl.lit(VolatilityRegime.LOW.value))
        .when(pl.col("rolling_volatility") > high_cutoff)
        .then(pl.lit(VolatilityRegime.HIGH.value))
        .otherwise(pl.lit(VolatilityRegime.NORMAL.value))
        .alias("vol_regime"),
        pl.when(pl.col("rolling_volatility") < low_cutoff)
        .then(-1)
        .when(pl.col("rolling_volatility") > high_cutoff)
        .then(1)
        .otherwise(0)
        .alias("vol_regime_numeric"),
    ])

    return df


def detect_trend_regime(
    df: pl.DataFrame,
    price_col: str = "close",
    fast_window: int = 10,
    slow_window: int = 30,
    adx_window: int = 14,
    adx_threshold: float = 25.0,
) -> pl.DataFrame:
    """Detect trend regime based on moving average relationships and ADX.

    Uses fast/slow moving average crossover and ADX strength to
    classify market as uptrend, downtrend, or ranging.

    Args:
        df: DataFrame containing price data.
        price_col: Name of price column. Defaults to "close".
        fast_window: Period for fast moving average. Defaults to 10.
        slow_window: Period for slow moving average. Defaults to 30.
        adx_window: Period for ADX calculation. Defaults to 14.
        adx_threshold: ADX level above which trend is considered strong.
            Defaults to 25.0.

    Returns:
        DataFrame with new columns:
        - ma_fast: Fast moving average
        - ma_slow: Slow moving average
        - trend_regime: Categorical regime label
        - trend_regime_numeric: Numeric regime (-1=down, 0=ranging, 1=up)

    Example:
        >>> df = pl.DataFrame({"close": list(range(100, 150))})
        >>> result = detect_trend_regime(df)
    """
    # Calculate moving averages
    df = df.with_columns([
        pl.col(price_col).rolling_mean(window_size=fast_window).alias("ma_fast"),
        pl.col(price_col).rolling_mean(window_size=slow_window).alias("ma_slow"),
    ])

    # Calculate price momentum
    df = df.with_columns(
        ((pl.col(price_col) / pl.col(price_col).shift(slow_window)) - 1).alias("_momentum")
    )

    # Approximate ADX using momentum volatility
    df = df.with_columns(
        pl.col("_momentum").abs().rolling_mean(window_size=adx_window).alias("_trend_strength")
    )

    # Normalize trend strength
    strength_series = df["_trend_strength"].drop_nulls()
    if strength_series.len() > 0:
        strength_mean = float(strength_series.mean())
        strength_scaled = adx_threshold / 100 * 2  # Scale factor
    else:
        strength_mean = 0.01
        strength_scaled = 0.02

    # Classify regime
    df = df.with_columns([
        pl.when(
            (pl.col("ma_fast") > pl.col("ma_slow")) &
            (pl.col("_trend_strength") > strength_scaled)
        )
        .then(pl.lit(TrendRegime.UPTREND.value))
        .when(
            (pl.col("ma_fast") < pl.col("ma_slow")) &
            (pl.col("_trend_strength") > strength_scaled)
        )
        .then(pl.lit(TrendRegime.DOWNTREND.value))
        .otherwise(pl.lit(TrendRegime.RANGING.value))
        .alias("trend_regime"),
        pl.when(
            (pl.col("ma_fast") > pl.col("ma_slow")) &
            (pl.col("_trend_strength") > strength_scaled)
        )
        .then(1)
        .when(
            (pl.col("ma_fast") < pl.col("ma_slow")) &
            (pl.col("_trend_strength") > strength_scaled)
        )
        .then(-1)
        .otherwise(0)
        .alias("trend_regime_numeric"),
    ])

    return df.drop(["_momentum", "_trend_strength"])


def detect_combined_regime(
    df: pl.DataFrame,
    price_col: str = "close",
    return_col: str = "return_1d",
) -> pl.DataFrame:
    """Detect combined market regime (trend + volatility).

    Combines trend and volatility regime detection to provide
    a comprehensive market state classification.

    Args:
        df: DataFrame containing price and return data.
        price_col: Name of price column. Defaults to "close".
        return_col: Name of return column. Defaults to "return_1d".

    Returns:
        DataFrame with combined regime information.

    Example:
        >>> df = pl.DataFrame({
        ...     "close": list(range(100, 150)),
        ...     "return_1d": np.random.randn(50) * 0.02
        ... })
        >>> result = detect_combined_regime(df)
    """
    # First detect volatility regime
    if "vol_regime" not in df.columns:
        df = detect_volatility_regime(df, return_col=return_col)

    # Then detect trend regime
    if "trend_regime" not in df.columns:
        df = detect_trend_regime(df, price_col=price_col)

    # Combine regimes
    df = df.with_columns(
        pl.when(
            (pl.col("trend_regime") == TrendRegime.UPTREND.value) &
            (pl.col("vol_regime") == VolatilityRegime.LOW.value)
        )
        .then(pl.lit(MarketRegime.BULL_QUIET.value))
        .when(
            (pl.col("trend_regime") == TrendRegime.UPTREND.value) &
            (pl.col("vol_regime") == VolatilityRegime.HIGH.value)
        )
        .then(pl.lit(MarketRegime.BULL_VOLATILE.value))
        .when(
            (pl.col("trend_regime") == TrendRegime.DOWNTREND.value) &
            (pl.col("vol_regime") == VolatilityRegime.LOW.value)
        )
        .then(pl.lit(MarketRegime.BEAR_QUIET.value))
        .when(
            (pl.col("trend_regime") == TrendRegime.DOWNTREND.value) &
            (pl.col("vol_regime") == VolatilityRegime.HIGH.value)
        )
        .then(pl.lit(MarketRegime.BEAR_VOLATILE.value))
        .when(pl.col("vol_regime") == VolatilityRegime.LOW.value)
        .then(pl.lit(MarketRegime.RANGING_QUIET.value))
        .otherwise(pl.lit(MarketRegime.RANGING_VOLATILE.value))
        .alias("market_regime")
    )

    return df


def regime_statistics(
    df: pl.DataFrame,
    regime_col: str = "market_regime",
    return_col: str = "return_1d",
) -> Dict[str, Dict[str, float]]:
    """Calculate performance statistics by regime.

    Provides return, volatility, and Sharpe statistics for each
    detected regime.

    Args:
        df: DataFrame with regime and return data.
        regime_col: Name of regime column. Defaults to "market_regime".
        return_col: Name of return column. Defaults to "return_1d".

    Returns:
        Dictionary mapping regime names to their statistics:
        - count: Number of periods
        - avg_return: Mean return
        - volatility: Standard deviation
        - sharpe: Sharpe ratio (annualized)
        - pct_time: Percentage of total time

    Example:
        >>> stats = regime_statistics(df)
        >>> for regime, metrics in stats.items():
        ...     print(f"{regime}: Sharpe={metrics['sharpe']:.2f}")
    """
    if regime_col not in df.columns or return_col not in df.columns:
        return {}

    total_count = df.height
    stats = {}

    # Get unique regimes
    regimes = df[regime_col].drop_nulls().unique().to_list()

    for regime in regimes:
        regime_df = df.filter(pl.col(regime_col) == regime)
        returns = regime_df[return_col].drop_nulls()

        if returns.len() == 0:
            continue

        count = regime_df.height
        avg_return = float(returns.mean())
        volatility = float(returns.std()) if returns.len() > 1 else 0.0
        sharpe = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0.0

        stats[str(regime)] = {
            "count": count,
            "avg_return": avg_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "pct_time": count / total_count if total_count > 0 else 0.0,
        }

    return stats


def regime_transition_matrix(
    df: pl.DataFrame,
    regime_col: str = "market_regime",
) -> pl.DataFrame:
    """Calculate regime transition probability matrix.

    Computes the probability of transitioning from one regime
    to another on the next period.

    Args:
        df: DataFrame with regime data.
        regime_col: Name of regime column. Defaults to "market_regime".

    Returns:
        DataFrame where rows are "from" regimes, columns are "to" regimes,
        and values are transition probabilities.

    Example:
        >>> trans_matrix = regime_transition_matrix(df)
        >>> print(trans_matrix)
    """
    if regime_col not in df.columns:
        return pl.DataFrame()

    # Get regime transitions
    df = df.with_columns(
        pl.col(regime_col).shift(-1).alias("_next_regime")
    )

    # Filter out nulls
    valid = df.filter(
        pl.col(regime_col).is_not_null() &
        pl.col("_next_regime").is_not_null()
    )

    if valid.height == 0:
        return pl.DataFrame()

    # Count transitions
    transition_counts = (
        valid.group_by([regime_col, "_next_regime"])
        .count()
        .rename({"count": "transitions"})
    )

    # Calculate totals per source regime
    totals = (
        transition_counts.group_by(regime_col)
        .agg(pl.col("transitions").sum().alias("total"))
    )

    # Join to get probabilities
    transitions = transition_counts.join(totals, on=regime_col)
    transitions = transitions.with_columns(
        (pl.col("transitions") / pl.col("total")).alias("probability")
    )

    # Pivot to matrix form
    regimes = transitions[regime_col].unique().sort().to_list()

    matrix_rows = []
    for from_regime in regimes:
        row = {"from_regime": from_regime}
        for to_regime in regimes:
            prob = transitions.filter(
                (pl.col(regime_col) == from_regime) &
                (pl.col("_next_regime") == to_regime)
            )
            if prob.height > 0:
                row[f"to_{to_regime}"] = float(prob["probability"][0])
            else:
                row[f"to_{to_regime}"] = 0.0
        matrix_rows.append(row)

    return pl.DataFrame(matrix_rows)


def estimate_regime_persistence(
    df: pl.DataFrame,
    regime_col: str = "market_regime",
) -> Dict[str, float]:
    """Estimate how long each regime persists on average.

    Calculates the average number of consecutive periods
    spent in each regime state.

    Args:
        df: DataFrame with regime data.
        regime_col: Name of regime column. Defaults to "market_regime".

    Returns:
        Dictionary mapping regime names to average persistence (periods).

    Example:
        >>> persistence = estimate_regime_persistence(df)
        >>> print(f"Bull quiet lasts {persistence['bull_quiet']:.1f} periods avg")
    """
    if regime_col not in df.columns:
        return {}

    regimes = df[regime_col].to_list()
    n = len(regimes)

    if n == 0:
        return {}

    # Track runs of each regime
    regime_runs: Dict[str, List[int]] = {}
    current_regime = regimes[0]
    current_run = 1

    for i in range(1, n):
        if regimes[i] == current_regime:
            current_run += 1
        else:
            if current_regime is not None:
                if current_regime not in regime_runs:
                    regime_runs[current_regime] = []
                regime_runs[current_regime].append(current_run)
            current_regime = regimes[i]
            current_run = 1

    # Add final run
    if current_regime is not None:
        if current_regime not in regime_runs:
            regime_runs[current_regime] = []
        regime_runs[current_regime].append(current_run)

    # Calculate average persistence
    persistence = {}
    for regime, runs in regime_runs.items():
        persistence[str(regime)] = float(np.mean(runs)) if runs else 0.0

    return persistence
