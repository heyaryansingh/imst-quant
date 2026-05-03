"""Rolling window utilities for time series feature engineering.

Provides optimized rolling window calculations for financial time series,
including exponential moving averages, volatility, and custom aggregations.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.rolling_windows import add_rolling_stats
    >>> df = pl.DataFrame({"price": [100, 102, 101, 105, 108]})
    >>> df = add_rolling_stats(df, "price", windows=[3, 5])
"""

import polars as pl
from typing import List, Optional, Union


def add_rolling_stats(
    df: pl.DataFrame,
    col: str,
    windows: List[int],
    stats: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Add rolling statistical features for specified windows.

    Args:
        df: Input DataFrame with time series data.
        col: Column name to compute rolling statistics on.
        windows: List of window sizes (e.g., [5, 10, 20]).
        stats: Statistics to compute. Options: "mean", "std", "min", "max", "median".
               Defaults to ["mean", "std"].

    Returns:
        DataFrame with added rolling statistic columns named {col}_roll{window}_{stat}.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 101, 105, 108, 106, 110]})
        >>> df = add_rolling_stats(df, "price", windows=[3], stats=["mean", "std"])
        >>> print(df.columns)
        ['price', 'price_roll3_mean', 'price_roll3_std']
    """
    if stats is None:
        stats = ["mean", "std"]

    result = df.clone()

    for window in windows:
        for stat in stats:
            col_name = f"{col}_roll{window}_{stat}"

            if stat == "mean":
                result = result.with_columns(
                    pl.col(col).rolling_mean(window_size=window).alias(col_name)
                )
            elif stat == "std":
                result = result.with_columns(
                    pl.col(col).rolling_std(window_size=window).alias(col_name)
                )
            elif stat == "min":
                result = result.with_columns(
                    pl.col(col).rolling_min(window_size=window).alias(col_name)
                )
            elif stat == "max":
                result = result.with_columns(
                    pl.col(col).rolling_max(window_size=window).alias(col_name)
                )
            elif stat == "median":
                result = result.with_columns(
                    pl.col(col).rolling_median(window_size=window).alias(col_name)
                )
            else:
                raise ValueError(f"Unsupported statistic: {stat}")

    return result


def add_ema(
    df: pl.DataFrame,
    col: str,
    spans: List[int],
    adjust: bool = True,
) -> pl.DataFrame:
    """Add exponential moving averages with specified spans.

    Args:
        df: Input DataFrame.
        col: Column to compute EMA on.
        spans: List of EMA spans (e.g., [12, 26, 50]).
        adjust: Whether to use adjusted EMA (default: True).

    Returns:
        DataFrame with EMA columns named {col}_ema{span}.

    Example:
        >>> df = pl.DataFrame({"price": list(range(100, 150))})
        >>> df = add_ema(df, "price", spans=[12, 26])
        >>> print(df.columns)
        ['price', 'price_ema12', 'price_ema26']
    """
    result = df.clone()

    for span in spans:
        col_name = f"{col}_ema{span}"
        result = result.with_columns(
            pl.col(col).ewm_mean(span=span, adjust=adjust).alias(col_name)
        )

    return result


def add_rolling_volatility(
    df: pl.DataFrame,
    returns_col: str = "returns",
    windows: List[int] = [5, 10, 20],
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Add rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame containing returns.
        returns_col: Column name for returns (default: "returns").
        windows: List of window sizes for volatility calculation.
        annualize: Whether to annualize volatility (default: True).
        periods_per_year: Number of periods per year for annualization (default: 252 for daily).

    Returns:
        DataFrame with volatility columns named vol{window}.

    Example:
        >>> df = pl.DataFrame({"returns": [0.01, -0.02, 0.015, 0.005, -0.01]})
        >>> df = add_rolling_volatility(df, windows=[3], annualize=False)
        >>> print(df["vol3"])
    """
    result = df.clone()

    for window in windows:
        col_name = f"vol{window}"
        vol_expr = pl.col(returns_col).rolling_std(window_size=window)

        if annualize:
            vol_expr = vol_expr * (periods_per_year ** 0.5)

        result = result.with_columns(vol_expr.alias(col_name))

    return result


def add_bollinger_bands(
    df: pl.DataFrame,
    price_col: str = "close",
    window: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Add Bollinger Bands (moving average ± std dev bands).

    Args:
        df: Input DataFrame.
        price_col: Price column name (default: "close").
        window: Rolling window size (default: 20).
        num_std: Number of standard deviations for bands (default: 2.0).

    Returns:
        DataFrame with columns: bb_middle, bb_upper, bb_lower, bb_width.

    Example:
        >>> df = pl.DataFrame({"close": list(range(100, 120))})
        >>> df = add_bollinger_bands(df, window=5)
        >>> print(df[["bb_upper", "bb_middle", "bb_lower"]])
    """
    result = df.with_columns([
        pl.col(price_col).rolling_mean(window_size=window).alias("bb_middle"),
        pl.col(price_col).rolling_std(window_size=window).alias("bb_std"),
    ])

    result = result.with_columns([
        (pl.col("bb_middle") + num_std * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - num_std * pl.col("bb_std")).alias("bb_lower"),
    ])

    result = result.with_columns(
        (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width")
    )

    return result.drop("bb_std")


def add_rolling_zscore(
    df: pl.DataFrame,
    col: str,
    window: int = 20,
) -> pl.DataFrame:
    """Add rolling z-score (standardized value within window).

    Args:
        df: Input DataFrame.
        col: Column to compute z-score for.
        window: Rolling window size (default: 20).

    Returns:
        DataFrame with column {col}_zscore{window}.

    Example:
        >>> df = pl.DataFrame({"price": [100, 105, 102, 110, 95, 98]})
        >>> df = add_rolling_zscore(df, "price", window=3)
        >>> print(df[["price", "price_zscore3"]])
    """
    col_name = f"{col}_zscore{window}"

    result = df.with_columns([
        pl.col(col).rolling_mean(window_size=window).alias("_roll_mean"),
        pl.col(col).rolling_std(window_size=window).alias("_roll_std"),
    ])

    result = result.with_columns(
        ((pl.col(col) - pl.col("_roll_mean")) / pl.col("_roll_std")).alias(col_name)
    )

    return result.drop(["_roll_mean", "_roll_std"])


def add_momentum_indicators(
    df: pl.DataFrame,
    price_col: str = "close",
    periods: List[int] = [5, 10, 20],
) -> pl.DataFrame:
    """Add momentum indicators (price change over N periods).

    Args:
        df: Input DataFrame.
        price_col: Price column name (default: "close").
        periods: List of lookback periods (e.g., [5, 10, 20]).

    Returns:
        DataFrame with momentum columns named mom{period}.

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 105, 103, 108, 110]})
        >>> df = add_momentum_indicators(df, periods=[3])
        >>> print(df[["close", "mom3"]])
    """
    result = df.clone()

    for period in periods:
        col_name = f"mom{period}"
        result = result.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(period) - 1.0).alias(col_name)
        )

    return result


def add_rsi(
    df: pl.DataFrame,
    price_col: str = "close",
    period: int = 14,
) -> pl.DataFrame:
    """Add Relative Strength Index (RSI).

    Args:
        df: Input DataFrame.
        price_col: Price column name (default: "close").
        period: RSI period (default: 14).

    Returns:
        DataFrame with rsi column.

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 108, 106, 110, 109]})
        >>> df = add_rsi(df, period=6)
        >>> print(df["rsi"])
    """
    delta = df.select(pl.col(price_col).diff().alias("delta"))["delta"]

    gain = delta.clip_min(0)
    loss = (-delta).clip_min(0)

    avg_gain = gain.ewm_mean(span=period, adjust=False)
    avg_loss = loss.ewm_mean(span=period, adjust=False)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return df.with_columns(rsi.alias("rsi"))
