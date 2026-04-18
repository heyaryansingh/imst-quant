"""Liquidity analysis utilities for trading strategy development.

This module provides tools for analyzing market liquidity including bid-ask
spreads, volume analysis, market impact estimation, and liquidity scoring.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.liquidity_analysis import (
    ...     calculate_amihud_illiquidity,
    ...     calculate_volume_profile,
    ...     estimate_market_impact,
    ... )
    >>> df = pl.read_parquet("ohlcv.parquet")
    >>> illiquidity = calculate_amihud_illiquidity(df)
    >>> impact = estimate_market_impact(df, order_size=10000)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class LiquidityMetrics:
    """Container for liquidity analysis results.

    Attributes:
        avg_daily_volume: Average daily trading volume.
        avg_dollar_volume: Average daily dollar volume.
        volume_volatility: Standard deviation of daily volume.
        amihud_illiquidity: Amihud illiquidity ratio.
        avg_spread_estimate: Estimated average bid-ask spread.
        volume_concentration: Volume concentration ratio (top 20% of days).
        liquidity_score: Composite liquidity score (0-100).
        days_analyzed: Number of trading days in the analysis.
    """

    avg_daily_volume: float
    avg_dollar_volume: float
    volume_volatility: float
    amihud_illiquidity: float
    avg_spread_estimate: float
    volume_concentration: float
    liquidity_score: float
    days_analyzed: int


def calculate_amihud_illiquidity(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    volume_col: str = "volume",
    price_col: str = "close",
) -> float:
    """Calculate Amihud illiquidity ratio.

    The Amihud ratio measures price impact per unit of trading volume.
    Higher values indicate less liquid markets.

    Args:
        df: DataFrame with returns, volume, and price data.
        return_col: Column name for daily returns.
        volume_col: Column name for trading volume.
        price_col: Column name for closing prices.

    Returns:
        Amihud illiquidity ratio (average |return| / dollar volume).

    Example:
        >>> illiquidity = calculate_amihud_illiquidity(df)
        >>> print(f"Amihud ratio: {illiquidity:.6e}")
    """
    if return_col not in df.columns or volume_col not in df.columns:
        return 0.0

    # Calculate dollar volume
    if price_col in df.columns:
        df = df.with_columns(
            (pl.col(volume_col) * pl.col(price_col)).alias("dollar_volume")
        )
    else:
        df = df.with_columns(pl.col(volume_col).alias("dollar_volume"))

    # Filter valid observations
    valid = df.filter(
        pl.col("dollar_volume").is_not_null()
        & pl.col("dollar_volume").gt(0)
        & pl.col(return_col).is_not_null()
    )

    if valid.height == 0:
        return 0.0

    # Amihud = mean(|return| / dollar_volume)
    amihud = valid.select(
        (pl.col(return_col).abs() / pl.col("dollar_volume")).mean()
    ).item()

    return float(amihud) if amihud else 0.0


def calculate_roll_spread(
    df: pl.DataFrame,
    return_col: str = "return_1d",
) -> float:
    """Estimate bid-ask spread using Roll's measure.

    Roll's spread estimate uses the autocovariance of returns to infer
    the effective bid-ask spread.

    Args:
        df: DataFrame with return data.
        return_col: Column name for returns.

    Returns:
        Estimated bid-ask spread as a decimal (e.g., 0.001 = 0.1%).

    Example:
        >>> spread = calculate_roll_spread(df)
        >>> print(f"Estimated spread: {spread:.4%}")
    """
    if return_col not in df.columns:
        return 0.0

    returns = df[return_col].drop_nulls()

    if returns.len() < 3:
        return 0.0

    # Calculate first-order autocovariance
    returns_list = returns.to_list()
    r_t = returns_list[1:]
    r_t1 = returns_list[:-1]

    n = len(r_t)
    mean_r = sum(returns_list) / len(returns_list)

    cov = sum((r_t[i] - mean_r) * (r_t1[i] - mean_r) for i in range(n)) / n

    # Roll's spread = 2 * sqrt(-cov) if cov < 0
    if cov < 0:
        import math
        return 2 * math.sqrt(-cov)
    return 0.0


def calculate_volume_profile(
    df: pl.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    n_buckets: int = 20,
) -> pl.DataFrame:
    """Calculate volume profile by price level.

    Groups volume into price buckets to identify high-volume nodes
    where liquidity concentrates.

    Args:
        df: DataFrame with price and volume data.
        price_col: Column name for price.
        volume_col: Column name for volume.
        n_buckets: Number of price buckets.

    Returns:
        DataFrame with price_bucket, volume, pct_volume columns.

    Example:
        >>> profile = calculate_volume_profile(df)
        >>> print(profile.head())
    """
    if price_col not in df.columns or volume_col not in df.columns:
        return pl.DataFrame()

    valid = df.filter(
        pl.col(price_col).is_not_null() & pl.col(volume_col).is_not_null()
    )

    if valid.height == 0:
        return pl.DataFrame()

    price_min = valid[price_col].min()
    price_max = valid[price_col].max()

    if price_min is None or price_max is None or price_min == price_max:
        return pl.DataFrame()

    bucket_size = (price_max - price_min) / n_buckets

    # Add bucket assignment
    profile = valid.with_columns(
        ((pl.col(price_col) - price_min) / bucket_size)
        .floor()
        .clip(0, n_buckets - 1)
        .cast(pl.Int32)
        .alias("bucket")
    )

    # Aggregate by bucket
    result = (
        profile.group_by("bucket")
        .agg(
            pl.col(volume_col).sum().alias("volume"),
            ((pl.col("bucket") * bucket_size) + price_min + bucket_size / 2)
            .first()
            .alias("price_midpoint"),
        )
        .sort("bucket")
    )

    total_volume = result["volume"].sum()
    result = result.with_columns(
        (pl.col("volume") / total_volume * 100).alias("pct_volume")
    )

    return result


def calculate_intraday_volume_pattern(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Analyze intraday volume patterns by hour.

    Identifies when liquidity is highest during the trading day.

    Args:
        df: DataFrame with timestamp and volume data.
        time_col: Column name for timestamp.
        volume_col: Column name for volume.

    Returns:
        DataFrame with hour, avg_volume, pct_total columns.

    Example:
        >>> pattern = calculate_intraday_volume_pattern(df)
        >>> peak_hour = pattern.filter(pl.col("pct_total") == pl.col("pct_total").max())
    """
    if time_col not in df.columns or volume_col not in df.columns:
        return pl.DataFrame()

    # Extract hour from timestamp
    result = (
        df.with_columns(pl.col(time_col).dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.col(volume_col).mean().alias("avg_volume"),
            pl.col(volume_col).sum().alias("total_volume"),
        )
        .sort("hour")
    )

    total = result["total_volume"].sum()
    result = result.with_columns(
        (pl.col("total_volume") / total * 100).alias("pct_total")
    )

    return result


def estimate_market_impact(
    df: pl.DataFrame,
    order_size: float,
    volume_col: str = "volume",
    price_col: str = "close",
    impact_coefficient: float = 0.1,
) -> dict[str, float]:
    """Estimate market impact of a hypothetical order.

    Uses a square-root market impact model to estimate price slippage.

    Args:
        df: DataFrame with volume and price data.
        order_size: Size of order in shares.
        volume_col: Column name for volume.
        price_col: Column name for price.
        impact_coefficient: Market impact coefficient (default 0.1).

    Returns:
        Dictionary with impact estimates.

    Example:
        >>> impact = estimate_market_impact(df, order_size=10000)
        >>> print(f"Expected slippage: {impact['expected_slippage_pct']:.2%}")
    """
    if volume_col not in df.columns:
        return {
            "order_size": order_size,
            "adv": 0.0,
            "participation_rate": 0.0,
            "expected_slippage_pct": 0.0,
            "expected_slippage_bps": 0.0,
        }

    # Average daily volume
    adv = df[volume_col].mean()
    if adv is None or adv == 0:
        return {
            "order_size": order_size,
            "adv": 0.0,
            "participation_rate": 0.0,
            "expected_slippage_pct": 0.0,
            "expected_slippage_bps": 0.0,
        }

    adv = float(adv)
    participation_rate = order_size / adv

    # Square-root market impact model
    # Impact = coefficient * sqrt(participation_rate) * volatility
    import math

    # Estimate volatility from returns if available
    if "return_1d" in df.columns:
        returns = df["return_1d"].drop_nulls()
        volatility = float(returns.std()) if returns.len() > 0 else 0.02
    else:
        volatility = 0.02  # Default 2% daily vol

    slippage_pct = impact_coefficient * math.sqrt(participation_rate) * volatility
    slippage_bps = slippage_pct * 10000

    return {
        "order_size": order_size,
        "adv": adv,
        "participation_rate": participation_rate,
        "expected_slippage_pct": slippage_pct,
        "expected_slippage_bps": slippage_bps,
        "volatility": volatility,
    }


def calculate_volume_concentration(
    df: pl.DataFrame,
    volume_col: str = "volume",
    top_pct: float = 0.2,
) -> float:
    """Calculate volume concentration ratio.

    Measures what percentage of total volume occurs on the top N% of days
    by volume. Higher concentration indicates less consistent liquidity.

    Args:
        df: DataFrame with volume data.
        volume_col: Column name for volume.
        top_pct: Percentage of top days to consider (e.g., 0.2 = top 20%).

    Returns:
        Concentration ratio (0-1, where 0.2 means top 20% = 20% of volume).

    Example:
        >>> concentration = calculate_volume_concentration(df)
        >>> print(f"Top 20% days = {concentration:.1%} of volume")
    """
    if volume_col not in df.columns:
        return 0.0

    volumes = df[volume_col].drop_nulls()
    if volumes.len() == 0:
        return 0.0

    sorted_volumes = volumes.sort(descending=True)
    total_volume = sorted_volumes.sum()

    if total_volume == 0:
        return 0.0

    top_n = max(1, int(volumes.len() * top_pct))
    top_volume = sorted_volumes.head(top_n).sum()

    return float(top_volume / total_volume)


def calculate_relative_volume(
    df: pl.DataFrame,
    volume_col: str = "volume",
    lookback: int = 20,
) -> pl.DataFrame:
    """Calculate relative volume (RVOL) compared to average.

    RVOL > 1 indicates higher than normal volume, potentially signaling
    increased institutional activity.

    Args:
        df: DataFrame with volume data.
        volume_col: Column name for volume.
        lookback: Number of periods for average calculation.

    Returns:
        DataFrame with added 'rvol' column.

    Example:
        >>> df_with_rvol = calculate_relative_volume(df)
        >>> high_volume_days = df_with_rvol.filter(pl.col("rvol") > 2)
    """
    if volume_col not in df.columns:
        return df

    return df.with_columns(
        (
            pl.col(volume_col)
            / pl.col(volume_col).rolling_mean(window_size=lookback)
        ).alias("rvol")
    )


def analyze_liquidity(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    volume_col: str = "volume",
    price_col: str = "close",
) -> LiquidityMetrics:
    """Comprehensive liquidity analysis.

    Calculates multiple liquidity metrics and combines them into
    a single composite score.

    Args:
        df: DataFrame with price, volume, and return data.
        return_col: Column name for returns.
        volume_col: Column name for volume.
        price_col: Column name for price.

    Returns:
        LiquidityMetrics with all calculated metrics.

    Example:
        >>> metrics = analyze_liquidity(df)
        >>> print(f"Liquidity score: {metrics.liquidity_score:.1f}/100")
    """
    # Basic volume metrics
    avg_volume = float(df[volume_col].mean()) if volume_col in df.columns else 0.0
    volume_std = float(df[volume_col].std()) if volume_col in df.columns else 0.0
    volume_volatility = volume_std / avg_volume if avg_volume > 0 else 0.0

    # Dollar volume
    if volume_col in df.columns and price_col in df.columns:
        dollar_vol = df.select(
            (pl.col(volume_col) * pl.col(price_col)).mean()
        ).item()
        avg_dollar_volume = float(dollar_vol) if dollar_vol else 0.0
    else:
        avg_dollar_volume = avg_volume

    # Amihud illiquidity
    amihud = calculate_amihud_illiquidity(df, return_col, volume_col, price_col)

    # Roll spread estimate
    spread_estimate = calculate_roll_spread(df, return_col)

    # Volume concentration
    concentration = calculate_volume_concentration(df, volume_col)

    # Calculate composite liquidity score (0-100)
    # Lower is worse for volume, higher is worse for illiquidity/spread/concentration
    import math

    # Normalize metrics to 0-1 scale (inverse for "bad" metrics)
    vol_score = min(1.0, math.log10(avg_dollar_volume + 1) / 10) if avg_dollar_volume > 0 else 0
    amihud_score = max(0, 1 - min(1, amihud * 1e6))  # Scale Amihud
    spread_score = max(0, 1 - min(1, spread_estimate * 100))  # Scale spread
    concentration_score = max(0, 1 - (concentration - 0.2) * 2)  # Deviation from ideal

    # Weight the components
    liquidity_score = (
        vol_score * 40  # 40% weight on volume
        + amihud_score * 25  # 25% on Amihud
        + spread_score * 20  # 20% on spread
        + concentration_score * 15  # 15% on concentration
    )

    return LiquidityMetrics(
        avg_daily_volume=avg_volume,
        avg_dollar_volume=avg_dollar_volume,
        volume_volatility=volume_volatility,
        amihud_illiquidity=amihud,
        avg_spread_estimate=spread_estimate,
        volume_concentration=concentration,
        liquidity_score=min(100, max(0, liquidity_score)),
        days_analyzed=df.height,
    )


def find_illiquid_periods(
    df: pl.DataFrame,
    volume_col: str = "volume",
    threshold_pct: float = 0.3,
    date_col: str = "date",
) -> pl.DataFrame:
    """Identify periods of unusually low liquidity.

    Finds dates where volume was significantly below average,
    which may indicate periods to avoid trading.

    Args:
        df: DataFrame with volume and date data.
        volume_col: Column name for volume.
        threshold_pct: Threshold as percentage of average (e.g., 0.3 = 30%).
        date_col: Column name for date.

    Returns:
        DataFrame with illiquid periods and their characteristics.

    Example:
        >>> illiquid = find_illiquid_periods(df)
        >>> print(f"Found {illiquid.height} illiquid days")
    """
    if volume_col not in df.columns:
        return pl.DataFrame()

    avg_volume = df[volume_col].mean()
    threshold = avg_volume * threshold_pct

    illiquid = df.filter(pl.col(volume_col) < threshold)

    if date_col in illiquid.columns:
        illiquid = illiquid.select(
            date_col,
            volume_col,
            (pl.col(volume_col) / avg_volume * 100).alias("pct_of_avg"),
        )

    return illiquid


def calculate_vwap_deviation(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Calculate VWAP and price deviation from VWAP.

    VWAP is a common benchmark for institutional execution quality.

    Args:
        df: DataFrame with OHLCV data.
        high_col: Column name for high price.
        low_col: Column name for low price.
        close_col: Column name for close price.
        volume_col: Column name for volume.

    Returns:
        DataFrame with vwap and vwap_deviation columns added.

    Example:
        >>> df = calculate_vwap_deviation(df)
        >>> above_vwap = df.filter(pl.col("vwap_deviation") > 0)
    """
    required = [high_col, low_col, close_col, volume_col]
    if not all(col in df.columns for col in required):
        return df

    # Typical price = (H + L + C) / 3
    df = df.with_columns(
        ((pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3).alias("typical_price")
    )

    # VWAP = cumsum(typical_price * volume) / cumsum(volume)
    df = df.with_columns(
        (
            (pl.col("typical_price") * pl.col(volume_col)).cum_sum()
            / pl.col(volume_col).cum_sum()
        ).alias("vwap")
    )

    # Deviation from VWAP
    df = df.with_columns(
        ((pl.col(close_col) - pl.col("vwap")) / pl.col("vwap")).alias("vwap_deviation")
    )

    return df.drop("typical_price")
