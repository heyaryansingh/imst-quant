"""Order flow analysis utilities for microstructure-based trading signals.

Provides functions to analyze trade flow imbalances, volume patterns,
and order book dynamics to detect short-term price pressure.

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Simulate tick data
    >>> prices = pd.Series([100.0, 100.05, 99.95, 100.10, 100.15])
    >>> volumes = pd.Series([1000, 1500, 800, 2000, 1200])
    >>> # Calculate volume imbalance
    >>> vimb = volume_imbalance(prices, volumes)
    >>> # Calculate Order Flow Imbalance
    >>> ofi = order_flow_imbalance(prices, volumes, window=3)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OrderFlowMetrics:
    """Container for order flow analysis results.

    Attributes:
        volume_imbalance: Buy vs sell volume imbalance ratio.
        ofi: Order Flow Imbalance indicator.
        vpin: Volume-Synchronized Probability of Informed Trading.
        trade_flow_toxicity: Measure of adverse selection risk.
        buy_volume_pct: Percentage of volume classified as buys.
        sell_volume_pct: Percentage of volume classified as sells.
        large_trade_ratio: Ratio of large trades to total trades.
        aggressor_imbalance: Net aggressor side imbalance.
    """

    volume_imbalance: float
    ofi: float
    vpin: float
    trade_flow_toxicity: float
    buy_volume_pct: float
    sell_volume_pct: float
    large_trade_ratio: float
    aggressor_imbalance: float


def classify_trades(
    prices: pd.Series,
    method: str = "tick",
) -> pd.Series:
    """Classify trades as buyer or seller initiated using tick rule.

    The tick rule classifies a trade based on price movement:
    - If price > previous price: buyer initiated (+1)
    - If price < previous price: seller initiated (-1)
    - If price == previous price: same as previous classification

    Args:
        prices: Series of trade prices.
        method: Classification method ("tick" supported).

    Returns:
        Series with +1 for buys, -1 for sells.

    Example:
        >>> prices = pd.Series([100.0, 100.05, 99.95, 100.10])
        >>> signs = classify_trades(prices)
        >>> signs.values  # [0, 1, -1, 1]
    """
    if method != "tick":
        raise ValueError(f"Unknown method: {method}. Use 'tick'.")

    price_changes = prices.diff()
    signs = np.sign(price_changes)

    # Forward fill zeros (unchanged prices inherit previous direction)
    signs = signs.replace(0, np.nan).ffill().fillna(0)

    return signs.astype(int)


def volume_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate rolling volume imbalance (buy volume - sell volume) / total.

    Volume imbalance measures the relative pressure between buyers and sellers.
    Positive values indicate buying pressure, negative indicates selling pressure.

    Args:
        prices: Series of trade prices for trade classification.
        volumes: Series of trade volumes.
        window: Rolling window size for imbalance calculation.

    Returns:
        Series of volume imbalance values in range [-1, 1].

    Example:
        >>> prices = pd.Series([100.0, 100.05, 99.95, 100.10, 100.15])
        >>> volumes = pd.Series([1000, 1500, 800, 2000, 1200])
        >>> vimb = volume_imbalance(prices, volumes, window=3)
    """
    signs = classify_trades(prices)
    signed_volume = volumes * signs

    buy_vol = signed_volume.clip(lower=0).rolling(window).sum()
    sell_vol = (-signed_volume.clip(upper=0)).rolling(window).sum()
    total_vol = buy_vol + sell_vol

    imbalance = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
    return imbalance.fillna(0)


def order_flow_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 20,
    normalize: bool = True,
) -> pd.Series:
    """Calculate Order Flow Imbalance (OFI) indicator.

    OFI aggregates signed trade volume to measure net order flow.
    It captures the cumulative buying/selling pressure over a window.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        window: Rolling window for OFI calculation.
        normalize: If True, normalize by rolling volume for comparability.

    Returns:
        Series of OFI values.

    Example:
        >>> prices = pd.Series([100, 101, 100.5, 101.5, 102])
        >>> volumes = pd.Series([100, 150, 80, 200, 120])
        >>> ofi = order_flow_imbalance(prices, volumes, window=3)
    """
    signs = classify_trades(prices)
    signed_volume = volumes * signs

    ofi = signed_volume.rolling(window).sum()

    if normalize:
        total_vol = volumes.abs().rolling(window).sum()
        ofi = ofi / total_vol.replace(0, np.nan)

    return ofi.fillna(0)


def calculate_vpin(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float = 10000.0,
    num_buckets: int = 50,
) -> pd.Series:
    """Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN estimates the probability of informed trading by measuring
    order flow toxicity in volume-time. Higher VPIN indicates higher
    probability of informed trading and potential adverse selection.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        bucket_size: Target volume per bucket.
        num_buckets: Number of buckets for VPIN calculation.

    Returns:
        Series of VPIN values in range [0, 1].

    Note:
        VPIN was developed by Easley, Lopez de Prado, and O'Hara (2012).
        Higher values (>0.4) may indicate elevated informed trading risk.
    """
    signs = classify_trades(prices)
    signed_volume = volumes * signs

    # Create volume buckets
    cumulative_vol = volumes.cumsum()
    bucket_ids = (cumulative_vol // bucket_size).astype(int)

    # Calculate buy/sell volume per bucket
    df = pd.DataFrame({
        "bucket": bucket_ids,
        "buy_vol": signed_volume.clip(lower=0),
        "sell_vol": (-signed_volume.clip(upper=0)),
    })

    bucket_agg = df.groupby("bucket").agg({
        "buy_vol": "sum",
        "sell_vol": "sum",
    })

    # VPIN = rolling mean of |buy - sell| / total
    imbalance = (bucket_agg["buy_vol"] - bucket_agg["sell_vol"]).abs()
    total = bucket_agg["buy_vol"] + bucket_agg["sell_vol"]

    vpin_per_bucket = (imbalance / total.replace(0, np.nan)).rolling(
        num_buckets, min_periods=1
    ).mean()

    # Map back to original index
    result = bucket_ids.map(vpin_per_bucket.to_dict())
    return result.fillna(0)


def trade_flow_toxicity(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 50,
) -> pd.Series:
    """Calculate trade flow toxicity indicator.

    Measures adverse selection risk by comparing price impact to volume.
    High toxicity indicates that trades are moving prices significantly
    relative to their size, suggesting informed trading.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        window: Rolling window for toxicity calculation.

    Returns:
        Series of toxicity values (higher = more toxic flow).

    Example:
        >>> prices = pd.Series([100, 100.5, 100.3, 101, 100.8])
        >>> volumes = pd.Series([1000, 500, 800, 300, 600])
        >>> toxicity = trade_flow_toxicity(prices, volumes, window=3)
    """
    returns = prices.pct_change()
    signs = classify_trades(prices)
    signed_volume = volumes * signs

    # Calculate price impact per unit volume
    abs_returns = returns.abs()
    abs_volume = signed_volume.abs().replace(0, np.nan)

    impact = abs_returns / abs_volume

    # Normalize by rolling average
    mean_impact = impact.rolling(window, min_periods=1).mean()
    toxicity = impact / mean_impact.replace(0, np.nan)

    return toxicity.fillna(1.0).clip(0, 10)


def detect_large_trades(
    volumes: pd.Series,
    threshold_percentile: float = 95,
    window: int = 100,
) -> Tuple[pd.Series, pd.Series]:
    """Detect large trades that may indicate institutional activity.

    Identifies trades with volume above a rolling percentile threshold.
    Large trades can signal institutional order flow or block trades.

    Args:
        volumes: Series of trade volumes.
        threshold_percentile: Percentile threshold for "large" classification.
        window: Rolling window for percentile calculation.

    Returns:
        Tuple of (is_large_trade boolean series, rolling threshold series).

    Example:
        >>> volumes = pd.Series([100, 150, 80, 500, 120, 90, 1000])
        >>> is_large, threshold = detect_large_trades(volumes, 90, window=5)
    """
    threshold = volumes.rolling(window, min_periods=1).quantile(
        threshold_percentile / 100
    )
    is_large = volumes > threshold

    return is_large, threshold


def aggressor_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate aggressor side imbalance.

    Measures the net direction of aggressive orders (market orders that
    cross the spread). Persistent imbalance indicates directional pressure.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        window: Rolling window for imbalance calculation.

    Returns:
        Series of aggressor imbalance values in range [-1, 1].
    """
    signs = classify_trades(prices)

    # Count trades by direction
    buy_count = (signs > 0).rolling(window).sum()
    sell_count = (signs < 0).rolling(window).sum()
    total_count = buy_count + sell_count

    imbalance = (buy_count - sell_count) / total_count.replace(0, np.nan)
    return imbalance.fillna(0)


def analyze_order_flow(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 50,
    vpin_bucket_size: float = 10000.0,
) -> OrderFlowMetrics:
    """Comprehensive order flow analysis with multiple indicators.

    Calculates all order flow metrics for a given price/volume dataset.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        window: Rolling window for calculations.
        vpin_bucket_size: Volume per bucket for VPIN calculation.

    Returns:
        OrderFlowMetrics containing all computed indicators.

    Example:
        >>> prices = pd.Series(np.random.randn(100).cumsum() + 100)
        >>> volumes = pd.Series(np.random.randint(100, 1000, 100))
        >>> metrics = analyze_order_flow(prices, volumes)
        >>> print(f"VPIN: {metrics.vpin:.3f}")
    """
    signs = classify_trades(prices)

    # Volume classification
    buy_mask = signs > 0
    sell_mask = signs < 0
    total_vol = volumes.sum()

    buy_vol_pct = volumes[buy_mask].sum() / total_vol if total_vol > 0 else 0.5
    sell_vol_pct = volumes[sell_mask].sum() / total_vol if total_vol > 0 else 0.5

    # Volume imbalance (final value)
    vimb = volume_imbalance(prices, volumes, window)
    vimb_final = vimb.iloc[-1] if len(vimb) > 0 else 0.0

    # OFI
    ofi = order_flow_imbalance(prices, volumes, window)
    ofi_final = ofi.iloc[-1] if len(ofi) > 0 else 0.0

    # VPIN
    vpin = calculate_vpin(prices, volumes, vpin_bucket_size)
    vpin_final = vpin.iloc[-1] if len(vpin) > 0 else 0.0

    # Toxicity
    toxicity = trade_flow_toxicity(prices, volumes, window)
    toxicity_final = toxicity.iloc[-1] if len(toxicity) > 0 else 1.0

    # Large trades
    is_large, _ = detect_large_trades(volumes)
    large_ratio = is_large.mean() if len(is_large) > 0 else 0.05

    # Aggressor imbalance
    agg_imb = aggressor_imbalance(prices, volumes, window)
    agg_imb_final = agg_imb.iloc[-1] if len(agg_imb) > 0 else 0.0

    return OrderFlowMetrics(
        volume_imbalance=float(vimb_final),
        ofi=float(ofi_final),
        vpin=float(vpin_final),
        trade_flow_toxicity=float(toxicity_final),
        buy_volume_pct=float(buy_vol_pct),
        sell_volume_pct=float(sell_vol_pct),
        large_trade_ratio=float(large_ratio),
        aggressor_imbalance=float(agg_imb_final),
    )


def order_flow_momentum(
    prices: pd.Series,
    volumes: pd.Series,
    fast_window: int = 10,
    slow_window: int = 50,
) -> pd.Series:
    """Calculate order flow momentum indicator.

    Compares short-term vs long-term order flow imbalance.
    Positive values indicate accelerating buying pressure.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        fast_window: Short-term window.
        slow_window: Long-term window.

    Returns:
        Series of momentum values.
    """
    fast_ofi = order_flow_imbalance(prices, volumes, fast_window)
    slow_ofi = order_flow_imbalance(prices, volumes, slow_window)

    momentum = fast_ofi - slow_ofi
    return momentum


def volume_clock_bars(
    prices: pd.Series,
    volumes: pd.Series,
    bar_volume: float = 10000.0,
) -> pd.DataFrame:
    """Convert tick data to volume clock bars.

    Creates bars based on volume rather than time, which adjusts
    for varying market activity and provides more uniform information
    content per bar.

    Args:
        prices: Series of trade prices.
        volumes: Series of trade volumes.
        bar_volume: Target volume per bar.

    Returns:
        DataFrame with OHLCV bars indexed by bar number.
    """
    cumulative_vol = volumes.cumsum()
    bar_ids = (cumulative_vol // bar_volume).astype(int)

    df = pd.DataFrame({
        "bar_id": bar_ids,
        "price": prices,
        "volume": volumes,
    })

    bars = df.groupby("bar_id").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum",
    })

    bars.columns = ["open", "high", "low", "close", "volume"]
    bars["vwap"] = (df.groupby("bar_id").apply(
        lambda x: (x["price"] * x["volume"]).sum() / x["volume"].sum()
        if x["volume"].sum() > 0 else x["price"].mean()
    ))

    return bars
