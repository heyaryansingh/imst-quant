"""
Market microstructure analysis utilities.

Provides tools for analyzing order flow, bid-ask spreads, liquidity,
and other microstructure patterns.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_bid_ask_spread(
    bid: pd.Series,
    ask: pd.Series,
    mid_price: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Calculate bid-ask spread metrics.

    Args:
        bid: Bid price series
        ask: Ask price series
        mid_price: Mid price series (optional, calculated if not provided)

    Returns:
        DataFrame with spread metrics
    """
    if mid_price is None:
        mid_price = (bid + ask) / 2

    absolute_spread = ask - bid
    relative_spread = absolute_spread / mid_price
    percentage_spread = relative_spread * 100

    return pd.DataFrame({
        'absolute_spread': absolute_spread,
        'relative_spread': relative_spread,
        'percentage_spread': percentage_spread,
        'mid_price': mid_price,
    })


def calculate_effective_spread(
    trade_price: pd.Series,
    mid_price: pd.Series,
    trade_direction: pd.Series,
) -> pd.Series:
    """
    Calculate effective spread from trade data.

    Effective spread = 2 * direction * (trade_price - mid_price)
    where direction = 1 for buy, -1 for sell

    Args:
        trade_price: Executed trade prices
        mid_price: Mid prices at trade time
        trade_direction: Trade direction (1 for buy, -1 for sell)

    Returns:
        Effective spread series
    """
    effective_spread = 2 * trade_direction * (trade_price - mid_price)
    return effective_spread


def calculate_realized_spread(
    trade_price: pd.Series,
    mid_price_t0: pd.Series,
    mid_price_t1: pd.Series,
    trade_direction: pd.Series,
) -> pd.Series:
    """
    Calculate realized spread.

    Measures the revenue captured by liquidity providers.
    Realized spread = 2 * direction * (trade_price - mid_price_t1)

    Args:
        trade_price: Executed trade prices
        mid_price_t0: Mid price at trade time
        mid_price_t1: Mid price after trade (e.g., 5 minutes later)
        trade_direction: Trade direction (1 for buy, -1 for sell)

    Returns:
        Realized spread series
    """
    realized_spread = 2 * trade_direction * (trade_price - mid_price_t1)
    return realized_spread


def calculate_price_impact(
    trade_price: pd.Series,
    mid_price_t0: pd.Series,
    mid_price_t1: pd.Series,
    trade_direction: pd.Series,
) -> pd.Series:
    """
    Calculate price impact of trades.

    Price impact = 2 * direction * (mid_price_t1 - mid_price_t0)

    Args:
        trade_price: Executed trade prices
        mid_price_t0: Mid price at trade time
        mid_price_t1: Mid price after trade
        trade_direction: Trade direction (1 for buy, -1 for sell)

    Returns:
        Price impact series
    """
    price_impact = 2 * trade_direction * (mid_price_t1 - mid_price_t0)
    return price_impact


def calculate_roll_spread(
    prices: pd.Series,
) -> float:
    """
    Estimate bid-ask spread using Roll's method.

    Assumes serial covariance in price changes is caused by bid-ask bounce.
    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_t-1))

    Args:
        prices: Price series

    Returns:
        Estimated bid-ask spread
    """
    price_changes = prices.diff().dropna()

    # Calculate serial covariance
    cov = price_changes.cov(price_changes.shift(1))

    if cov >= 0:
        return 0.0  # Roll method assumes negative covariance

    spread = 2 * np.sqrt(-cov)
    return float(spread)


def estimate_order_flow_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
) -> pd.Series:
    """
    Calculate order flow imbalance.

    OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    Args:
        buy_volume: Buy volume series
        sell_volume: Sell volume series

    Returns:
        Order flow imbalance series
    """
    total_volume = buy_volume + sell_volume
    ofi = (buy_volume - sell_volume) / total_volume
    return ofi.fillna(0)


def calculate_amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Calculate Amihud illiquidity measure.

    Amihud = |return| / volume

    Higher values indicate lower liquidity.

    Args:
        returns: Return series
        volume: Volume series

    Returns:
        Amihud illiquidity series
    """
    illiquidity = returns.abs() / volume
    return illiquidity.replace([np.inf, -np.inf], np.nan)


def calculate_volume_synchronized_probability(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Calculate Volume Synchronized Probability of Informed Trading (VPIN).

    Simplified VPIN calculation based on volume buckets.

    Args:
        prices: Price series
        volumes: Volume series
        window: Number of volume buckets for calculation

    Returns:
        VPIN series
    """
    # Create volume buckets
    cumulative_volume = volumes.cumsum()
    bucket_size = cumulative_volume.iloc[-1] / window

    results = []
    bucket_volumes_buy = []
    bucket_volumes_sell = []

    current_bucket_vol = 0
    current_bucket_buy = 0
    current_bucket_sell = 0

    price_changes = prices.diff()

    for i in range(len(prices)):
        vol = volumes.iloc[i]
        price_change = price_changes.iloc[i] if i > 0 else 0

        # Classify as buy or sell based on price change
        if price_change > 0:
            buy_vol = vol
            sell_vol = 0
        elif price_change < 0:
            buy_vol = 0
            sell_vol = vol
        else:
            buy_vol = vol / 2
            sell_vol = vol / 2

        current_bucket_vol += vol
        current_bucket_buy += buy_vol
        current_bucket_sell += sell_vol

        # When bucket is full, calculate VPIN
        if current_bucket_vol >= bucket_size:
            bucket_volumes_buy.append(current_bucket_buy)
            bucket_volumes_sell.append(current_bucket_sell)

            if len(bucket_volumes_buy) >= window:
                # Keep only last window buckets
                bucket_volumes_buy = bucket_volumes_buy[-window:]
                bucket_volumes_sell = bucket_volumes_sell[-window:]

                # Calculate VPIN
                total_buy = sum(bucket_volumes_buy)
                total_sell = sum(bucket_volumes_sell)
                total_volume = total_buy + total_sell

                if total_volume > 0:
                    vpin = abs(total_buy - total_sell) / total_volume
                else:
                    vpin = 0

                results.append((prices.index[i], vpin))

            # Reset bucket
            current_bucket_vol = 0
            current_bucket_buy = 0
            current_bucket_sell = 0

    if len(results) == 0:
        return pd.Series(dtype=float)

    vpin_df = pd.DataFrame(results, columns=['timestamp', 'vpin'])
    return vpin_df.set_index('timestamp')['vpin']


def calculate_kyle_lambda(
    returns: pd.Series,
    signed_volume: pd.Series,
) -> float:
    """
    Estimate Kyle's lambda (price impact coefficient).

    Lambda measures the price impact per unit of order flow.
    Estimated via regression: return = lambda * signed_volume

    Args:
        returns: Return series
        signed_volume: Signed volume (positive for buy, negative for sell)

    Returns:
        Kyle's lambda estimate
    """
    # Simple OLS: regress returns on signed volume
    valid_data = ~(returns.isna() | signed_volume.isna())
    y = returns[valid_data].values
    X = signed_volume[valid_data].values

    if len(y) < 2 or X.std() == 0:
        return 0.0

    # Lambda = Cov(R, V) / Var(V)
    lambda_estimate = np.cov(y, X)[0, 1] / np.var(X)

    return float(lambda_estimate)


def calculate_volume_weighted_spread(
    spread: pd.Series,
    volume: pd.Series,
) -> float:
    """
    Calculate volume-weighted average spread.

    Args:
        spread: Spread series
        volume: Volume series

    Returns:
        Volume-weighted spread
    """
    total_volume = volume.sum()

    if total_volume == 0:
        return 0.0

    vw_spread = (spread * volume).sum() / total_volume
    return float(vw_spread)


def detect_quote_stuffing(
    quote_updates: pd.DataFrame,
    time_col: str = 'timestamp',
    window: pd.Timedelta = pd.Timedelta(seconds=1),
    threshold: int = 100,
) -> pd.Series:
    """
    Detect potential quote stuffing behavior.

    Quote stuffing is the practice of flooding the market with orders
    to slow down competitors.

    Args:
        quote_updates: DataFrame of quote updates
        time_col: Name of timestamp column
        window: Time window for counting updates
        threshold: Update count threshold to flag as quote stuffing

    Returns:
        Boolean series indicating potential quote stuffing
    """
    quote_updates = quote_updates.set_index(time_col)

    # Count updates in rolling window
    quote_updates['update_count'] = 1
    rolling_count = quote_updates['update_count'].rolling(window).sum()

    # Flag as quote stuffing if exceeds threshold
    is_stuffing = rolling_count > threshold

    return is_stuffing


def calculate_market_depth_imbalance(
    bid_depth: pd.Series,
    ask_depth: pd.Series,
    n_levels: int = 5,
) -> pd.Series:
    """
    Calculate market depth imbalance.

    Measures asymmetry between bid and ask side depth.

    Args:
        bid_depth: Total bid depth (sum of volumes)
        ask_depth: Total ask depth (sum of volumes)
        n_levels: Number of levels considered

    Returns:
        Depth imbalance series (-1 to 1, positive = more bids)
    """
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth
    return imbalance.fillna(0)


def calculate_trade_intensity(
    trade_times: pd.Series,
    window: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.Series:
    """
    Calculate trade arrival intensity (trades per unit time).

    Args:
        trade_times: Timestamps of trades
        window: Time window for counting trades

    Returns:
        Trade intensity series
    """
    trade_df = pd.DataFrame({'timestamp': trade_times})
    trade_df['trade_count'] = 1
    trade_df = trade_df.set_index('timestamp')

    # Count trades in rolling window
    intensity = trade_df['trade_count'].rolling(window).sum()

    # Convert to rate (trades per minute)
    window_minutes = window.total_seconds() / 60
    intensity = intensity / window_minutes

    return intensity


def calculate_quoted_spread_components(
    bid: pd.Series,
    ask: pd.Series,
    mid_price: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    Decompose quoted spread into components.

    Args:
        bid: Bid price series
        ask: Ask price series
        mid_price: Mid price series (optional)

    Returns:
        Dictionary with spread components
    """
    if mid_price is None:
        mid_price = (bid + ask) / 2

    half_spread = (ask - bid) / 2

    # Relative measures
    relative_half_spread = half_spread / mid_price

    return {
        'half_spread': half_spread,
        'relative_half_spread': relative_half_spread,
        'bid_depth_to_mid': (mid_price - bid) / mid_price,
        'ask_depth_to_mid': (ask - mid_price) / mid_price,
    }


def estimate_adverse_selection_cost(
    effective_spread: pd.Series,
    realized_spread: pd.Series,
) -> pd.Series:
    """
    Estimate adverse selection component of trading costs.

    Adverse selection = Effective spread - Realized spread

    Args:
        effective_spread: Effective spread series
        realized_spread: Realized spread series

    Returns:
        Adverse selection cost series
    """
    adverse_selection = effective_spread - realized_spread
    return adverse_selection
