"""Transaction cost analysis and estimation utilities.

This module provides comprehensive transaction cost analysis including
commission, slippage, market impact, and total execution cost estimation
for portfolio rebalancing and trading operations.

Functions:
    estimate_commission: Calculate brokerage commission costs
    estimate_slippage: Estimate slippage costs based on order size
    estimate_market_impact: Calculate market impact based on Kyle's model
    estimate_total_cost: Comprehensive transaction cost estimate
    analyze_turnover_costs: Analyze costs associated with portfolio turnover
    optimal_execution_schedule: Calculate optimal trade execution schedule

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.transaction_costs import estimate_total_cost
    >>> trade_size = 10000.0  # $10k order
    >>> avg_volume = 1000000.0  # $1M daily volume
    >>> cost = estimate_total_cost(trade_size, avg_volume, spread_bps=5.0)
    >>> print(f"Total transaction cost: {cost:.2f} bps")
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def estimate_commission(
    trade_value: float,
    commission_rate: float = 0.0005,
    min_commission: float = 1.0,
    max_commission: Optional[float] = None,
) -> float:
    """Calculate brokerage commission costs.

    Args:
        trade_value: Total value of the trade in USD.
        commission_rate: Commission rate as a decimal (default: 0.0005 = 5 bps).
        min_commission: Minimum commission per trade (default: $1.00).
        max_commission: Optional maximum commission cap.

    Returns:
        Commission cost in USD.

    Example:
        >>> commission = estimate_commission(10000.0, commission_rate=0.0005)
        >>> print(f"Commission: ${commission:.2f}")
    """
    commission = trade_value * commission_rate
    commission = max(commission, min_commission)

    if max_commission is not None:
        commission = min(commission, max_commission)

    return commission


def estimate_slippage(
    order_size: float,
    avg_daily_volume: float,
    spread_bps: float = 5.0,
    volatility: float = 0.02,
) -> float:
    """Estimate slippage costs based on order size and market conditions.

    Uses a simple model combining bid-ask spread crossing and
    proportional market impact based on volume participation.

    Args:
        order_size: Size of order in USD.
        avg_daily_volume: Average daily trading volume in USD.
        spread_bps: Bid-ask spread in basis points (default: 5.0).
        volatility: Daily price volatility as decimal (default: 0.02 = 2%).

    Returns:
        Estimated slippage cost in basis points.

    Example:
        >>> slippage_bps = estimate_slippage(10000, 1000000, spread_bps=5.0)
        >>> print(f"Slippage: {slippage_bps:.2f} bps")
    """
    # Spread crossing cost (pay half spread on average)
    spread_cost = spread_bps / 2.0

    # Volume participation rate
    participation_rate = order_size / avg_daily_volume

    # Market impact proportional to sqrt of participation * volatility
    # Using simplified Kyle's lambda model
    impact_bps = 1000.0 * volatility * np.sqrt(participation_rate)

    total_slippage = spread_cost + impact_bps

    return float(total_slippage)


def estimate_market_impact(
    order_size: float,
    avg_daily_volume: float,
    volatility: float = 0.02,
    lambda_param: float = 0.1,
) -> float:
    """Calculate permanent market impact using Kyle's lambda model.

    Kyle's lambda model: impact = lambda * (Q / V) * sigma
    where Q = order size, V = daily volume, sigma = volatility

    Args:
        order_size: Size of order in USD.
        avg_daily_volume: Average daily trading volume in USD.
        volatility: Daily price volatility (default: 0.02).
        lambda_param: Kyle's lambda parameter (default: 0.1).

    Returns:
        Permanent market impact in basis points.

    Example:
        >>> impact = estimate_market_impact(50000, 2000000, volatility=0.015)
        >>> print(f"Market impact: {impact:.2f} bps")
    """
    participation_rate = order_size / avg_daily_volume
    impact_decimal = lambda_param * participation_rate * volatility
    impact_bps = impact_decimal * 10000.0

    return float(impact_bps)


def estimate_total_cost(
    order_size: float,
    avg_daily_volume: float,
    spread_bps: float = 5.0,
    volatility: float = 0.02,
    commission_rate: float = 0.0005,
    min_commission: float = 1.0,
) -> float:
    """Comprehensive transaction cost estimate.

    Combines commission, slippage, and market impact into total
    execution cost in basis points.

    Args:
        order_size: Size of order in USD.
        avg_daily_volume: Average daily trading volume in USD.
        spread_bps: Bid-ask spread in basis points (default: 5.0).
        volatility: Daily price volatility (default: 0.02).
        commission_rate: Commission rate as decimal (default: 0.0005).
        min_commission: Minimum commission in USD (default: 1.0).

    Returns:
        Total transaction cost in basis points.

    Example:
        >>> cost = estimate_total_cost(25000, 1500000)
        >>> print(f"Total cost: {cost:.2f} bps")
    """
    # Commission cost in bps
    commission_usd = estimate_commission(order_size, commission_rate, min_commission)
    commission_bps = (commission_usd / order_size) * 10000.0

    # Slippage cost
    slippage_bps = estimate_slippage(order_size, avg_daily_volume, spread_bps, volatility)

    # Market impact
    impact_bps = estimate_market_impact(order_size, avg_daily_volume, volatility)

    total_cost_bps = commission_bps + slippage_bps + impact_bps

    return float(total_cost_bps)


def analyze_turnover_costs(
    portfolio_value: float,
    turnover_rate: float,
    avg_cost_bps: float = 10.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Analyze costs associated with portfolio turnover.

    Calculates annual transaction costs based on portfolio value,
    turnover rate, and average execution costs.

    Args:
        portfolio_value: Total portfolio value in USD.
        turnover_rate: Annual turnover rate as decimal (1.0 = 100% turnover).
        avg_cost_bps: Average transaction cost in basis points (default: 10.0).
        periods_per_year: Trading periods per year (default: 252).

    Returns:
        Dictionary containing:
            - annual_cost_usd: Total annual cost in USD
            - annual_cost_pct: Annual cost as % of portfolio value
            - daily_avg_cost_usd: Average daily transaction cost
            - breakeven_alpha: Alpha needed to break even after costs

    Example:
        >>> turnover = analyze_turnover_costs(1000000, 2.0, avg_cost_bps=15.0)
        >>> print(f"Annual cost: ${turnover['annual_cost_usd']:,.2f}")
        >>> print(f"Breakeven alpha: {turnover['breakeven_alpha']:.2%}")
    """
    # Convert bps to decimal
    avg_cost_decimal = avg_cost_bps / 10000.0

    # Total annual trading volume
    annual_volume = portfolio_value * turnover_rate

    # Annual transaction costs
    annual_cost_usd = annual_volume * avg_cost_decimal
    annual_cost_pct = annual_cost_usd / portfolio_value

    # Daily average
    daily_avg_cost_usd = annual_cost_usd / periods_per_year

    # Breakeven alpha (alpha needed to overcome costs)
    breakeven_alpha = annual_cost_pct

    return {
        "annual_cost_usd": annual_cost_usd,
        "annual_cost_pct": annual_cost_pct,
        "daily_avg_cost_usd": daily_avg_cost_usd,
        "breakeven_alpha": breakeven_alpha,
    }


def optimal_execution_schedule(
    total_order_size: float,
    avg_daily_volume: float,
    max_participation_rate: float = 0.05,
    time_horizon_days: int = 5,
) -> List[Dict[str, float]]:
    """Calculate optimal trade execution schedule to minimize market impact.

    Spreads large order across multiple days to stay below participation
    rate threshold and minimize price impact.

    Args:
        total_order_size: Total order size in USD.
        avg_daily_volume: Average daily trading volume in USD.
        max_participation_rate: Maximum daily volume participation (default: 0.05 = 5%).
        time_horizon_days: Maximum days to complete order (default: 5).

    Returns:
        List of dictionaries with day, order_size, participation_rate,
        cumulative_filled.

    Example:
        >>> schedule = optimal_execution_schedule(500000, 2000000, max_participation_rate=0.05)
        >>> for day in schedule:
        ...     print(f"Day {day['day']}: ${day['order_size']:,.0f} ({day['participation_rate']:.2%})")
    """
    max_daily_size = avg_daily_volume * max_participation_rate

    if total_order_size <= max_daily_size:
        # Can execute in single day
        return [{
            "day": 1,
            "order_size": total_order_size,
            "participation_rate": total_order_size / avg_daily_volume,
            "cumulative_filled": total_order_size,
        }]

    # Calculate optimal daily chunks
    num_days_needed = int(np.ceil(total_order_size / max_daily_size))
    num_days = min(num_days_needed, time_horizon_days)

    # VWAP-weighted schedule (higher volume at beginning/end)
    # Using simple uniform schedule for now
    daily_size = total_order_size / num_days

    schedule = []
    cumulative = 0.0

    for day in range(1, num_days + 1):
        if day == num_days:
            # Last day: fill remaining amount
            order_size = total_order_size - cumulative
        else:
            order_size = daily_size

        cumulative += order_size

        schedule.append({
            "day": day,
            "order_size": order_size,
            "participation_rate": order_size / avg_daily_volume,
            "cumulative_filled": cumulative,
        })

    return schedule


def batch_cost_analysis(
    trades_df: pl.DataFrame,
    volume_col: str = "avg_daily_volume",
    size_col: str = "order_size",
    spread_col: str = "spread_bps",
) -> pl.DataFrame:
    """Perform transaction cost analysis on a batch of trades.

    Args:
        trades_df: DataFrame with columns: [symbol, order_size, avg_daily_volume, spread_bps, volatility].
        volume_col: Column name for average daily volume (default: avg_daily_volume).
        size_col: Column name for order size (default: order_size).
        spread_col: Column name for spread in bps (default: spread_bps).

    Returns:
        DataFrame with added columns: commission_bps, slippage_bps, impact_bps, total_cost_bps.

    Example:
        >>> trades = pl.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "order_size": [50000, 75000],
        ...     "avg_daily_volume": [2000000, 3000000],
        ...     "spread_bps": [3.0, 4.0],
        ...     "volatility": [0.015, 0.018],
        ... })
        >>> result = batch_cost_analysis(trades)
    """
    results = []

    for row in trades_df.iter_rows(named=True):
        order_size = row[size_col]
        avg_volume = row[volume_col]
        spread = row[spread_col]
        volatility = row.get("volatility", 0.02)

        commission_bps = (estimate_commission(order_size) / order_size) * 10000.0
        slippage_bps = estimate_slippage(order_size, avg_volume, spread, volatility)
        impact_bps = estimate_market_impact(order_size, avg_volume, volatility)
        total_cost = commission_bps + slippage_bps + impact_bps

        results.append({
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "impact_bps": impact_bps,
            "total_cost_bps": total_cost,
        })

    cost_df = pl.DataFrame(results)
    return pl.concat([trades_df, cost_df], how="horizontal")
