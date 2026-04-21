"""Execution analytics for trade quality assessment.

Provides tools for analyzing trade execution quality, including:
- Slippage analysis and tracking
- Fill rate and partial fill handling
- Implementation shortfall calculation
- Execution timing analysis
- Best execution benchmarking

Example:
    >>> from imst_quant.utils.execution_analytics import ExecutionAnalyzer
    >>> analyzer = ExecutionAnalyzer(trade_log)
    >>> metrics = analyzer.analyze()
    >>> print(f"Average slippage: {metrics.avg_slippage_bps:.1f} bps")
    Average slippage: 2.3 bps

References:
    - Perold (1988): The Implementation Shortfall
    - Almgren, Chriss (2001): Optimal Execution of Portfolio Transactions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import polars as pl


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type classification."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class ExecutedTrade:
    """Record of an executed trade.

    Attributes:
        trade_id: Unique trade identifier.
        symbol: Trading symbol.
        side: Buy or sell.
        order_type: Type of order.
        decision_price: Price when decision was made.
        order_price: Price when order was placed.
        fill_price: Actual execution price.
        quantity_ordered: Shares/contracts ordered.
        quantity_filled: Shares/contracts filled.
        decision_time: When trading decision was made.
        order_time: When order was submitted.
        fill_time: When order was filled.
        commission: Trading commission.
        venue: Execution venue.
    """
    trade_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    decision_price: float
    order_price: float
    fill_price: float
    quantity_ordered: float
    quantity_filled: float
    decision_time: Optional[datetime] = None
    order_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    venue: str = "unknown"


@dataclass
class ExecutionMetrics:
    """Aggregate execution quality metrics.

    Attributes:
        total_trades: Number of trades analyzed.
        fill_rate: Percentage of orders fully filled.
        avg_slippage_bps: Average slippage in basis points.
        avg_slippage_pct: Average slippage as percentage.
        total_slippage_cost: Total slippage cost in currency.
        implementation_shortfall: Total implementation shortfall.
        avg_fill_time_seconds: Average time to fill.
        market_impact_estimate: Estimated market impact.
        timing_cost: Cost from timing delays.
        spread_cost: Cost from bid-ask spread.
        commission_cost: Total commissions paid.
    """
    total_trades: int = 0
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_slippage_pct: float = 0.0
    total_slippage_cost: float = 0.0
    implementation_shortfall: float = 0.0
    avg_fill_time_seconds: float = 0.0
    market_impact_estimate: float = 0.0
    timing_cost: float = 0.0
    spread_cost: float = 0.0
    commission_cost: float = 0.0
    slippage_by_venue: dict[str, float] = field(default_factory=dict)
    slippage_by_symbol: dict[str, float] = field(default_factory=dict)
    slippage_by_size_bucket: dict[str, float] = field(default_factory=dict)


def calculate_slippage(
    trade: ExecutedTrade,
) -> dict:
    """Calculate slippage metrics for a single trade.

    Computes slippage relative to both decision price (implementation
    shortfall) and order price (execution slippage).

    Args:
        trade: Executed trade record.

    Returns:
        Dict with slippage metrics.
    """
    # Execution slippage: fill vs order price
    if trade.side == OrderSide.BUY:
        exec_slippage = trade.fill_price - trade.order_price
    else:
        exec_slippage = trade.order_price - trade.fill_price

    exec_slippage_pct = exec_slippage / trade.order_price if trade.order_price > 0 else 0
    exec_slippage_bps = exec_slippage_pct * 10000

    # Implementation shortfall: fill vs decision price
    if trade.side == OrderSide.BUY:
        impl_shortfall = trade.fill_price - trade.decision_price
    else:
        impl_shortfall = trade.decision_price - trade.fill_price

    impl_shortfall_pct = impl_shortfall / trade.decision_price if trade.decision_price > 0 else 0
    impl_shortfall_bps = impl_shortfall_pct * 10000

    # Dollar costs
    notional = trade.fill_price * trade.quantity_filled
    slippage_cost = exec_slippage * trade.quantity_filled
    shortfall_cost = impl_shortfall * trade.quantity_filled

    # Fill rate
    fill_rate = trade.quantity_filled / trade.quantity_ordered if trade.quantity_ordered > 0 else 0

    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "side": trade.side.value,
        "exec_slippage_bps": exec_slippage_bps,
        "exec_slippage_pct": exec_slippage_pct,
        "impl_shortfall_bps": impl_shortfall_bps,
        "impl_shortfall_pct": impl_shortfall_pct,
        "slippage_cost": slippage_cost,
        "shortfall_cost": shortfall_cost,
        "notional": notional,
        "fill_rate": fill_rate,
        "commission": trade.commission,
        "venue": trade.venue,
    }


def analyze_execution_quality(
    trades: list[ExecutedTrade],
) -> ExecutionMetrics:
    """Analyze execution quality across multiple trades.

    Args:
        trades: List of executed trades.

    Returns:
        ExecutionMetrics with aggregate statistics.
    """
    if not trades:
        return ExecutionMetrics()

    slippages = [calculate_slippage(t) for t in trades]

    # Aggregate metrics
    total_trades = len(trades)
    fully_filled = sum(1 for s in slippages if s["fill_rate"] >= 0.999)
    fill_rate = fully_filled / total_trades

    # Slippage statistics
    slippage_bps = [s["exec_slippage_bps"] for s in slippages]
    avg_slippage_bps = np.mean(slippage_bps)
    avg_slippage_pct = np.mean([s["exec_slippage_pct"] for s in slippages])

    total_slippage_cost = sum(s["slippage_cost"] for s in slippages)
    implementation_shortfall = sum(s["shortfall_cost"] for s in slippages)
    commission_cost = sum(s["commission"] for s in slippages)

    # Timing metrics (if timestamps available)
    fill_times = []
    for t in trades:
        if t.order_time and t.fill_time:
            delta = (t.fill_time - t.order_time).total_seconds()
            fill_times.append(delta)
    avg_fill_time = np.mean(fill_times) if fill_times else 0.0

    # Slippage by venue
    slippage_by_venue: dict[str, list] = {}
    for s in slippages:
        venue = s["venue"]
        if venue not in slippage_by_venue:
            slippage_by_venue[venue] = []
        slippage_by_venue[venue].append(s["exec_slippage_bps"])
    slippage_by_venue = {k: np.mean(v) for k, v in slippage_by_venue.items()}

    # Slippage by symbol
    slippage_by_symbol: dict[str, list] = {}
    for s in slippages:
        symbol = s["symbol"]
        if symbol not in slippage_by_symbol:
            slippage_by_symbol[symbol] = []
        slippage_by_symbol[symbol].append(s["exec_slippage_bps"])
    slippage_by_symbol = {k: np.mean(v) for k, v in slippage_by_symbol.items()}

    # Slippage by size bucket
    notionals = [s["notional"] for s in slippages]
    p33, p66 = np.percentile(notionals, [33, 66])

    size_buckets = {"small": [], "medium": [], "large": []}
    for s in slippages:
        if s["notional"] <= p33:
            size_buckets["small"].append(s["exec_slippage_bps"])
        elif s["notional"] <= p66:
            size_buckets["medium"].append(s["exec_slippage_bps"])
        else:
            size_buckets["large"].append(s["exec_slippage_bps"])

    slippage_by_size = {
        k: np.mean(v) if v else 0.0 for k, v in size_buckets.items()
    }

    return ExecutionMetrics(
        total_trades=total_trades,
        fill_rate=fill_rate,
        avg_slippage_bps=avg_slippage_bps,
        avg_slippage_pct=avg_slippage_pct,
        total_slippage_cost=total_slippage_cost,
        implementation_shortfall=implementation_shortfall,
        avg_fill_time_seconds=avg_fill_time,
        commission_cost=commission_cost,
        slippage_by_venue=slippage_by_venue,
        slippage_by_symbol=slippage_by_symbol,
        slippage_by_size_bucket=slippage_by_size,
    )


def estimate_expected_slippage(
    order_size: float,
    avg_daily_volume: float,
    avg_spread_bps: float,
    volatility: float,
) -> dict:
    """Estimate expected slippage for a hypothetical order.

    Uses market microstructure models to estimate execution costs.

    Args:
        order_size: Order size in shares.
        avg_daily_volume: Average daily volume.
        avg_spread_bps: Average bid-ask spread in basis points.
        volatility: Daily volatility.

    Returns:
        Dict with slippage estimates.
    """
    participation_rate = order_size / avg_daily_volume if avg_daily_volume > 0 else 1.0

    # Spread cost (half spread assumed)
    spread_cost_bps = avg_spread_bps / 2

    # Market impact (square root model: impact ~ sigma * sqrt(participation))
    # Coefficient calibrated to ~0.5 for typical equities
    impact_coefficient = 0.5
    market_impact_bps = impact_coefficient * volatility * 10000 * np.sqrt(participation_rate)

    # Timing cost (opportunity cost from delay)
    # Assumes 30min average execution time for larger orders
    timing_cost_bps = 0.1 * volatility * 10000 * participation_rate

    total_cost_bps = spread_cost_bps + market_impact_bps + timing_cost_bps

    return {
        "participation_rate": participation_rate,
        "spread_cost_bps": spread_cost_bps,
        "market_impact_bps": market_impact_bps,
        "timing_cost_bps": timing_cost_bps,
        "total_expected_cost_bps": total_cost_bps,
        "is_large_order": participation_rate > 0.1,
        "recommendation": (
            "Consider algorithmic execution" if participation_rate > 0.05
            else "Market order acceptable"
        ),
    }


def vwap_deviation(
    fills: list[dict],
    market_vwap: float,
) -> dict:
    """Calculate VWAP deviation for executed fills.

    Compares execution price to market VWAP benchmark.

    Args:
        fills: List of fills with price, quantity, side.
        market_vwap: Market VWAP for the period.

    Returns:
        Dict with VWAP analysis.
    """
    if not fills or market_vwap <= 0:
        return {"error": "Invalid inputs"}

    # Calculate execution VWAP
    total_notional = sum(f["price"] * f["quantity"] for f in fills)
    total_quantity = sum(f["quantity"] for f in fills)
    exec_vwap = total_notional / total_quantity if total_quantity > 0 else 0

    # Deviation
    deviation = exec_vwap - market_vwap
    deviation_bps = (deviation / market_vwap) * 10000 if market_vwap > 0 else 0

    # Adjust for side
    side = fills[0].get("side", "buy")
    if side == "sell":
        deviation_bps = -deviation_bps  # Positive = good for sells

    return {
        "exec_vwap": exec_vwap,
        "market_vwap": market_vwap,
        "deviation": deviation,
        "deviation_bps": deviation_bps,
        "beat_vwap": deviation_bps < 0,
        "total_quantity": total_quantity,
        "total_notional": total_notional,
    }


def generate_execution_report(
    trades: list[ExecutedTrade],
    period_label: str = "Analysis Period",
) -> str:
    """Generate a text report of execution quality.

    Args:
        trades: List of executed trades.
        period_label: Label for the analysis period.

    Returns:
        Formatted report string.
    """
    metrics = analyze_execution_quality(trades)

    lines = [
        f"{'=' * 50}",
        f"EXECUTION QUALITY REPORT: {period_label}",
        f"{'=' * 50}",
        "",
        f"Total Trades:           {metrics.total_trades:,}",
        f"Fill Rate:              {metrics.fill_rate:.1%}",
        f"Avg Fill Time:          {metrics.avg_fill_time_seconds:.1f}s",
        "",
        "--- Slippage Analysis ---",
        f"Avg Slippage:           {metrics.avg_slippage_bps:.2f} bps",
        f"Total Slippage Cost:    ${metrics.total_slippage_cost:,.2f}",
        f"Impl. Shortfall:        ${metrics.implementation_shortfall:,.2f}",
        f"Commission Cost:        ${metrics.commission_cost:,.2f}",
        "",
    ]

    if metrics.slippage_by_venue:
        lines.append("--- By Venue ---")
        for venue, slip in sorted(metrics.slippage_by_venue.items()):
            lines.append(f"  {venue:20} {slip:+.2f} bps")
        lines.append("")

    if metrics.slippage_by_size_bucket:
        lines.append("--- By Order Size ---")
        for bucket, slip in metrics.slippage_by_size_bucket.items():
            lines.append(f"  {bucket:20} {slip:+.2f} bps")
        lines.append("")

    return "\n".join(lines)
