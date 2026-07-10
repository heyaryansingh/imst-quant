"""Portfolio rebalancing signal generator.

Generates rebalancing signals when portfolio weights drift beyond configured
thresholds. Supports calendar-based, threshold-based, and volatility-adjusted
rebalancing strategies.

Functions:
    calculate_drift: Measure absolute and relative weight drift from targets.
    threshold_rebalance_signal: Signal when any asset exceeds drift threshold.
    calendar_rebalance_signal: Signal on fixed calendar intervals.
    volatility_adjusted_threshold: Dynamic threshold based on recent volatility.
    generate_rebalance_orders: Compute trade sizes to restore target weights.
    rebalance_summary: Full rebalance analysis with drift, signals, and orders.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.rebalance_signals import rebalance_summary
    >>> current = {"AAPL": 0.35, "GOOG": 0.30, "MSFT": 0.35}
    >>> target = {"AAPL": 0.33, "GOOG": 0.34, "MSFT": 0.33}
    >>> summary = rebalance_summary(current, target, portfolio_value=100_000)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DriftMetrics:
    """Weight drift metrics for a single asset.

    Attributes:
        asset: Ticker or identifier.
        current_weight: Current portfolio weight.
        target_weight: Target portfolio weight.
        absolute_drift: current_weight - target_weight.
        relative_drift: absolute_drift / target_weight (if target > 0).
    """

    asset: str
    current_weight: float
    target_weight: float
    absolute_drift: float
    relative_drift: float


@dataclass
class RebalanceOrder:
    """Trade order to rebalance a single position.

    Attributes:
        asset: Ticker or identifier.
        side: BUY or SELL.
        weight_delta: Signed weight change (positive = buy).
        dollar_amount: Estimated trade size in dollars.
    """

    asset: str
    side: Literal["BUY", "SELL"]
    weight_delta: float
    dollar_amount: float


@dataclass
class RebalanceSummary:
    """Full rebalance analysis result.

    Attributes:
        drift_metrics: Per-asset drift measurements.
        max_absolute_drift: Largest absolute drift across all assets.
        max_relative_drift: Largest relative drift across all assets.
        trigger: Whether rebalance is recommended.
        reason: Human-readable reason for the signal.
        orders: Trade orders to restore target weights.
        total_turnover: Sum of absolute weight deltas (one-way turnover).
        estimated_cost: Rough transaction cost estimate.
    """

    drift_metrics: List[DriftMetrics]
    max_absolute_drift: float
    max_relative_drift: float
    trigger: bool
    reason: str
    orders: List[RebalanceOrder]
    total_turnover: float
    estimated_cost: float


def calculate_drift(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
) -> List[DriftMetrics]:
    """Measure weight drift from targets for each asset.

    Args:
        current_weights: Map of asset -> current portfolio weight.
        target_weights: Map of asset -> target portfolio weight.

    Returns:
        List of DriftMetrics, one per asset found in either dict.
    """
    all_assets = set(current_weights) | set(target_weights)
    metrics = []
    for asset in sorted(all_assets):
        current = current_weights.get(asset, 0.0)
        target = target_weights.get(asset, 0.0)
        abs_drift = current - target
        rel_drift = abs_drift / target if target != 0 else float("inf") if abs_drift != 0 else 0.0
        metrics.append(
            DriftMetrics(
                asset=asset,
                current_weight=current,
                target_weight=target,
                absolute_drift=abs_drift,
                relative_drift=rel_drift,
            )
        )
    return metrics


def threshold_rebalance_signal(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    abs_threshold: float = 0.05,
    rel_threshold: float = 0.25,
) -> tuple[bool, str]:
    """Signal rebalance when drift exceeds absolute or relative threshold.

    Args:
        current_weights: Current portfolio weights.
        target_weights: Target portfolio weights.
        abs_threshold: Maximum allowed absolute drift (default: 5%).
        rel_threshold: Maximum allowed relative drift (default: 25%).

    Returns:
        Tuple of (trigger, reason).
    """
    drift = calculate_drift(current_weights, target_weights)

    for d in drift:
        if abs(d.absolute_drift) > abs_threshold:
            return True, f"{d.asset} absolute drift {d.absolute_drift:+.2%} exceeds {abs_threshold:.0%}"
        if d.target_weight > 0 and abs(d.relative_drift) > rel_threshold:
            return True, f"{d.asset} relative drift {d.relative_drift:+.1%} exceeds {rel_threshold:.0%}"

    return False, "All weights within thresholds"


def calendar_rebalance_signal(
    last_rebalance: datetime,
    now: Optional[datetime] = None,
    frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly",
) -> tuple[bool, str]:
    """Signal rebalance on a calendar schedule.

    Args:
        last_rebalance: Timestamp of last rebalance.
        now: Current timestamp (defaults to utcnow).
        frequency: Rebalance interval.

    Returns:
        Tuple of (trigger, reason).
    """
    if now is None:
        now = datetime.utcnow()

    intervals = {
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "monthly": timedelta(days=30),
        "quarterly": timedelta(days=91),
    }
    interval = intervals[frequency]
    elapsed = now - last_rebalance

    if elapsed >= interval:
        return True, f"{frequency} rebalance due ({elapsed.days} days since last)"
    return False, f"Next {frequency} rebalance in {(interval - elapsed).days} days"


def volatility_adjusted_threshold(
    base_threshold: float,
    recent_volatility: float,
    long_term_volatility: float,
    sensitivity: float = 1.0,
) -> float:
    """Adjust rebalance threshold based on current vs. long-term volatility.

    In high-volatility regimes, widen the threshold to avoid excessive trading.
    In low-volatility regimes, tighten it to capture drift sooner.

    Args:
        base_threshold: Baseline drift threshold (e.g. 0.05).
        recent_volatility: Recent realized volatility (annualized).
        long_term_volatility: Long-term average volatility (annualized).
        sensitivity: Scaling factor (1.0 = proportional adjustment).

    Returns:
        Adjusted threshold.
    """
    if long_term_volatility <= 0:
        return base_threshold

    vol_ratio = recent_volatility / long_term_volatility
    adjustment = 1.0 + sensitivity * (vol_ratio - 1.0)
    adjusted = base_threshold * max(0.5, min(adjustment, 2.0))

    logger.debug(
        "vol_adjusted_threshold",
        base=base_threshold,
        vol_ratio=vol_ratio,
        adjusted=adjusted,
    )
    return adjusted


def generate_rebalance_orders(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    min_trade_size: float = 100.0,
) -> List[RebalanceOrder]:
    """Compute trade orders to restore target weights.

    Args:
        current_weights: Current portfolio weights.
        target_weights: Target portfolio weights.
        portfolio_value: Total portfolio value in dollars.
        min_trade_size: Minimum trade size to include (filters noise).

    Returns:
        List of RebalanceOrder for each asset needing adjustment.
    """
    drift = calculate_drift(current_weights, target_weights)
    orders = []

    for d in drift:
        dollar_amount = abs(d.absolute_drift) * portfolio_value
        if dollar_amount < min_trade_size:
            continue

        orders.append(
            RebalanceOrder(
                asset=d.asset,
                side="BUY" if d.absolute_drift < 0 else "SELL",
                weight_delta=d.absolute_drift,
                dollar_amount=dollar_amount,
            )
        )

    # Sort by dollar amount descending for execution priority
    orders.sort(key=lambda o: o.dollar_amount, reverse=True)
    return orders


def rebalance_summary(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float = 100_000.0,
    abs_threshold: float = 0.05,
    rel_threshold: float = 0.25,
    transaction_cost_bps: float = 10.0,
    min_trade_size: float = 100.0,
) -> RebalanceSummary:
    """Generate a full rebalance analysis.

    Args:
        current_weights: Current portfolio weights.
        target_weights: Target portfolio weights.
        portfolio_value: Total portfolio value in dollars.
        abs_threshold: Absolute drift threshold for triggering.
        rel_threshold: Relative drift threshold for triggering.
        transaction_cost_bps: Estimated transaction cost in basis points.
        min_trade_size: Minimum trade size in dollars.

    Returns:
        RebalanceSummary with drift, signal, orders, and cost estimate.
    """
    drift = calculate_drift(current_weights, target_weights)
    trigger, reason = threshold_rebalance_signal(
        current_weights, target_weights, abs_threshold, rel_threshold
    )
    orders = generate_rebalance_orders(
        current_weights, target_weights, portfolio_value, min_trade_size
    )

    max_abs = max((abs(d.absolute_drift) for d in drift), default=0.0)
    max_rel = max(
        (abs(d.relative_drift) for d in drift if d.target_weight > 0),
        default=0.0,
    )
    total_turnover = sum(abs(d.absolute_drift) for d in drift) / 2.0
    total_trade_value = sum(o.dollar_amount for o in orders)
    estimated_cost = total_trade_value * (transaction_cost_bps / 10_000.0)

    logger.info(
        "rebalance_summary",
        trigger=trigger,
        max_abs_drift=max_abs,
        total_turnover=total_turnover,
        estimated_cost=estimated_cost,
    )

    return RebalanceSummary(
        drift_metrics=drift,
        max_absolute_drift=max_abs,
        max_relative_drift=max_rel,
        trigger=trigger,
        reason=reason,
        orders=orders,
        total_turnover=total_turnover,
        estimated_cost=estimated_cost,
    )
