"""Portfolio turnover analysis and cost estimation.

This module analyzes portfolio turnover patterns, estimates rebalancing
costs, and provides turnover-aware optimization guidance. High turnover
erodes returns through transaction costs and market impact, so monitoring
and minimizing unnecessary turnover is critical for live trading.

Functions:
    calculate_turnover: Compute portfolio turnover from weight changes
    turnover_decomposition: Decompose turnover into drift vs rebalance
    estimate_turnover_cost: Estimate total cost of turnover
    turnover_budget: Calculate remaining turnover budget for a period
    turnover_summary: Comprehensive turnover analysis report

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.portfolio_turnover import calculate_turnover
    >>> weights_before = {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.4}
    >>> weights_after = {"AAPL": 0.25, "MSFT": 0.35, "GOOG": 0.4}
    >>> turnover = calculate_turnover(weights_before, weights_after)
    >>> print(f"One-way turnover: {turnover:.2%}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


@dataclass
class TurnoverDecomposition:
    """Breakdown of turnover into sources."""

    total_turnover: float
    drift_turnover: float
    rebalance_turnover: float
    new_positions: List[str]
    closed_positions: List[str]
    increased_positions: Dict[str, float]
    decreased_positions: Dict[str, float]


@dataclass
class TurnoverCost:
    """Estimated costs from portfolio turnover."""

    total_cost_bps: float
    commission_cost_bps: float
    spread_cost_bps: float
    market_impact_bps: float
    turnover_one_way: float
    cost_drag_annualized_bps: float


@dataclass
class TurnoverSummary:
    """Comprehensive turnover analysis."""

    avg_monthly_turnover: float
    annualized_turnover: float
    estimated_annual_cost_bps: float
    turnover_trend: str
    high_turnover_assets: List[Tuple[str, float]]
    budget_remaining: Optional[float]
    recommendation: str


def calculate_turnover(
    weights_before: Dict[str, float],
    weights_after: Dict[str, float],
    one_way: bool = True,
) -> float:
    """Compute portfolio turnover from weight changes.

    Turnover is the sum of absolute weight changes. One-way turnover
    counts only buys (or equivalently sells); two-way counts both.

    Args:
        weights_before: Portfolio weights before rebalance {asset: weight}.
        weights_after: Portfolio weights after rebalance {asset: weight}.
        one_way: If True (default), return one-way turnover (sum/2).
            If False, return two-way turnover (full sum).

    Returns:
        Turnover as a decimal (e.g., 0.10 = 10%).

    Example:
        >>> before = {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.4}
        >>> after = {"AAPL": 0.25, "MSFT": 0.35, "GOOG": 0.4}
        >>> calculate_turnover(before, after)
        0.05
    """
    all_assets = set(weights_before.keys()) | set(weights_after.keys())
    total_change = sum(
        abs(weights_after.get(a, 0.0) - weights_before.get(a, 0.0))
        for a in all_assets
    )

    if one_way:
        return total_change / 2.0
    return total_change


def turnover_decomposition(
    weights_before: Dict[str, float],
    weights_after: Dict[str, float],
    returns: Optional[Dict[str, float]] = None,
) -> TurnoverDecomposition:
    """Decompose turnover into drift-driven and active rebalance components.

    If period returns are provided, separates natural weight drift
    (from price movements) from active trading decisions.

    Args:
        weights_before: Weights at start of period.
        weights_after: Target weights after rebalance.
        returns: Period returns per asset (if available, for drift calculation).

    Returns:
        TurnoverDecomposition with breakdown by source and per-asset changes.

    Example:
        >>> before = {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.4}
        >>> after = {"AAPL": 0.25, "MSFT": 0.35, "GOOG": 0.4}
        >>> rets = {"AAPL": -0.02, "MSFT": 0.03, "GOOG": 0.01}
        >>> decomp = turnover_decomposition(before, after, rets)
    """
    all_assets = set(weights_before.keys()) | set(weights_after.keys())

    # Compute drifted weights if returns available
    if returns is not None:
        portfolio_return = sum(
            weights_before.get(a, 0.0) * returns.get(a, 0.0)
            for a in all_assets
        )
        drifted = {}
        for a in all_assets:
            w = weights_before.get(a, 0.0)
            r = returns.get(a, 0.0)
            drifted[a] = w * (1 + r) / (1 + portfolio_return) if (1 + portfolio_return) != 0 else w
    else:
        drifted = dict(weights_before)

    # Total turnover
    total = calculate_turnover(weights_before, weights_after, one_way=True)

    # Drift turnover (what changed without trading)
    drift = calculate_turnover(weights_before, drifted, one_way=True)

    # Rebalance turnover (active trading beyond drift)
    rebalance = calculate_turnover(drifted, weights_after, one_way=True)

    # Categorize positions
    new_positions = [
        a for a in all_assets
        if weights_before.get(a, 0.0) == 0 and weights_after.get(a, 0.0) > 0
    ]
    closed_positions = [
        a for a in all_assets
        if weights_before.get(a, 0.0) > 0 and weights_after.get(a, 0.0) == 0
    ]
    increased = {
        a: weights_after.get(a, 0.0) - weights_before.get(a, 0.0)
        for a in all_assets
        if weights_after.get(a, 0.0) > weights_before.get(a, 0.0)
        and a not in new_positions
    }
    decreased = {
        a: weights_before.get(a, 0.0) - weights_after.get(a, 0.0)
        for a in all_assets
        if weights_after.get(a, 0.0) < weights_before.get(a, 0.0)
        and a not in closed_positions
    }

    return TurnoverDecomposition(
        total_turnover=total,
        drift_turnover=drift,
        rebalance_turnover=rebalance,
        new_positions=sorted(new_positions),
        closed_positions=sorted(closed_positions),
        increased_positions=dict(sorted(increased.items(), key=lambda x: -x[1])),
        decreased_positions=dict(sorted(decreased.items(), key=lambda x: -x[1])),
    )


def estimate_turnover_cost(
    weights_before: Dict[str, float],
    weights_after: Dict[str, float],
    portfolio_value: float = 1_000_000.0,
    commission_bps: float = 1.0,
    spread_bps: Optional[Dict[str, float]] = None,
    default_spread_bps: float = 5.0,
    market_impact_factor: float = 0.1,
) -> TurnoverCost:
    """Estimate the total cost of a portfolio rebalance.

    Accounts for commissions, bid-ask spread crossing costs, and
    market impact (which scales with trade size).

    Args:
        weights_before: Weights before rebalance.
        weights_after: Weights after rebalance.
        portfolio_value: Total portfolio value in currency (default: 1M).
        commission_bps: Commission per trade in basis points (default: 1).
        spread_bps: Per-asset bid-ask spread in bps (optional).
        default_spread_bps: Default spread if per-asset not provided (default: 5).
        market_impact_factor: Market impact scaling factor (default: 0.1).

    Returns:
        TurnoverCost with cost breakdown in basis points.

    Example:
        >>> before = {"AAPL": 0.5, "MSFT": 0.5}
        >>> after = {"AAPL": 0.3, "MSFT": 0.7}
        >>> cost = estimate_turnover_cost(before, after, portfolio_value=1_000_000)
        >>> print(f"Total cost: {cost.total_cost_bps:.1f} bps")
    """
    if spread_bps is None:
        spread_bps = {}

    all_assets = set(weights_before.keys()) | set(weights_after.keys())
    turnover = calculate_turnover(weights_before, weights_after, one_way=True)

    total_commission = 0.0
    total_spread = 0.0
    total_impact = 0.0

    for asset in all_assets:
        w_before = weights_before.get(asset, 0.0)
        w_after = weights_after.get(asset, 0.0)
        trade_size = abs(w_after - w_before)

        if trade_size == 0:
            continue

        trade_value = trade_size * portfolio_value

        # Commission
        total_commission += trade_value * commission_bps / 10000

        # Spread cost (half-spread per trade)
        asset_spread = spread_bps.get(asset, default_spread_bps)
        total_spread += trade_value * (asset_spread / 2) / 10000

        # Market impact (square root model: impact ~ factor * sqrt(trade_size))
        impact = market_impact_factor * np.sqrt(trade_size) * trade_value / 10000
        total_impact += impact

    total_cost = total_commission + total_spread + total_impact

    # Convert to bps of portfolio
    commission_as_bps = (total_commission / portfolio_value) * 10000
    spread_as_bps = (total_spread / portfolio_value) * 10000
    impact_as_bps = (total_impact / portfolio_value) * 10000
    total_as_bps = (total_cost / portfolio_value) * 10000

    # Annualized cost drag (assuming monthly rebalancing)
    annual_cost_bps = total_as_bps * 12

    return TurnoverCost(
        total_cost_bps=total_as_bps,
        commission_cost_bps=commission_as_bps,
        spread_cost_bps=spread_as_bps,
        market_impact_bps=impact_as_bps,
        turnover_one_way=turnover,
        cost_drag_annualized_bps=annual_cost_bps,
    )


def turnover_budget(
    annual_budget_pct: float,
    ytd_turnover_pct: float,
    months_remaining: int,
) -> Dict[str, float]:
    """Calculate remaining turnover budget for a period.

    Given an annual turnover budget and year-to-date usage, computes
    how much turnover capacity remains per month.

    Args:
        annual_budget_pct: Annual turnover budget as percentage (e.g., 200 = 200%).
        ytd_turnover_pct: Year-to-date turnover used (same units).
        months_remaining: Months remaining in the budget period.

    Returns:
        Dictionary with:
        - budget_remaining_pct: Total remaining budget
        - monthly_allowance_pct: Remaining per month
        - utilization_pct: Percentage of budget used
        - is_over_budget: Whether budget is exceeded

    Example:
        >>> budget = turnover_budget(200.0, 150.0, 3)
        >>> print(f"Monthly allowance: {budget['monthly_allowance_pct']:.1f}%")
    """
    remaining = annual_budget_pct - ytd_turnover_pct
    monthly = remaining / max(months_remaining, 1)
    utilization = (ytd_turnover_pct / annual_budget_pct * 100) if annual_budget_pct > 0 else 0.0

    return {
        "budget_remaining_pct": max(remaining, 0.0),
        "monthly_allowance_pct": max(monthly, 0.0),
        "utilization_pct": utilization,
        "is_over_budget": remaining < 0,
    }


def turnover_summary(
    weight_history: List[Dict[str, float]],
    returns_history: Optional[List[Dict[str, float]]] = None,
    annual_budget_pct: Optional[float] = None,
    months_elapsed: int = 0,
) -> TurnoverSummary:
    """Generate comprehensive turnover analysis from weight history.

    Analyzes a sequence of portfolio weight snapshots to compute
    turnover statistics, identify high-turnover assets, and assess
    budget utilization.

    Args:
        weight_history: List of weight dicts in chronological order.
        returns_history: Optional list of return dicts (aligned with weight changes).
        annual_budget_pct: Optional annual turnover budget (e.g., 200%).
        months_elapsed: Months elapsed in budget period.

    Returns:
        TurnoverSummary with statistics and recommendations.

    Example:
        >>> history = [
        ...     {"AAPL": 0.5, "MSFT": 0.5},
        ...     {"AAPL": 0.4, "MSFT": 0.6},
        ...     {"AAPL": 0.45, "MSFT": 0.55},
        ... ]
        >>> summary = turnover_summary(history)
    """
    if len(weight_history) < 2:
        return TurnoverSummary(
            avg_monthly_turnover=0.0,
            annualized_turnover=0.0,
            estimated_annual_cost_bps=0.0,
            turnover_trend="insufficient_data",
            high_turnover_assets=[],
            budget_remaining=None,
            recommendation="Need at least 2 weight snapshots for analysis.",
        )

    # Compute per-period turnover
    period_turnovers = []
    asset_turnovers: Dict[str, float] = {}

    for i in range(1, len(weight_history)):
        w_before = weight_history[i - 1]
        w_after = weight_history[i]
        t = calculate_turnover(w_before, w_after, one_way=True)
        period_turnovers.append(t)

        # Per-asset turnover contribution
        all_assets = set(w_before.keys()) | set(w_after.keys())
        for a in all_assets:
            change = abs(w_after.get(a, 0.0) - w_before.get(a, 0.0)) / 2
            asset_turnovers[a] = asset_turnovers.get(a, 0.0) + change

    avg_turnover = float(np.mean(period_turnovers))
    annualized = avg_turnover * 12  # Assuming monthly snapshots

    # Turnover trend (compare first half vs second half)
    mid = len(period_turnovers) // 2
    if mid > 0:
        first_half_avg = float(np.mean(period_turnovers[:mid]))
        second_half_avg = float(np.mean(period_turnovers[mid:]))
        if second_half_avg > first_half_avg * 1.2:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # High-turnover assets (top 5)
    sorted_assets = sorted(asset_turnovers.items(), key=lambda x: -x[1])
    high_turnover = sorted_assets[:5]

    # Estimated annual cost (rough estimate at 5 bps per unit turnover)
    est_cost_bps = annualized * 5 * 100  # turnover * cost_per_unit_turnover

    # Budget check
    budget_remaining = None
    if annual_budget_pct is not None:
        total_turnover_pct = sum(period_turnovers) * 100
        months_remaining = max(12 - months_elapsed, 1)
        budget_info = turnover_budget(annual_budget_pct, total_turnover_pct, months_remaining)
        budget_remaining = budget_info["budget_remaining_pct"]

    # Recommendation
    if annualized > 4.0:
        recommendation = "Very high annualized turnover (>400%). Review rebalancing frequency and signal stability to reduce cost drag."
    elif annualized > 2.0:
        recommendation = "High turnover. Consider widening rebalancing bands or reducing signal sensitivity."
    elif annualized < 0.2:
        recommendation = "Low turnover. Portfolio may be stale; ensure signals are being acted upon."
    else:
        recommendation = "Turnover is within typical bounds for an active strategy."

    return TurnoverSummary(
        avg_monthly_turnover=avg_turnover,
        annualized_turnover=annualized,
        estimated_annual_cost_bps=est_cost_bps,
        turnover_trend=trend,
        high_turnover_assets=high_turnover,
        budget_remaining=budget_remaining,
        recommendation=recommendation,
    )
