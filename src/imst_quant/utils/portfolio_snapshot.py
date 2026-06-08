"""Portfolio snapshot utilities for point-in-time portfolio analysis.

This module provides comprehensive point-in-time portfolio summaries designed
for CLI dashboard output. It generates formatted snapshots including holdings,
P&L metrics, risk summaries, and automated alerts.

Classes:
    HoldingSnapshot: Individual position summary
    PortfolioSnapshot: Complete portfolio state at a point in time

Functions:
    generate_snapshot: Create snapshot from positions and market data
    format_snapshot_text: Format snapshot for CLI output
    format_snapshot_json: Convert snapshot to JSON-serializable dict
    check_alerts: Generate risk alerts based on thresholds

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.portfolio_snapshot import (
    ...     generate_snapshot,
    ...     format_snapshot_text,
    ... )
    >>> positions = {"AAPL": 100, "GOOGL": 50}
    >>> prices_df = pl.DataFrame({
    ...     "asset_id": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
    ...     "date": ["2024-01-14", "2024-01-15", "2024-01-14", "2024-01-15"],
    ...     "close": [180.0, 185.5, 140.0, 142.3],
    ... })
    >>> returns_df = pl.DataFrame({
    ...     "asset_id": ["AAPL", "GOOGL"],
    ...     "date": ["2024-01-15", "2024-01-15"],
    ...     "return_1d": [0.0306, 0.0164],
    ... })
    >>> snapshot = generate_snapshot(positions, prices_df, returns_df)
    >>> print(format_snapshot_text(snapshot))
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import json
import numpy as np
import polars as pl


@dataclass
class HoldingSnapshot:
    """Snapshot of a single portfolio holding.

    Represents the current state of an individual position including
    market value, weights, returns, and unrealized P&L.

    Attributes:
        asset_id: Unique identifier for the asset (ticker/symbol).
        quantity: Number of units held (can be negative for shorts).
        current_price: Most recent market price.
        market_value: Current market value (quantity * price).
        weight: Portfolio weight as decimal (0.10 = 10%).
        daily_return: One-day return as decimal.
        total_return: Return since position inception as decimal.
        unrealized_pnl: Unrealized profit/loss in currency units.
    """

    asset_id: str
    quantity: float
    current_price: float
    market_value: float
    weight: float
    daily_return: float
    total_return: float
    unrealized_pnl: float


@dataclass
class PortfolioSnapshot:
    """Complete point-in-time portfolio snapshot.

    Provides a comprehensive view of portfolio state including aggregate
    metrics, individual holdings, risk summary, and alerts.

    Attributes:
        timestamp: ISO format timestamp of the snapshot.
        total_value: Total portfolio market value.
        daily_pnl: Profit/loss for the current day in currency units.
        daily_pnl_pct: Daily P&L as percentage.
        total_pnl: Total profit/loss since inception in currency units.
        total_pnl_pct: Total P&L as percentage of initial capital.
        holdings: List of HoldingSnapshot for each position.
        risk_summary: Dict of risk metrics (Sharpe, Sortino, MaxDD, VaR).
        alerts: List of alert messages for risk conditions.
    """

    timestamp: str
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    holdings: List[HoldingSnapshot] = field(default_factory=list)
    risk_summary: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


def _calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annualized risk-free rate (default 0).
        annualization_factor: Periods per year (252 for daily).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / annualization_factor)
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)

    if std_returns == 0 or np.isnan(std_returns):
        return 0.0

    return float(mean_excess / std_returns * np.sqrt(annualization_factor))


def _calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Calculate annualized Sortino ratio.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annualized risk-free rate (default 0).
        annualization_factor: Periods per year (252 for daily).

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / annualization_factor)
    mean_excess = np.mean(excess_returns)

    # Downside deviation: std of negative returns only
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return float(mean_excess * np.sqrt(annualization_factor)) if mean_excess > 0 else 0.0

    downside_std = np.std(negative_returns, ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(annualization_factor))


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from return series.

    Args:
        returns: Array of period returns.

    Returns:
        Maximum drawdown as positive decimal (0.10 = 10% drawdown).
    """
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max

    return float(np.max(drawdown))


def _calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Calculate Value at Risk using historical method.

    Args:
        returns: Array of period returns.
        confidence: Confidence level (default 0.95 = 95%).

    Returns:
        VaR as positive decimal (loss at given confidence level).
    """
    if len(returns) < 10:
        return 0.0

    percentile = (1 - confidence) * 100
    var = np.percentile(returns, percentile)

    return float(-var) if var < 0 else 0.0


def _compute_portfolio_returns(
    positions: Dict[str, float],
    returns_df: pl.DataFrame,
    prices_df: pl.DataFrame,
) -> np.ndarray:
    """Compute historical portfolio returns from position returns.

    Args:
        positions: Dict mapping asset_id to quantity.
        returns_df: DataFrame with columns [asset_id, date, return_1d].
        prices_df: DataFrame with columns [asset_id, date, close].

    Returns:
        Array of portfolio daily returns.
    """
    if returns_df.height == 0:
        return np.array([])

    # Get unique dates and sort
    dates = returns_df.select("date").unique().sort("date")["date"].to_list()

    if len(dates) == 0:
        return np.array([])

    # Get latest prices for weights
    latest_prices = (
        prices_df
        .sort("date", descending=True)
        .group_by("asset_id")
        .first()
        .select(["asset_id", "close"])
    )

    # Calculate market values and weights
    asset_values = {}
    total_value = 0.0

    for asset_id, qty in positions.items():
        price_row = latest_prices.filter(pl.col("asset_id") == asset_id)
        if price_row.height > 0:
            price = price_row["close"][0]
            value = abs(qty * price)
            asset_values[asset_id] = value
            total_value += value

    if total_value == 0:
        return np.array([])

    weights = {asset_id: val / total_value for asset_id, val in asset_values.items()}

    # Calculate weighted portfolio returns for each date
    portfolio_returns = []

    for date in dates:
        date_returns = returns_df.filter(pl.col("date") == date)
        daily_return = 0.0

        for asset_id, weight in weights.items():
            asset_return = date_returns.filter(pl.col("asset_id") == asset_id)
            if asset_return.height > 0:
                ret = asset_return["return_1d"][0]
                if ret is not None and not np.isnan(ret):
                    daily_return += weight * ret

        portfolio_returns.append(daily_return)

    return np.array(portfolio_returns)


def generate_snapshot(
    positions: Dict[str, float],
    prices_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    initial_capital: float = 100000.0,
    cost_basis: Optional[Dict[str, float]] = None,
) -> PortfolioSnapshot:
    """Generate a comprehensive portfolio snapshot.

    Creates a point-in-time summary of portfolio state including
    holdings, P&L metrics, risk summary, and alerts.

    Args:
        positions: Dict mapping asset_id to quantity held.
            Positive values are long positions, negative are short.
        prices_df: DataFrame with columns [asset_id, date, close].
            Must contain at least the latest date for each asset.
        returns_df: DataFrame with columns [asset_id, date, return_1d].
            Used for calculating risk metrics and daily returns.
        initial_capital: Starting capital for P&L calculations.
            Defaults to 100000.0.
        cost_basis: Optional dict mapping asset_id to average cost per unit.
            If not provided, assumes initial_capital was invested at
            earliest available prices.

    Returns:
        PortfolioSnapshot with complete portfolio state.

    Example:
        >>> positions = {"AAPL": 100, "GOOGL": 50, "MSFT": 75}
        >>> snapshot = generate_snapshot(positions, prices_df, returns_df)
        >>> print(f"Portfolio Value: ${snapshot.total_value:,.2f}")
        >>> print(f"Daily P&L: {snapshot.daily_pnl_pct:+.2%}")
    """
    timestamp = datetime.now().isoformat()

    # Handle empty positions
    if not positions:
        return PortfolioSnapshot(
            timestamp=timestamp,
            total_value=initial_capital,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            holdings=[],
            risk_summary={
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
            },
            alerts=[],
        )

    # Get latest prices for each asset
    latest_prices = (
        prices_df
        .sort("date", descending=True)
        .group_by("asset_id")
        .first()
        .select(["asset_id", "close"])
    )

    # Get previous day prices for daily return calculation
    prices_sorted = prices_df.sort(["asset_id", "date"], descending=[False, True])
    prev_prices_df = (
        prices_sorted
        .group_by("asset_id")
        .agg(pl.col("close").slice(1, 1).alias("prev_close"))
    )

    # Get latest returns
    latest_returns = (
        returns_df
        .sort("date", descending=True)
        .group_by("asset_id")
        .first()
        .select(["asset_id", "return_1d"])
    )

    # Build holdings and calculate totals
    holdings: List[HoldingSnapshot] = []
    total_value = 0.0
    total_cost = 0.0

    holding_data: List[Dict[str, Any]] = []

    for asset_id, qty in positions.items():
        # Get current price
        price_row = latest_prices.filter(pl.col("asset_id") == asset_id)
        if price_row.height == 0:
            continue

        current_price = float(price_row["close"][0])
        market_value = qty * current_price

        # Get previous price for daily return
        prev_price_row = prev_prices_df.filter(pl.col("asset_id") == asset_id)
        if prev_price_row.height > 0 and prev_price_row["prev_close"][0] is not None:
            prev_list = prev_price_row["prev_close"][0]
            prev_price = float(prev_list) if prev_list else current_price
        else:
            prev_price = current_price

        # Get daily return from returns_df or calculate from prices
        return_row = latest_returns.filter(pl.col("asset_id") == asset_id)
        if return_row.height > 0 and return_row["return_1d"][0] is not None:
            daily_return = float(return_row["return_1d"][0])
        elif prev_price > 0:
            daily_return = (current_price - prev_price) / prev_price
        else:
            daily_return = 0.0

        # Calculate cost basis
        if cost_basis and asset_id in cost_basis:
            unit_cost = cost_basis[asset_id]
        else:
            # Use earliest available price as cost basis estimate
            earliest_price = (
                prices_df
                .filter(pl.col("asset_id") == asset_id)
                .sort("date")
                .select("close")
                .head(1)
            )
            if earliest_price.height > 0:
                unit_cost = float(earliest_price["close"][0])
            else:
                unit_cost = current_price

        cost_value = qty * unit_cost
        unrealized_pnl = market_value - cost_value

        if cost_value != 0:
            total_return = (market_value - cost_value) / abs(cost_value)
        else:
            total_return = 0.0

        holding_data.append({
            "asset_id": asset_id,
            "quantity": qty,
            "current_price": current_price,
            "market_value": market_value,
            "daily_return": daily_return,
            "total_return": total_return,
            "unrealized_pnl": unrealized_pnl,
            "cost_value": cost_value,
        })

        total_value += market_value
        total_cost += cost_value

    # Calculate weights and create HoldingSnapshot objects
    for data in holding_data:
        weight = data["market_value"] / total_value if total_value != 0 else 0.0

        holdings.append(HoldingSnapshot(
            asset_id=data["asset_id"],
            quantity=data["quantity"],
            current_price=data["current_price"],
            market_value=data["market_value"],
            weight=weight,
            daily_return=data["daily_return"],
            total_return=data["total_return"],
            unrealized_pnl=data["unrealized_pnl"],
        ))

    # Sort holdings by market value descending
    holdings.sort(key=lambda h: abs(h.market_value), reverse=True)

    # Calculate portfolio-level metrics
    if total_value > 0:
        # Daily P&L (weighted sum of position daily P&L)
        daily_pnl = sum(
            h.market_value * h.daily_return / (1 + h.daily_return)
            for h in holdings
            if h.daily_return != -1.0
        )
        prev_value = total_value - daily_pnl
        daily_pnl_pct = daily_pnl / prev_value if prev_value != 0 else 0.0
    else:
        daily_pnl = 0.0
        daily_pnl_pct = 0.0

    # Total P&L relative to initial capital
    total_pnl = total_value - initial_capital
    total_pnl_pct = total_pnl / initial_capital if initial_capital != 0 else 0.0

    # Calculate risk metrics from historical returns
    portfolio_returns = _compute_portfolio_returns(positions, returns_df, prices_df)

    risk_summary = {
        "sharpe_ratio": _calculate_sharpe_ratio(portfolio_returns),
        "sortino_ratio": _calculate_sortino_ratio(portfolio_returns),
        "max_drawdown": _calculate_max_drawdown(portfolio_returns),
        "var_95": _calculate_var(portfolio_returns, confidence=0.95),
    }

    # Generate alerts
    alerts = check_alerts(
        PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            holdings=holdings,
            risk_summary=risk_summary,
            alerts=[],
        )
    )

    return PortfolioSnapshot(
        timestamp=timestamp,
        total_value=total_value,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        holdings=holdings,
        risk_summary=risk_summary,
        alerts=alerts,
    )


def check_alerts(
    snapshot: PortfolioSnapshot,
    max_weight: float = 0.10,
    max_drawdown: float = 0.20,
    min_sharpe: float = 0.5,
) -> List[str]:
    """Check portfolio for risk conditions and generate alerts.

    Evaluates the portfolio snapshot against configurable thresholds
    and returns a list of warning messages for any breached limits.

    Args:
        snapshot: PortfolioSnapshot to evaluate.
        max_weight: Maximum allowed position weight (default 0.10 = 10%).
            Positions exceeding this trigger concentration alerts.
        max_drawdown: Maximum allowed drawdown (default 0.20 = 20%).
            Drawdown exceeding this triggers drawdown alerts.
        min_sharpe: Minimum acceptable Sharpe ratio (default 0.5).
            Sharpe below this triggers performance alerts.

    Returns:
        List of alert message strings.

    Example:
        >>> alerts = check_alerts(snapshot, max_weight=0.15, max_drawdown=0.10)
        >>> for alert in alerts:
        ...     print(f"WARNING: {alert}")
    """
    alerts: List[str] = []

    # Check concentration risk
    for holding in snapshot.holdings:
        if abs(holding.weight) > max_weight:
            weight_pct = abs(holding.weight) * 100
            threshold_pct = max_weight * 100
            alerts.append(
                f"{holding.asset_id} weight ({weight_pct:.2f}%) "
                f"exceeds {threshold_pct:.0f}% concentration threshold"
            )

    # Check drawdown threshold
    current_dd = snapshot.risk_summary.get("max_drawdown", 0.0)
    if current_dd > max_drawdown:
        dd_pct = current_dd * 100
        threshold_pct = max_drawdown * 100
        alerts.append(
            f"Max drawdown ({dd_pct:.1f}%) exceeds {threshold_pct:.0f}% threshold"
        )

    # Check Sharpe ratio
    sharpe = snapshot.risk_summary.get("sharpe_ratio", 0.0)
    if sharpe < min_sharpe and sharpe != 0.0:
        alerts.append(
            f"Sharpe ratio ({sharpe:.2f}) below {min_sharpe:.1f} minimum threshold"
        )

    # Check for significant daily losses
    if snapshot.daily_pnl_pct < -0.05:  # More than 5% daily loss
        loss_pct = abs(snapshot.daily_pnl_pct) * 100
        alerts.append(f"Significant daily loss: -{loss_pct:.2f}%")

    # Check for very high VaR
    var_95 = snapshot.risk_summary.get("var_95", 0.0)
    if var_95 > 0.03:  # VaR exceeds 3%
        var_pct = var_95 * 100
        alerts.append(f"Elevated VaR(95%): {var_pct:.2f}% daily")

    return alerts


def _format_currency(value: float, width: int = 12) -> str:
    """Format currency value with thousands separators.

    Args:
        value: Currency amount.
        width: Minimum field width.

    Returns:
        Formatted string like "$1,234.56".
    """
    if value >= 0:
        return f"${value:,.2f}".rjust(width)
    else:
        return f"-${abs(value):,.2f}".rjust(width)


def _format_pct(value: float, width: int = 7, show_sign: bool = True) -> str:
    """Format percentage value.

    Args:
        value: Decimal percentage (0.10 = 10%).
        width: Minimum field width.
        show_sign: Include +/- prefix.

    Returns:
        Formatted string like "+10.5%".
    """
    pct = value * 100
    if show_sign:
        return f"{pct:+.2f}%".rjust(width)
    else:
        return f"{pct:.2f}%".rjust(width)


def _format_number(value: float, width: int = 10, decimals: int = 2) -> str:
    """Format number with specified precision.

    Args:
        value: Numeric value.
        width: Minimum field width.
        decimals: Decimal places.

    Returns:
        Formatted string.
    """
    return f"{value:,.{decimals}f}".rjust(width)


def format_snapshot_text(snapshot: PortfolioSnapshot) -> str:
    """Format portfolio snapshot for CLI text output.

    Generates a formatted ASCII table suitable for terminal display
    with holdings, summary statistics, and alerts.

    Args:
        snapshot: PortfolioSnapshot to format.

    Returns:
        Multi-line string with formatted snapshot display.

    Example:
        >>> text = format_snapshot_text(snapshot)
        >>> print(text)
    """
    lines: List[str] = []
    width = 60

    # Header
    lines.append("=" * width)
    lines.append(f"PORTFOLIO SNAPSHOT  {snapshot.timestamp[:19]}")
    lines.append("=" * width)

    # Summary line 1
    total_val = _format_currency(snapshot.total_value, 14)
    daily_pnl = _format_currency(snapshot.daily_pnl, 12)
    daily_pct = _format_pct(snapshot.daily_pnl_pct, 7)
    lines.append(f"Total Value: {total_val}    Daily P&L: {daily_pnl} ({daily_pct})")

    # Summary line 2
    total_pnl = _format_currency(snapshot.total_pnl, 14)
    total_pct = _format_pct(snapshot.total_pnl_pct, 7)
    lines.append(f"Total P&L:   {total_pnl} ({total_pct})")

    lines.append("")

    # Holdings table
    if snapshot.holdings:
        lines.append("HOLDINGS")
        lines.append("-" * width)

        # Table header
        header = (
            f"{'Asset':<8} {'Qty':>8} {'Price':>10} "
            f"{'Value':>12} {'Weight':>8} {'Day Ret':>8}"
        )
        lines.append(header)
        lines.append("-" * width)

        # Table rows
        for h in snapshot.holdings:
            qty_str = f"{h.quantity:,.0f}" if h.quantity == int(h.quantity) else f"{h.quantity:,.2f}"
            row = (
                f"{h.asset_id:<8} {qty_str:>8} "
                f"${h.current_price:>9,.2f} "
                f"${h.market_value:>11,.2f} "
                f"{h.weight * 100:>6.2f}% "
                f"{h.daily_return * 100:>+7.2f}%"
            )
            lines.append(row)

        lines.append("")

    # Risk summary
    lines.append("RISK SUMMARY")
    lines.append("-" * width)

    sharpe = snapshot.risk_summary.get("sharpe_ratio", 0.0)
    sortino = snapshot.risk_summary.get("sortino_ratio", 0.0)
    max_dd = snapshot.risk_summary.get("max_drawdown", 0.0)
    var_95 = snapshot.risk_summary.get("var_95", 0.0)

    risk_line = (
        f"Sharpe: {sharpe:>5.2f}  "
        f"Sortino: {sortino:>5.2f}  "
        f"Max DD: {max_dd * 100:>5.1f}%  "
        f"VaR(95): {var_95 * 100:>5.2f}%"
    )
    lines.append(risk_line)

    lines.append("")

    # Alerts
    if snapshot.alerts:
        lines.append("ALERTS")
        lines.append("-" * width)
        for alert in snapshot.alerts:
            lines.append(f"! {alert}")
        lines.append("")

    lines.append("=" * width)

    return "\n".join(lines)


def format_snapshot_json(snapshot: PortfolioSnapshot) -> Dict[str, Any]:
    """Convert portfolio snapshot to JSON-serializable dictionary.

    Transforms all dataclass fields into primitive types suitable
    for JSON serialization.

    Args:
        snapshot: PortfolioSnapshot to convert.

    Returns:
        Dict that can be passed to json.dumps().

    Example:
        >>> data = format_snapshot_json(snapshot)
        >>> json_str = json.dumps(data, indent=2)
    """
    holdings_list = [
        {
            "asset_id": h.asset_id,
            "quantity": h.quantity,
            "current_price": h.current_price,
            "market_value": h.market_value,
            "weight": h.weight,
            "daily_return": h.daily_return,
            "total_return": h.total_return,
            "unrealized_pnl": h.unrealized_pnl,
        }
        for h in snapshot.holdings
    ]

    return {
        "timestamp": snapshot.timestamp,
        "total_value": snapshot.total_value,
        "daily_pnl": snapshot.daily_pnl,
        "daily_pnl_pct": snapshot.daily_pnl_pct,
        "total_pnl": snapshot.total_pnl,
        "total_pnl_pct": snapshot.total_pnl_pct,
        "holdings": holdings_list,
        "risk_summary": snapshot.risk_summary.copy(),
        "alerts": snapshot.alerts.copy(),
    }


def snapshot_to_dataframe(snapshot: PortfolioSnapshot) -> pl.DataFrame:
    """Convert holdings to a Polars DataFrame.

    Useful for further analysis or export to various formats.

    Args:
        snapshot: PortfolioSnapshot with holdings.

    Returns:
        DataFrame with one row per holding.

    Example:
        >>> df = snapshot_to_dataframe(snapshot)
        >>> df.write_csv("holdings.csv")
    """
    if not snapshot.holdings:
        return pl.DataFrame({
            "asset_id": [],
            "quantity": [],
            "current_price": [],
            "market_value": [],
            "weight": [],
            "daily_return": [],
            "total_return": [],
            "unrealized_pnl": [],
        })

    return pl.DataFrame({
        "asset_id": [h.asset_id for h in snapshot.holdings],
        "quantity": [h.quantity for h in snapshot.holdings],
        "current_price": [h.current_price for h in snapshot.holdings],
        "market_value": [h.market_value for h in snapshot.holdings],
        "weight": [h.weight for h in snapshot.holdings],
        "daily_return": [h.daily_return for h in snapshot.holdings],
        "total_return": [h.total_return for h in snapshot.holdings],
        "unrealized_pnl": [h.unrealized_pnl for h in snapshot.holdings],
    })


def compare_snapshots(
    current: PortfolioSnapshot,
    previous: PortfolioSnapshot,
) -> Dict[str, Any]:
    """Compare two portfolio snapshots to show changes.

    Calculates deltas between snapshots for tracking portfolio evolution.

    Args:
        current: Current (newer) snapshot.
        previous: Previous (older) snapshot.

    Returns:
        Dict with change metrics including value change, new/closed positions,
        and weight changes.

    Example:
        >>> changes = compare_snapshots(today_snapshot, yesterday_snapshot)
        >>> print(f"Value changed by: ${changes['value_change']:,.2f}")
    """
    # Value changes
    value_change = current.total_value - previous.total_value
    value_change_pct = (
        value_change / previous.total_value
        if previous.total_value != 0 else 0.0
    )

    # Position changes
    current_assets = {h.asset_id for h in current.holdings}
    previous_assets = {h.asset_id for h in previous.holdings}

    new_positions = current_assets - previous_assets
    closed_positions = previous_assets - current_assets
    continuing_positions = current_assets & previous_assets

    # Weight changes for continuing positions
    prev_weights = {h.asset_id: h.weight for h in previous.holdings}
    curr_weights = {h.asset_id: h.weight for h in current.holdings}

    weight_changes = {}
    for asset_id in continuing_positions:
        change = curr_weights.get(asset_id, 0) - prev_weights.get(asset_id, 0)
        if abs(change) > 0.001:  # Only significant changes
            weight_changes[asset_id] = change

    return {
        "value_change": value_change,
        "value_change_pct": value_change_pct,
        "new_positions": list(new_positions),
        "closed_positions": list(closed_positions),
        "weight_changes": weight_changes,
        "pnl_change": current.total_pnl - previous.total_pnl,
        "sharpe_change": (
            current.risk_summary.get("sharpe_ratio", 0)
            - previous.risk_summary.get("sharpe_ratio", 0)
        ),
        "drawdown_change": (
            current.risk_summary.get("max_drawdown", 0)
            - previous.risk_summary.get("max_drawdown", 0)
        ),
    }
