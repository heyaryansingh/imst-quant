"""Portfolio rebalancing strategy implementations.

This module provides various rebalancing strategies for portfolio management,
including calendar-based, threshold-based, and dynamic rebalancing approaches.

Functions:
    calendar_rebalance: Time-based rebalancing (monthly, quarterly, etc.)
    threshold_rebalance: Rebalance when allocations drift beyond threshold
    volatility_based_rebalance: Adjust frequency based on market volatility
    cost_aware_rebalance: Minimize transaction costs in rebalancing
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import polars as pl


def calendar_rebalance(
    df: pl.DataFrame,
    target_weights: Dict[str, float],
    frequency: str = "monthly",
    date_col: str = "date",
    asset_col: str = "asset",
    value_col: str = "value",
) -> pl.DataFrame:
    """Generate calendar-based rebalancing schedule.

    Args:
        df: DataFrame with portfolio positions over time
        target_weights: Dictionary mapping asset names to target weights
        frequency: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
        date_col: Column name for dates
        asset_col: Column name for asset identifiers
        value_col: Column name for position values

    Returns:
        DataFrame with rebalancing instructions including:
        - rebalance_date: Date of rebalancing
        - asset: Asset identifier
        - current_weight: Current portfolio weight
        - target_weight: Target weight
        - adjustment: Required position adjustment
    """
    if df.is_empty():
        return pl.DataFrame()

    # Determine rebalancing dates based on frequency
    df_with_period = df.with_columns(
        pl.col(date_col).dt.strftime(_get_period_format(frequency)).alias("period")
    )

    # Get first date of each period for rebalancing
    rebalance_dates = (
        df_with_period
        .group_by("period")
        .agg(pl.col(date_col).min().alias("rebalance_date"))
        .sort("rebalance_date")
    )

    # Calculate portfolio totals on rebalancing dates
    rebalancing_instructions = []

    for rebal_date in rebalance_dates["rebalance_date"]:
        # Get portfolio snapshot at rebalance date
        snapshot = df.filter(pl.col(date_col) == rebal_date)

        # Calculate current weights
        total_value = snapshot[value_col].sum()

        if total_value == 0:
            continue

        current_weights = (
            snapshot
            .group_by(asset_col)
            .agg((pl.col(value_col).sum() / total_value).alias("current_weight"))
        )

        # Calculate adjustments for each asset
        for asset, target_weight in target_weights.items():
            current = current_weights.filter(pl.col(asset_col) == asset)
            current_weight = current["current_weight"][0] if len(current) > 0 else 0.0

            adjustment = (target_weight - current_weight) * total_value

            rebalancing_instructions.append({
                "rebalance_date": rebal_date,
                "asset": asset,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "adjustment": adjustment,
            })

    return pl.DataFrame(rebalancing_instructions)


def threshold_rebalance(
    df: pl.DataFrame,
    target_weights: Dict[str, float],
    drift_threshold: float = 0.05,
    date_col: str = "date",
    asset_col: str = "asset",
    value_col: str = "value",
) -> pl.DataFrame:
    """Generate threshold-based rebalancing triggers.

    Rebalances when any asset's weight drifts from target by more than threshold.

    Args:
        df: DataFrame with portfolio positions over time
        target_weights: Dictionary mapping asset names to target weights
        drift_threshold: Maximum allowed drift from target (e.g., 0.05 = 5%)
        date_col: Column name for dates
        asset_col: Column name for asset identifiers
        value_col: Column name for position values

    Returns:
        DataFrame with rebalancing triggers when threshold is exceeded
    """
    if df.is_empty():
        return pl.DataFrame()

    # Calculate daily weights
    daily_totals = df.group_by(date_col).agg(pl.col(value_col).sum().alias("total_value"))

    df_with_totals = df.join(daily_totals, on=date_col)

    df_weights = df_with_totals.with_columns(
        (pl.col(value_col) / pl.col("total_value")).alias("weight")
    )

    # Check drift for each asset
    rebalance_signals = []

    for date in df_weights[date_col].unique().sort():
        snapshot = df_weights.filter(pl.col(date_col) == date)

        needs_rebalance = False
        max_drift = 0.0

        for asset, target_weight in target_weights.items():
            asset_data = snapshot.filter(pl.col(asset_col) == asset)
            current_weight = asset_data["weight"][0] if len(asset_data) > 0 else 0.0

            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)

            if drift > drift_threshold:
                needs_rebalance = True

        if needs_rebalance:
            total_value = snapshot["total_value"][0]

            for asset, target_weight in target_weights.items():
                asset_data = snapshot.filter(pl.col(asset_col) == asset)
                current_weight = asset_data["weight"][0] if len(asset_data) > 0 else 0.0

                adjustment = (target_weight - current_weight) * total_value

                rebalance_signals.append({
                    "rebalance_date": date,
                    "asset": asset,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift": abs(current_weight - target_weight),
                    "adjustment": adjustment,
                    "max_drift": max_drift,
                })

    return pl.DataFrame(rebalance_signals) if rebalance_signals else pl.DataFrame()


def volatility_based_rebalance(
    df: pl.DataFrame,
    target_weights: Dict[str, float],
    vol_window: int = 20,
    vol_threshold_high: float = 0.02,
    vol_threshold_low: float = 0.01,
    date_col: str = "date",
    asset_col: str = "asset",
    returns_col: str = "returns",
) -> pl.DataFrame:
    """Adjust rebalancing frequency based on market volatility.

    Higher volatility triggers more frequent rebalancing.

    Args:
        df: DataFrame with returns data
        target_weights: Dictionary mapping asset names to target weights
        vol_window: Rolling window for volatility calculation
        vol_threshold_high: High volatility threshold (daily rebalance)
        vol_threshold_low: Low volatility threshold (monthly rebalance)
        date_col: Column name for dates
        asset_col: Column name for asset identifiers
        returns_col: Column name for returns

    Returns:
        DataFrame with volatility-adjusted rebalancing schedule
    """
    if df.is_empty():
        return pl.DataFrame()

    # Calculate rolling volatility for each asset
    df_vol = (
        df
        .sort([asset_col, date_col])
        .with_columns([
            pl.col(returns_col)
            .rolling_std(window_size=vol_window)
            .over(asset_col)
            .alias("volatility")
        ])
    )

    # Calculate portfolio-level volatility (equally weighted for simplicity)
    portfolio_vol = (
        df_vol
        .group_by(date_col)
        .agg(pl.col("volatility").mean().alias("portfolio_vol"))
    )

    # Determine rebalancing frequency based on volatility
    rebalance_schedule = []

    for row in portfolio_vol.iter_rows(named=True):
        date = row[date_col]
        vol = row["portfolio_vol"]

        if vol is None:
            continue

        # Determine frequency based on volatility level
        if vol > vol_threshold_high:
            frequency = "daily"
        elif vol > vol_threshold_low:
            frequency = "weekly"
        else:
            frequency = "monthly"

        rebalance_schedule.append({
            "date": date,
            "portfolio_vol": vol,
            "frequency": frequency,
            "should_rebalance": _check_rebalance_day(date, frequency),
        })

    return pl.DataFrame(rebalance_schedule)


def cost_aware_rebalance(
    df: pl.DataFrame,
    target_weights: Dict[str, float],
    transaction_cost_pct: float = 0.001,
    min_improvement: float = 0.002,
    date_col: str = "date",
    asset_col: str = "asset",
    value_col: str = "value",
) -> pl.DataFrame:
    """Minimize transaction costs in rebalancing decisions.

    Only rebalances when expected benefit exceeds transaction costs.

    Args:
        df: DataFrame with portfolio positions
        target_weights: Dictionary mapping asset names to target weights
        transaction_cost_pct: Transaction cost as percentage (e.g., 0.001 = 0.1%)
        min_improvement: Minimum expected improvement to trigger rebalance
        date_col: Column name for dates
        asset_col: Column name for asset identifiers
        value_col: Column name for position values

    Returns:
        DataFrame with cost-efficient rebalancing instructions
    """
    if df.is_empty():
        return pl.DataFrame()

    # Calculate current weights and total value
    daily_totals = df.group_by(date_col).agg(pl.col(value_col).sum().alias("total_value"))

    df_with_totals = df.join(daily_totals, on=date_col)

    df_weights = df_with_totals.with_columns(
        (pl.col(value_col) / pl.col("total_value")).alias("weight")
    )

    cost_efficient_rebalancing = []

    for date in df_weights[date_col].unique().sort():
        snapshot = df_weights.filter(pl.col(date_col) == date)
        total_value = snapshot["total_value"][0]

        # Calculate total adjustment needed
        total_adjustment_value = 0.0
        adjustments = {}

        for asset, target_weight in target_weights.items():
            asset_data = snapshot.filter(pl.col(asset_col) == asset)
            current_weight = asset_data["weight"][0] if len(asset_data) > 0 else 0.0

            adjustment = (target_weight - current_weight) * total_value
            adjustments[asset] = adjustment
            total_adjustment_value += abs(adjustment)

        # Calculate transaction costs
        transaction_cost = total_adjustment_value * transaction_cost_pct

        # Estimate benefit (reduced tracking error)
        tracking_error = sum(
            abs(current_weight - target_weight)
            for asset, target_weight in target_weights.items()
            for current_weight in [
                snapshot.filter(pl.col(asset_col) == asset)["weight"][0]
                if len(snapshot.filter(pl.col(asset_col) == asset)) > 0
                else 0.0
            ]
        )

        expected_benefit = tracking_error * total_value

        # Only rebalance if benefit > cost + minimum improvement
        if expected_benefit > transaction_cost * (1 + min_improvement):
            for asset, adjustment in adjustments.items():
                current = snapshot.filter(pl.col(asset_col) == asset)
                current_weight = current["weight"][0] if len(current) > 0 else 0.0

                cost_efficient_rebalancing.append({
                    "rebalance_date": date,
                    "asset": asset,
                    "current_weight": current_weight,
                    "target_weight": target_weights[asset],
                    "adjustment": adjustment,
                    "transaction_cost": abs(adjustment) * transaction_cost_pct,
                    "expected_benefit": expected_benefit,
                    "net_benefit": expected_benefit - transaction_cost,
                })

    return pl.DataFrame(cost_efficient_rebalancing) if cost_efficient_rebalancing else pl.DataFrame()


def _get_period_format(frequency: str) -> str:
    """Get date format string for period grouping."""
    formats = {
        "daily": "%Y-%m-%d",
        "weekly": "%Y-W%W",
        "monthly": "%Y-%m",
        "quarterly": "%Y-Q%q",
        "yearly": "%Y",
    }
    return formats.get(frequency, "%Y-%m")


def _check_rebalance_day(date, frequency: str) -> bool:
    """Check if given date should trigger rebalancing for frequency."""
    if frequency == "daily":
        return True
    elif frequency == "weekly":
        return date.weekday() == 0  # Monday
    elif frequency == "monthly":
        return date.day == 1
    return False
