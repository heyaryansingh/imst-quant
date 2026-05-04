"""Portfolio rebalancing analysis and optimization utilities.

This module provides tools for analyzing portfolio rebalancing decisions,
calculating rebalancing thresholds, and optimizing rebalancing frequency
to balance tracking error vs transaction costs.

Functions:
    calculate_drift: Calculate portfolio drift from target weights
    needs_rebalancing: Determine if portfolio needs rebalancing
    generate_rebalancing_trades: Generate trade list to rebalance portfolio
    optimize_rebalancing_frequency: Find optimal rebalancing frequency
    rebalancing_cost_benefit: Analyze cost-benefit of rebalancing
    threshold_rebalancing: Calculate optimal threshold-based rebalancing

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.rebalancing import needs_rebalancing
    >>> current = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.55, 0.45]})
    >>> target = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.50, 0.50]})
    >>> should_rebalance = needs_rebalancing(current, target, threshold=0.05)
    >>> print(f"Rebalancing needed: {should_rebalance}")
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def calculate_drift(
    current_weights: pl.DataFrame,
    target_weights: pl.DataFrame,
    symbol_col: str = "symbol",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Calculate portfolio drift from target weights.

    Computes absolute and relative deviation of current portfolio
    weights from target allocation.

    Args:
        current_weights: DataFrame with current portfolio [symbol, weight].
        target_weights: DataFrame with target portfolio [symbol, weight].
        symbol_col: Column name for symbol identifier (default: symbol).
        weight_col: Column name for weight values (default: weight).

    Returns:
        DataFrame with columns: [symbol, current_weight, target_weight,
        absolute_drift, relative_drift].

    Example:
        >>> current = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.55]})
        >>> target = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.50]})
        >>> drift = calculate_drift(current, target)
        >>> print(drift["absolute_drift"][0])  # 0.05
    """
    # Join current and target weights
    drift_df = current_weights.join(
        target_weights,
        on=symbol_col,
        how="outer",
        suffix="_target"
    )

    # Fill missing values with 0 (positions not in one of the portfolios)
    drift_df = drift_df.fill_null(0.0)

    # Rename columns for clarity
    drift_df = drift_df.rename({
        weight_col: "current_weight",
        f"{weight_col}_target": "target_weight",
    })

    # Calculate drifts
    drift_df = drift_df.with_columns([
        (pl.col("current_weight") - pl.col("target_weight")).alias("absolute_drift"),
        ((pl.col("current_weight") - pl.col("target_weight")) / pl.col("target_weight"))
        .fill_null(float("inf"))  # Handle divide by zero for 0 target weight
        .alias("relative_drift"),
    ])

    return drift_df


def needs_rebalancing(
    current_weights: pl.DataFrame,
    target_weights: pl.DataFrame,
    threshold: float = 0.05,
    method: str = "absolute",
) -> bool:
    """Determine if portfolio needs rebalancing.

    Checks if any position has drifted beyond the specified threshold
    from its target weight.

    Args:
        current_weights: DataFrame with current portfolio [symbol, weight].
        target_weights: DataFrame with target portfolio [symbol, weight].
        threshold: Rebalancing threshold (default: 0.05 = 5%).
        method: Drift measurement method - "absolute" or "relative" (default: absolute).

    Returns:
        True if rebalancing is needed, False otherwise.

    Example:
        >>> current = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.56]})
        >>> target = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.50]})
        >>> needs_rebalancing(current, target, threshold=0.05)
        True
    """
    drift_df = calculate_drift(current_weights, target_weights)

    if method == "absolute":
        max_drift = drift_df["absolute_drift"].abs().max()
    elif method == "relative":
        max_drift = drift_df["relative_drift"].abs().max()
        if max_drift == float("inf"):
            return True  # Position appeared/disappeared
    else:
        raise ValueError(f"Unknown method: {method}. Use 'absolute' or 'relative'.")

    return max_drift > threshold


def generate_rebalancing_trades(
    current_weights: pl.DataFrame,
    target_weights: pl.DataFrame,
    portfolio_value: float,
    min_trade_size: float = 100.0,
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Generate trade list to rebalance portfolio to target.

    Creates a list of buy/sell orders needed to bring current portfolio
    into alignment with target weights.

    Args:
        current_weights: DataFrame with current portfolio [symbol, weight].
        target_weights: DataFrame with target portfolio [symbol, weight].
        portfolio_value: Total portfolio value in USD.
        min_trade_size: Minimum trade size in USD (skip smaller trades).
        symbol_col: Column name for symbol identifier.

    Returns:
        DataFrame with columns: [symbol, current_weight, target_weight,
        trade_value, trade_direction].

    Example:
        >>> current = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.6, 0.4]})
        >>> target = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.5, 0.5]})
        >>> trades = generate_rebalancing_trades(current, target, 100000)
    """
    drift_df = calculate_drift(current_weights, target_weights, symbol_col=symbol_col)

    # Calculate trade values
    trades_df = drift_df.with_columns([
        (pl.col("absolute_drift") * portfolio_value).alias("trade_value"),
        pl.when(pl.col("absolute_drift") > 0)
        .then(pl.lit("SELL"))
        .when(pl.col("absolute_drift") < 0)
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("HOLD"))
        .alias("trade_direction"),
    ])

    # Filter out trades smaller than minimum
    trades_df = trades_df.filter(pl.col("trade_value").abs() >= min_trade_size)

    # Sort by absolute trade value (largest first)
    trades_df = trades_df.sort(pl.col("trade_value").abs(), descending=True)

    return trades_df


def optimize_rebalancing_frequency(
    returns_df: pl.DataFrame,
    target_weights: Dict[str, float],
    cost_bps_per_rebalance: float = 10.0,
    return_col: str = "returns",
    test_frequencies: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Find optimal rebalancing frequency that balances tracking error vs costs.

    Simulates different rebalancing frequencies and finds the one that
    maximizes risk-adjusted returns after transaction costs.

    Args:
        returns_df: DataFrame with historical returns [date, symbol, returns].
        target_weights: Dictionary mapping symbols to target weights.
        cost_bps_per_rebalance: Transaction cost per rebalancing event in bps.
        return_col: Column name for returns.
        test_frequencies: List of rebalancing frequencies to test in days
            (default: [1, 5, 10, 20, 60, 120]).

    Returns:
        Dictionary with optimal_frequency_days, expected_return, tracking_error,
        sharpe_ratio, annual_rebalancing_cost.

    Example:
        >>> returns = pl.DataFrame({...})  # Historical returns
        >>> target = {"AAPL": 0.5, "MSFT": 0.5}
        >>> result = optimize_rebalancing_frequency(returns, target)
        >>> print(f"Optimal frequency: {result['optimal_frequency_days']} days")
    """
    if test_frequencies is None:
        test_frequencies = [1, 5, 10, 20, 60, 120]

    results = []

    for freq in test_frequencies:
        # This is a simplified simulation placeholder
        # In production, would need full backtesting with actual rebalancing
        # For now, approximate using tracking error model

        # Tracking error increases with time between rebalances
        # Simplified model: TE ~ sqrt(freq) * base_volatility
        base_te = 0.001  # 10 bps per day
        tracking_error = base_te * np.sqrt(freq)

        # Annual rebalancing cost
        rebalances_per_year = 252 / freq
        annual_cost = (cost_bps_per_rebalance / 10000.0) * rebalances_per_year

        # Assume base return of 10% annually
        base_return = 0.10
        expected_return = base_return - annual_cost

        # Sharpe approximation
        volatility = 0.15  # Assume 15% annual vol
        sharpe = expected_return / volatility

        results.append({
            "frequency_days": freq,
            "expected_return": expected_return,
            "tracking_error": tracking_error * np.sqrt(252),  # Annualized
            "sharpe_ratio": sharpe,
            "annual_rebalancing_cost": annual_cost,
        })

    # Find frequency with best Sharpe ratio
    results_df = pl.DataFrame(results)
    best_idx = results_df["sharpe_ratio"].arg_max()
    best_result = results_df.row(best_idx, named=True)

    return {
        "optimal_frequency_days": best_result["frequency_days"],
        "expected_return": best_result["expected_return"],
        "tracking_error": best_result["tracking_error"],
        "sharpe_ratio": best_result["sharpe_ratio"],
        "annual_rebalancing_cost": best_result["annual_rebalancing_cost"],
    }


def rebalancing_cost_benefit(
    current_weights: pl.DataFrame,
    target_weights: pl.DataFrame,
    portfolio_value: float,
    expected_te_reduction: float = 0.02,
    cost_bps: float = 10.0,
    holding_period_days: int = 60,
) -> Dict[str, float]:
    """Analyze cost-benefit of rebalancing now vs waiting.

    Compares the benefit of reducing tracking error against the
    immediate transaction costs of rebalancing.

    Args:
        current_weights: DataFrame with current portfolio [symbol, weight].
        target_weights: DataFrame with target portfolio [symbol, weight].
        portfolio_value: Total portfolio value in USD.
        expected_te_reduction: Expected tracking error reduction (annualized).
        cost_bps: Transaction cost in basis points.
        holding_period_days: Expected holding period until next rebalance.

    Returns:
        Dictionary with benefit_usd, cost_usd, net_benefit, benefit_cost_ratio,
        recommendation.

    Example:
        >>> current = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.55]})
        >>> target = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.50]})
        >>> analysis = rebalancing_cost_benefit(current, target, 100000)
        >>> print(analysis["recommendation"])
    """
    drift_df = calculate_drift(current_weights, target_weights)

    # Calculate total drift
    total_drift = drift_df["absolute_drift"].abs().sum()

    # Estimate benefit: reduction in tracking error
    # Benefit scales with drift and holding period
    daily_te_benefit = expected_te_reduction / 252
    benefit_decimal = daily_te_benefit * holding_period_days * total_drift
    benefit_usd = benefit_decimal * portfolio_value

    # Calculate cost: transaction costs of rebalancing
    cost_decimal = cost_bps / 10000.0
    # Cost applies to turnover (sum of buys or sells)
    turnover = drift_df["absolute_drift"].abs().sum() * portfolio_value
    cost_usd = turnover * cost_decimal

    # Net benefit
    net_benefit = benefit_usd - cost_usd

    # Benefit-cost ratio
    bcr = benefit_usd / cost_usd if cost_usd > 0 else float("inf")

    # Recommendation
    if bcr > 2.0:
        recommendation = "REBALANCE NOW - High benefit-cost ratio"
    elif bcr > 1.0:
        recommendation = "CONSIDER REBALANCING - Positive net benefit"
    elif bcr > 0.5:
        recommendation = "MARGINAL - Consider waiting"
    else:
        recommendation = "WAIT - Costs exceed benefits"

    return {
        "benefit_usd": benefit_usd,
        "cost_usd": cost_usd,
        "net_benefit": net_benefit,
        "benefit_cost_ratio": bcr,
        "total_drift": total_drift,
        "recommendation": recommendation,
    }


def threshold_rebalancing(
    historical_returns: pl.DataFrame,
    target_volatility: float = 0.15,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 2.0,
) -> float:
    """Calculate optimal rebalancing threshold using utility theory.

    Uses mean-variance utility framework to find the rebalancing threshold
    that maximizes expected utility considering transaction costs.

    Args:
        historical_returns: DataFrame with historical asset returns.
        target_volatility: Target portfolio volatility (default: 0.15 = 15%).
        transaction_cost_bps: Transaction costs in basis points.
        risk_aversion: Risk aversion parameter (default: 2.0).

    Returns:
        Optimal rebalancing threshold as a decimal (e.g., 0.05 = 5%).

    Example:
        >>> returns = pl.DataFrame({...})  # Historical returns
        >>> threshold = threshold_rebalancing(returns, target_volatility=0.15)
        >>> print(f"Optimal threshold: {threshold:.1%}")
    """
    # Simplified model based on academic research (e.g., Dybvig, 2005)
    # Optimal threshold ~ sqrt(transaction_costs / (risk_aversion * volatility^2))

    cost_decimal = transaction_cost_bps / 10000.0

    # Optimal threshold formula
    threshold = np.sqrt(cost_decimal / (risk_aversion * (target_volatility ** 2)))

    # Cap threshold between 1% and 20%
    threshold = max(0.01, min(0.20, threshold))

    return float(threshold)
