"""Tail risk analysis and extreme event quantification.

This module provides tools for analyzing tail risk, extreme events,
and downside protection strategies. Includes Conditional Value at Risk (CVaR),
extreme value theory (EVT), and tail dependency metrics.

Functions:
    conditional_var: Calculate CVaR (Expected Shortfall)
    tail_ratio: Ratio of right tail to left tail
    omega_ratio: Probability-weighted ratio of gains to losses
    extreme_value_at_risk: EVT-based VaR for extreme quantiles
    tail_dependency: Measure tail dependency between assets
    stress_test_scenarios: Apply historical/hypothetical stress scenarios

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.tail_risk import conditional_var, tail_ratio
    >>> returns = pl.Series("returns", [0.01, -0.03, 0.02, -0.05, 0.015])
    >>> cvar = conditional_var(returns, confidence_level=0.95)
    >>> print(f"95% CVaR: {cvar:.2%}")
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy import stats


def conditional_var(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    return_col: str = "returns",
) -> float:
    """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR measures the expected loss given that the loss exceeds VaR.
    It's a coherent risk measure that captures tail risk better than VaR.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.95 = 95%).
        return_col: Column name if returns is a DataFrame.

    Returns:
        CVaR as a positive decimal (average loss in worst (1-confidence)% cases).

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.05, 0.02])
        >>> cvar = conditional_var(returns, confidence_level=0.95)
        >>> print(f"95% CVaR: {cvar:.2%}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Calculate VaR threshold
    alpha = 1 - confidence_level
    var_threshold = ret_series.quantile(alpha, interpolation="linear")

    # CVaR is the mean of returns below VaR
    tail_losses = ret_series.filter(ret_series <= var_threshold)

    if tail_losses.len() == 0:
        return 0.0

    cvar = tail_losses.mean()
    return float(-cvar) if cvar is not None else 0.0


def tail_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    percentile: float = 0.95,
    return_col: str = "returns",
) -> float:
    """Calculate tail ratio (ratio of right tail to left tail).

    Tail ratio measures the asymmetry of returns distribution.
    A value > 1 indicates fatter right tail (more upside outliers).

    Args:
        returns: Series or DataFrame containing return data.
        percentile: Percentile threshold for tails (default: 0.95).
        return_col: Column name if returns is a DataFrame.

    Returns:
        Tail ratio. Returns 0.0 if left tail is zero.

    Example:
        >>> returns = pl.Series([0.05, -0.03, 0.02, -0.04, 0.08])
        >>> tr = tail_ratio(returns, percentile=0.95)
        >>> print(f"Tail Ratio: {tr:.2f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Right tail (gains)
    right_threshold = ret_series.quantile(percentile, interpolation="linear")
    right_tail = ret_series.filter(ret_series >= right_threshold).mean()

    # Left tail (losses)
    left_threshold = ret_series.quantile(1 - percentile, interpolation="linear")
    left_tail = ret_series.filter(ret_series <= left_threshold).mean()

    if left_tail is None or abs(left_tail) < 1e-10:
        return 0.0

    if right_tail is None:
        return 0.0

    return float(abs(right_tail / left_tail))


def omega_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    threshold: float = 0.0,
    return_col: str = "returns",
) -> float:
    """Calculate Omega ratio (probability-weighted gains over losses).

    Omega ratio is the probability-weighted ratio of gains above a threshold
    to losses below the threshold. It captures the entire return distribution.

    Args:
        returns: Series or DataFrame containing return data.
        threshold: Return threshold (default: 0.0 for MAR = 0%).
        return_col: Column name if returns is a DataFrame.

    Returns:
        Omega ratio. Returns 0.0 if no losses below threshold.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> omega = omega_ratio(returns, threshold=0.0)
        >>> print(f"Omega Ratio: {omega:.2f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    gains = ret_series.filter(ret_series > threshold) - threshold
    losses = threshold - ret_series.filter(ret_series < threshold)

    sum_gains = gains.sum() if gains.len() > 0 else 0.0
    sum_losses = losses.sum() if losses.len() > 0 else 0.0

    if sum_losses == 0:
        return float("inf") if sum_gains > 0 else 0.0

    return float(sum_gains / sum_losses)


def extreme_value_at_risk(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.99,
    block_size: int = 21,
    return_col: str = "returns",
) -> float:
    """Calculate VaR using Extreme Value Theory (EVT).

    Uses the Block Maxima method (GEV distribution) to estimate VaR
    at extreme confidence levels (e.g., 99%, 99.9%) where historical
    data is sparse.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.99 = 99%).
        block_size: Block size for maxima extraction (default: 21 days).
        return_col: Column name if returns is a DataFrame.

    Returns:
        EVT-based VaR as a positive decimal.

    Example:
        >>> returns = pl.Series(np.random.randn(1000) * 0.02)
        >>> evt_var = extreme_value_at_risk(returns, confidence_level=0.99)
        >>> print(f"99% EVT VaR: {evt_var:.2%}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Convert to numpy for scipy
    ret_np = ret_series.to_numpy()

    # Extract block minima (we care about losses)
    n_blocks = len(ret_np) // block_size
    if n_blocks < 10:
        # Fall back to standard VaR if insufficient data
        var = ret_series.quantile(1 - confidence_level, interpolation="linear")
        return float(-var) if var is not None else 0.0

    block_minima = []
    for i in range(n_blocks):
        block = ret_np[i * block_size : (i + 1) * block_size]
        if len(block) > 0:
            block_minima.append(block.min())

    # Fit Generalized Extreme Value (GEV) distribution
    try:
        shape, loc, scale = stats.genextreme.fit(block_minima)

        # Calculate VaR at the desired confidence level
        # GEV quantile for lower tail
        var = stats.genextreme.ppf(1 - confidence_level, shape, loc=loc, scale=scale)
        return float(-var)

    except Exception:
        # Fall back to empirical quantile
        var = ret_series.quantile(1 - confidence_level, interpolation="linear")
        return float(-var) if var is not None else 0.0


def tail_dependency(
    returns1: pl.Series,
    returns2: pl.Series,
    quantile: float = 0.95,
) -> Tuple[float, float]:
    """Measure tail dependency between two asset return series.

    Tail dependency measures the probability of joint extreme events.
    Useful for understanding correlation breakdown during crises.

    Args:
        returns1: Return series for first asset.
        returns2: Return series for second asset.
        quantile: Quantile threshold for defining tail events.

    Returns:
        Tuple of (lower_tail_dependency, upper_tail_dependency).

    Example:
        >>> returns_a = pl.Series([0.01, -0.03, 0.02, -0.05, 0.015])
        >>> returns_b = pl.Series([0.015, -0.025, 0.018, -0.048, 0.012])
        >>> lower, upper = tail_dependency(returns_a, returns_b)
        >>> print(f"Lower tail dep: {lower:.2f}, Upper tail dep: {upper:.2f}")
    """
    # Align series
    if len(returns1) != len(returns2):
        raise ValueError("Return series must have the same length")

    r1 = returns1.to_numpy()
    r2 = returns2.to_numpy()

    # Lower tail (both assets experiencing losses)
    lower_threshold_1 = float(returns1.quantile(1 - quantile, interpolation="linear"))
    lower_threshold_2 = float(returns2.quantile(1 - quantile, interpolation="linear"))

    both_in_lower_tail = np.sum((r1 <= lower_threshold_1) & (r2 <= lower_threshold_2))
    either_in_lower_tail = np.sum((r1 <= lower_threshold_1) | (r2 <= lower_threshold_2))

    lower_tail_dep = (
        float(both_in_lower_tail / either_in_lower_tail)
        if either_in_lower_tail > 0
        else 0.0
    )

    # Upper tail (both assets experiencing gains)
    upper_threshold_1 = float(returns1.quantile(quantile, interpolation="linear"))
    upper_threshold_2 = float(returns2.quantile(quantile, interpolation="linear"))

    both_in_upper_tail = np.sum((r1 >= upper_threshold_1) & (r2 >= upper_threshold_2))
    either_in_upper_tail = np.sum((r1 >= upper_threshold_1) | (r2 >= upper_threshold_2))

    upper_tail_dep = (
        float(both_in_upper_tail / either_in_upper_tail)
        if either_in_upper_tail > 0
        else 0.0
    )

    return lower_tail_dep, upper_tail_dep


def stress_test_scenarios(
    returns: Union[pl.Series, pl.DataFrame],
    scenarios: Optional[Dict[str, float]] = None,
    return_col: str = "returns",
) -> Dict[str, float]:
    """Apply stress test scenarios to portfolio returns.

    Calculates portfolio loss under various historical or hypothetical
    stress scenarios.

    Args:
        returns: Series or DataFrame containing return data.
        scenarios: Dict of scenario names to market shocks (e.g., {"2008 Crisis": -0.50}).
                  If None, uses default historical scenarios.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary mapping scenario names to estimated portfolio losses.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> scenarios = {"Mild Shock": -0.10, "Severe Shock": -0.30}
        >>> losses = stress_test_scenarios(returns, scenarios=scenarios)
        >>> print(losses)
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    if scenarios is None:
        # Default historical stress scenarios
        scenarios = {
            "2008 Financial Crisis": -0.50,
            "2020 COVID Crash": -0.35,
            "1987 Black Monday": -0.22,
            "2000 Dot-com Bust": -0.40,
            "Mild Recession": -0.15,
            "Severe Recession": -0.30,
        }

    # Calculate portfolio beta (simplified assumption: beta = 1 if no benchmark)
    # In practice, you'd regress against a market index
    portfolio_volatility = float(ret_series.std()) if ret_series.std() else 0.0
    market_volatility = 0.02  # Assumed market daily vol

    beta = portfolio_volatility / market_volatility if market_volatility > 0 else 1.0

    stress_results = {}
    for scenario_name, market_shock in scenarios.items():
        # Estimated portfolio loss = beta * market_shock
        portfolio_shock = beta * market_shock
        stress_results[scenario_name] = float(portfolio_shock)

    return stress_results


def calculate_all_tail_metrics(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    return_col: str = "returns",
) -> Dict[str, float]:
    """Calculate all tail risk metrics at once.

    Convenience function that computes CVaR, tail ratio, Omega ratio,
    and EVT VaR in a single call.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level for CVaR and VaR.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with keys: cvar, tail_ratio, omega_ratio, evt_var.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> metrics = calculate_all_tail_metrics(returns)
        >>> for name, value in metrics.items():
        ...     print(f"{name}: {value:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    return {
        "cvar": conditional_var(ret_series, confidence_level),
        "tail_ratio": tail_ratio(ret_series, percentile=confidence_level),
        "omega_ratio": omega_ratio(ret_series, threshold=0.0),
        "evt_var": extreme_value_at_risk(ret_series, confidence_level),
    }
