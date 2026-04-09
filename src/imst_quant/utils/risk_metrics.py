"""Risk metrics calculations for portfolio performance analysis.

This module provides standard quantitative finance risk metrics including
Sharpe ratio, Sortino ratio, maximum drawdown, Value at Risk (VaR),
and Calmar ratio. All metrics work with Polars DataFrames for performance.

Functions:
    sharpe_ratio: Risk-adjusted return metric (excess return / volatility)
    sortino_ratio: Downside-risk-adjusted return metric
    max_drawdown: Maximum peak-to-trough decline
    value_at_risk: Historical VaR at specified confidence level
    calmar_ratio: Return over maximum drawdown ratio
    calculate_all_metrics: Compute all risk metrics at once

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.risk_metrics import calculate_all_metrics
    >>> returns = pl.Series("returns", [0.01, -0.02, 0.015, -0.005, 0.02])
    >>> metrics = calculate_all_metrics(returns, risk_free_rate=0.0001)
    >>> print(f"Sharpe: {metrics['sharpe']:.4f}")
"""

from typing import Dict, Union

import numpy as np
import polars as pl


def sharpe_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    return_col: str = "returns",
) -> float:
    """Calculate annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted returns by dividing excess
    returns (above risk-free rate) by standard deviation of returns.

    Args:
        returns: Series or DataFrame containing return data.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year for annualization
            (252 for daily, 52 for weekly, 12 for monthly).
        return_col: Column name if returns is a DataFrame.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005])
        >>> sharpe = sharpe_ratio(returns, risk_free_rate=0.0001)
        >>> print(f"Sharpe Ratio: {sharpe:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    excess_returns = ret_series - risk_free_rate
    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std()

    if std_dev is None or std_dev == 0:
        return 0.0

    return float((mean_excess / std_dev) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    return_col: str = "returns",
) -> float:
    """Calculate annualized Sortino ratio.

    The Sortino ratio is similar to Sharpe but uses downside deviation
    instead of total standard deviation, penalizing only negative volatility.

    Args:
        returns: Series or DataFrame containing return data.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year for annualization.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005])
        >>> sortino = sortino_ratio(returns, risk_free_rate=0.0001)
        >>> print(f"Sortino Ratio: {sortino:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    excess_returns = ret_series - risk_free_rate
    mean_excess = excess_returns.mean()

    # Calculate downside deviation (std of negative returns only)
    downside_returns = excess_returns.filter(excess_returns < 0)

    if downside_returns.len() == 0:
        # No negative returns = infinite Sortino (cap at large value)
        return 100.0 if mean_excess > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std is None or downside_std == 0:
        return 0.0

    return float((mean_excess / downside_std) * np.sqrt(periods_per_year))


def max_drawdown(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
) -> float:
    """Calculate maximum drawdown from peak equity.

    Maximum drawdown measures the largest peak-to-trough decline in
    cumulative returns, expressed as a positive decimal.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.15 = 15% drawdown).

    Example:
        >>> returns = pl.Series([0.05, -0.10, 0.02, -0.15, 0.08])
        >>> mdd = max_drawdown(returns)
        >>> print(f"Max Drawdown: {mdd:.2%}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Calculate cumulative returns (equity curve)
    cumulative = (1 + ret_series).cum_prod()

    # Calculate running maximum
    running_max = cumulative.cum_max()

    # Drawdown at each point
    drawdowns = (running_max - cumulative) / running_max

    max_dd = drawdowns.max()
    return float(max_dd) if max_dd is not None else 0.0


def value_at_risk(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    return_col: str = "returns",
) -> float:
    """Calculate historical Value at Risk (VaR).

    VaR estimates the maximum expected loss over a given time period
    at a specified confidence level using historical returns distribution.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.95 = 95%).
        return_col: Column name if returns is a DataFrame.

    Returns:
        VaR as a positive decimal (loss amount at confidence level).

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.05, 0.02])
        >>> var = value_at_risk(returns, confidence_level=0.95)
        >>> print(f"95% VaR: {var:.2%}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # VaR is the (1 - confidence) quantile of returns
    alpha = 1 - confidence_level
    var = ret_series.quantile(alpha, interpolation="linear")

    return float(-var) if var is not None else 0.0


def calmar_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    periods_per_year: int = 252,
    return_col: str = "returns",
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    The Calmar ratio measures return relative to maximum drawdown risk,
    useful for evaluating strategies with significant drawdown periods.

    Args:
        returns: Series or DataFrame containing return data.
        periods_per_year: Number of periods per year for annualization.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005])
        >>> calmar = calmar_ratio(returns)
        >>> print(f"Calmar Ratio: {calmar:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Annualized return
    mean_return = ret_series.mean()
    if mean_return is None:
        return 0.0
    annualized_return = mean_return * periods_per_year

    # Max drawdown
    mdd = max_drawdown(ret_series)

    if mdd == 0:
        return 100.0 if annualized_return > 0 else 0.0

    return float(annualized_return / mdd)


def calculate_all_metrics(
    returns: Union[pl.Series, pl.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    var_confidence: float = 0.95,
    return_col: str = "returns",
) -> Dict[str, float]:
    """Calculate all risk metrics at once.

    Convenience function that computes Sharpe, Sortino, max drawdown,
    VaR, and Calmar ratio in a single call.

    Args:
        returns: Series or DataFrame containing return data.
        risk_free_rate: Daily risk-free rate for Sharpe/Sortino.
        periods_per_year: Trading periods per year for annualization.
        var_confidence: Confidence level for VaR calculation.
        return_col: Column name if returns is a DataFrame.

    Returns:
        Dictionary with keys: sharpe, sortino, max_drawdown, var, calmar,
        total_return, annualized_return, volatility.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> metrics = calculate_all_metrics(returns)
        >>> for name, value in metrics.items():
        ...     print(f"{name}: {value:.4f}")
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
    else:
        ret_series = returns

    # Calculate cumulative return
    total_return = float((1 + ret_series).product() - 1)

    # Calculate annualized return
    mean_return = ret_series.mean()
    annualized_return = float(mean_return * periods_per_year) if mean_return else 0.0

    # Calculate annualized volatility
    std_dev = ret_series.std()
    volatility = float(std_dev * np.sqrt(periods_per_year)) if std_dev else 0.0

    return {
        "sharpe": sharpe_ratio(ret_series, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(ret_series, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(ret_series),
        "var": value_at_risk(ret_series, var_confidence),
        "calmar": calmar_ratio(ret_series, periods_per_year),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
    }
