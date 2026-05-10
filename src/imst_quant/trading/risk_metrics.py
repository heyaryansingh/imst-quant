"""Risk metrics calculations for portfolio analysis.

This module provides comprehensive risk metrics including VaR, CVaR, maximum
drawdown, Sortino ratio, and Calmar ratio for portfolio performance evaluation.

Example:
    >>> from imst_quant.trading.risk_metrics import calculate_risk_metrics
    >>> import polars as pl
    >>> returns = pl.Series("returns", [0.01, -0.02, 0.015, -0.01, 0.02])
    >>> metrics = calculate_risk_metrics(returns)
    >>> print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
"""

from typing import Dict

import polars as pl


def calculate_max_drawdown(equity_curve: pl.Series) -> float:
    """Calculate maximum peak-to-trough decline.

    Args:
        equity_curve: Series of cumulative equity values.

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 = 15% drawdown).
    """
    if len(equity_curve) == 0:
        return 0.0

    running_max = equity_curve.cum_max()
    drawdown = (equity_curve - running_max) / running_max
    return abs(float(drawdown.min()))


def calculate_var(returns: pl.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) at specified confidence level.

    VaR represents the maximum expected loss at the given confidence level.

    Args:
        returns: Series of returns.
        confidence: Confidence level (default: 0.95 for 95% VaR).

    Returns:
        VaR as decimal (negative value representing loss).
    """
    if len(returns) == 0:
        return 0.0

    quantile = 1 - confidence
    return float(returns.quantile(quantile))


def calculate_cvar(returns: pl.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    CVaR represents the expected loss given that VaR has been exceeded.

    Args:
        returns: Series of returns.
        confidence: Confidence level (default: 0.95).

    Returns:
        CVaR as decimal (negative value representing expected loss).
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence)
    tail_losses = returns.filter(returns <= var)

    if len(tail_losses) == 0:
        return var

    return float(tail_losses.mean())


def calculate_sortino_ratio(
    returns: pl.Series, risk_free_rate: float = 0.0, target_return: float = 0.0
) -> float:
    """Calculate Sortino ratio using downside deviation.

    Sortino ratio penalizes only downside volatility, unlike Sharpe which
    penalizes all volatility.

    Args:
        returns: Series of returns.
        risk_free_rate: Risk-free rate to subtract (default: 0).
        target_return: Target return threshold (default: 0).

    Returns:
        Sortino ratio (higher is better).
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = float(excess_returns.mean())

    downside_returns = excess_returns.filter(excess_returns < target_return)
    if len(downside_returns) == 0:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = float(downside_returns.std())
    if downside_std == 0:
        return 0.0

    return mean_excess / downside_std


def calculate_calmar_ratio(returns: pl.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year for annualization (default: 252 trading days).

    Returns:
        Calmar ratio (higher is better).
    """
    if len(returns) == 0:
        return 0.0

    # Calculate annualized return
    total_return = (1 + returns).product() - 1
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Calculate max drawdown
    equity_curve = (1 + returns).cum_prod()
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float("inf") if annualized_return > 0 else 0.0

    return annualized_return / max_dd


def calculate_sharpe_ratio(
    returns: pl.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of returns.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = float(excess_returns.mean())
    std_excess = float(excess_returns.std())

    if std_excess == 0:
        return 0.0

    return (mean_excess / std_excess) * (periods_per_year**0.5)


def calculate_risk_metrics(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    var_confidence: float = 0.95,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive risk metrics for a return series.

    Args:
        returns: Series of returns.
        risk_free_rate: Daily risk-free rate (default: 0).
        var_confidence: Confidence level for VaR/CVaR (default: 0.95).
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Dictionary containing:
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Sortino ratio
            - calmar_ratio: Calmar ratio
            - max_drawdown: Maximum drawdown
            - var_95: Value at Risk at 95% confidence
            - cvar_95: Conditional VaR at 95% confidence
            - total_return: Cumulative return
            - volatility: Annualized volatility
    """
    if len(returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

    equity_curve = (1 + returns).cum_prod()
    total_return = float(equity_curve[-1] - 1)
    volatility = float(returns.std() * (periods_per_year**0.5))

    return {
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "var_95": calculate_var(returns, var_confidence),
        "cvar_95": calculate_cvar(returns, var_confidence),
        "total_return": total_return,
        "volatility": volatility,
    }
