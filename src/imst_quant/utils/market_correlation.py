"""Market correlation and beta calculation utilities.

This module provides functions to calculate portfolio beta, market correlation,
and related metrics for comparing portfolio performance against benchmarks.

Functions:
    calculate_beta: Portfolio beta relative to market
    calculate_alpha: Jensen's alpha (risk-adjusted excess return)
    calculate_correlation: Pearson correlation with benchmark
    calculate_treynor_ratio: Treynor ratio (excess return per unit of beta)
    calculate_information_ratio: Information ratio vs benchmark
    calculate_tracking_error: Standard deviation of excess returns
    calculate_market_metrics: Comprehensive market-relative metrics

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.market_correlation import calculate_market_metrics
    >>> portfolio = pl.Series("returns", [0.01, -0.02, 0.015, 0.02])
    >>> market = pl.Series("returns", [0.008, -0.015, 0.012, 0.018])
    >>> metrics = calculate_market_metrics(portfolio, market)
    >>> print(f"Beta: {metrics['beta']:.4f}")
    >>> print(f"Alpha: {metrics['alpha']:.4%}")
"""

from typing import Dict, Tuple, Union

import numpy as np
import polars as pl


def calculate_beta(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
) -> float:
    """Calculate portfolio beta relative to market.

    Beta measures the portfolio's sensitivity to market movements.
    Beta = 1.0 means the portfolio moves in line with the market.
    Beta > 1.0 means more volatile than market.
    Beta < 1.0 means less volatile than market.

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market/benchmark returns.

    Returns:
        Portfolio beta coefficient.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> market = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> beta = calculate_beta(portfolio, market)
        >>> print(f"Beta: {beta:.4f}")
        Beta: 1.1250
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length")

    if len(portfolio_returns) < 2:
        return 0.0

    # Calculate covariance and variance
    portfolio_np = portfolio_returns.to_numpy()
    market_np = market_returns.to_numpy()

    covariance = np.cov(portfolio_np, market_np)[0, 1]
    market_variance = np.var(market_np, ddof=1)

    if market_variance == 0:
        return 0.0

    return float(covariance / market_variance)


def calculate_alpha(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Jensen's alpha (risk-adjusted excess return).

    Alpha represents the portfolio's excess return over what would be
    expected given its beta and the market return (CAPM).

    Positive alpha indicates outperformance relative to risk taken.

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market/benchmark returns.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Annualized alpha.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> market = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> alpha = calculate_alpha(portfolio, market, risk_free_rate=0.0001)
        >>> print(f"Alpha: {alpha:.4%}")
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length")

    if len(portfolio_returns) == 0:
        return 0.0

    beta = calculate_beta(portfolio_returns, market_returns)

    portfolio_excess = portfolio_returns - risk_free_rate
    market_excess = market_returns - risk_free_rate

    portfolio_avg = float(portfolio_excess.mean()) * periods_per_year
    market_avg = float(market_excess.mean()) * periods_per_year

    # Jensen's alpha = Portfolio Return - (Risk-free + Beta * (Market Return - Risk-free))
    alpha = portfolio_avg - (beta * market_avg)

    return float(alpha)


def calculate_correlation(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
) -> float:
    """Calculate Pearson correlation between portfolio and market.

    Correlation ranges from -1 to +1:
    +1 = perfect positive correlation
    0 = no correlation
    -1 = perfect negative correlation

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market/benchmark returns.

    Returns:
        Correlation coefficient.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> market = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> corr = calculate_correlation(portfolio, market)
        >>> print(f"Correlation: {corr:.4f}")
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length")

    if len(portfolio_returns) < 2:
        return 0.0

    portfolio_np = portfolio_returns.to_numpy()
    market_np = market_returns.to_numpy()

    correlation_matrix = np.corrcoef(portfolio_np, market_np)
    return float(correlation_matrix[0, 1])


def calculate_tracking_error(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate tracking error (annualized standard deviation of excess returns).

    Tracking error measures how closely a portfolio follows its benchmark.
    Lower tracking error indicates closer tracking.

    Args:
        portfolio_returns: Series of portfolio returns.
        benchmark_returns: Series of benchmark returns.
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Annualized tracking error.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> benchmark = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> te = calculate_tracking_error(portfolio, benchmark)
        >>> print(f"Tracking Error: {te:.4%}")
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have the same length")

    if len(portfolio_returns) == 0:
        return 0.0

    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = float(excess_returns.std()) * np.sqrt(periods_per_year)

    return tracking_error


def calculate_information_ratio(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate information ratio (excess return / tracking error).

    Information ratio measures risk-adjusted active return relative to
    a benchmark. Higher is better.

    IR = (Portfolio Return - Benchmark Return) / Tracking Error

    Args:
        portfolio_returns: Series of portfolio returns.
        benchmark_returns: Series of benchmark returns.
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Information ratio.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> benchmark = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> ir = calculate_information_ratio(portfolio, benchmark)
        >>> print(f"Information Ratio: {ir:.4f}")
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have the same length")

    if len(portfolio_returns) == 0:
        return 0.0

    excess_returns = portfolio_returns - benchmark_returns
    excess_mean = float(excess_returns.mean()) * periods_per_year

    tracking_error = calculate_tracking_error(
        portfolio_returns, benchmark_returns, periods_per_year
    )

    if tracking_error == 0:
        return 0.0

    return excess_mean / tracking_error


def calculate_treynor_ratio(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Treynor ratio (excess return per unit of systematic risk).

    Treynor ratio = (Portfolio Return - Risk-free) / Beta

    Similar to Sharpe ratio but uses beta instead of total volatility,
    measuring return per unit of systematic (market) risk.

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market returns.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Annualized Treynor ratio.

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> market = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> treynor = calculate_treynor_ratio(portfolio, market)
        >>> print(f"Treynor Ratio: {treynor:.4f}")
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length")

    if len(portfolio_returns) == 0:
        return 0.0

    beta = calculate_beta(portfolio_returns, market_returns)

    if beta == 0:
        return 0.0

    portfolio_excess = portfolio_returns - risk_free_rate
    avg_excess = float(portfolio_excess.mean()) * periods_per_year

    return avg_excess / beta


def calculate_market_metrics(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive market-relative performance metrics.

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market/benchmark returns.
        risk_free_rate: Daily risk-free rate (default: 0).
        periods_per_year: Number of periods per year (default: 252).

    Returns:
        Dictionary containing:
            - beta: Portfolio beta
            - alpha: Jensen's alpha (annualized)
            - correlation: Pearson correlation
            - r_squared: R-squared (correlation squared)
            - tracking_error: Annualized tracking error
            - information_ratio: Information ratio
            - treynor_ratio: Treynor ratio
            - systematic_risk: Beta * Market Volatility
            - idiosyncratic_risk: Portfolio volatility - Systematic risk

    Example:
        >>> portfolio = pl.Series([0.01, -0.02, 0.015, 0.02])
        >>> market = pl.Series([0.008, -0.015, 0.012, 0.018])
        >>> metrics = calculate_market_metrics(portfolio, market)
        >>> for key, value in metrics.items():
        ...     print(f"{key}: {value:.4f}")
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length")

    if len(portfolio_returns) == 0:
        return {
            "beta": 0.0,
            "alpha": 0.0,
            "correlation": 0.0,
            "r_squared": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "treynor_ratio": 0.0,
            "systematic_risk": 0.0,
            "idiosyncratic_risk": 0.0,
        }

    beta = calculate_beta(portfolio_returns, market_returns)
    correlation = calculate_correlation(portfolio_returns, market_returns)
    r_squared = correlation ** 2

    market_vol = float(market_returns.std()) * np.sqrt(periods_per_year)
    portfolio_vol = float(portfolio_returns.std()) * np.sqrt(periods_per_year)

    systematic_risk = abs(beta) * market_vol
    idiosyncratic_risk = np.sqrt(max(0, portfolio_vol**2 - systematic_risk**2))

    return {
        "beta": beta,
        "alpha": calculate_alpha(
            portfolio_returns, market_returns, risk_free_rate, periods_per_year
        ),
        "correlation": correlation,
        "r_squared": r_squared,
        "tracking_error": calculate_tracking_error(
            portfolio_returns, market_returns, periods_per_year
        ),
        "information_ratio": calculate_information_ratio(
            portfolio_returns, market_returns, periods_per_year
        ),
        "treynor_ratio": calculate_treynor_ratio(
            portfolio_returns, market_returns, risk_free_rate, periods_per_year
        ),
        "systematic_risk": systematic_risk,
        "idiosyncratic_risk": idiosyncratic_risk,
    }
