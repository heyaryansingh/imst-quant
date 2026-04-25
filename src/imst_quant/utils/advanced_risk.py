"""Advanced risk metrics and calculations.

Extended risk analysis including tail risk, stress testing, and risk decomposition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats
import structlog

logger = structlog.get_logger()


def calculate_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> Dict[str, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Calculation method ('historical', 'parametric', 'cornish-fisher')

    Returns:
        Dict with VaR and CVaR values
    """
    returns_clean = returns.dropna()

    if method == "historical":
        var = returns_clean.quantile(1 - confidence_level)
        cvar = returns_clean[returns_clean <= var].mean()

    elif method == "parametric":
        mean = returns_clean.mean()
        std = returns_clean.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std
        # CVaR for normal distribution
        cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)

    elif method == "cornish-fisher":
        # Adjust for skewness and kurtosis
        mean = returns_clean.mean()
        std = returns_clean.std()
        skew = returns_clean.skew()
        kurt = returns_clean.kurtosis()

        z = stats.norm.ppf(1 - confidence_level)
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = mean + z_cf * std
        # Approximate CVaR
        cvar = mean - std * stats.norm.pdf(z_cf) / (1 - confidence_level)

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "var": float(var),
        "cvar": float(cvar),
        "confidence_level": confidence_level,
        "method": method,
    }


def tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive tail risk metrics.

    Args:
        returns: Series of returns

    Returns:
        Dict with tail risk metrics including skew, kurtosis, tail ratio
    """
    returns_clean = returns.dropna()

    # Basic tail statistics
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()  # Excess kurtosis

    # Tail ratio (right tail / left tail)
    right_tail = returns_clean.quantile(0.95)
    left_tail = returns_clean.quantile(0.05)
    tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else np.inf

    # Downside deviation (semi-deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0

    # Sortino ratio (assuming 0% risk-free rate)
    mean_return = returns_clean.mean()
    sortino = (mean_return / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0

    # Extreme value metrics (beyond 2 standard deviations)
    threshold = returns_clean.mean() - 2 * returns_clean.std()
    extreme_losses = returns_clean[returns_clean < threshold]
    extreme_loss_pct = len(extreme_losses) / len(returns_clean) if len(returns_clean) > 0 else 0

    return {
        "skewness": float(skewness),
        "excess_kurtosis": float(kurtosis),
        "tail_ratio": float(tail_ratio),
        "downside_deviation": float(downside_deviation),
        "sortino_ratio": float(sortino),
        "extreme_loss_pct": float(extreme_loss_pct),
        "avg_extreme_loss": float(extreme_losses.mean()) if len(extreme_losses) > 0 else 0,
    }


def rolling_risk_metrics(
    returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Calculate rolling risk metrics over time.

    Args:
        returns: Series of returns
        window: Rolling window size (default: 252 trading days)

    Returns:
        DataFrame with rolling volatility, Sharpe, VaR, and max drawdown
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol

    # Rolling VaR (95%)
    rolling_var = returns.rolling(window).quantile(0.05)

    # Rolling max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    rolling_dd = (cumulative - rolling_max) / rolling_max

    df = pd.DataFrame({
        "volatility": rolling_vol,
        "sharpe": rolling_sharpe,
        "var_95": rolling_var,
        "max_drawdown": rolling_dd.rolling(window).min(),
    })

    return df


def stress_test(
    returns: pd.Series,
    scenarios: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Perform stress testing with various market scenarios.

    Args:
        returns: Series of returns
        scenarios: Dict of scenario names to return shocks (e.g., {"crash": -0.20})

    Returns:
        Dict with stress test results for each scenario
    """
    if scenarios is None:
        scenarios = {
            "mild_correction": -0.05,
            "correction": -0.10,
            "bear_market": -0.20,
            "crash": -0.30,
            "black_swan": -0.40,
        }

    results = {}

    for scenario_name, shock in scenarios.items():
        # Apply shock to each return
        shocked_returns = returns + shock

        # Calculate metrics under stress
        total_return = (1 + shocked_returns).prod() - 1
        volatility = shocked_returns.std() * np.sqrt(252)
        sharpe = (shocked_returns.mean() / shocked_returns.std() * np.sqrt(252)) if shocked_returns.std() > 0 else 0

        # Maximum drawdown under stress
        cumulative = (1 + shocked_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        results[scenario_name] = {
            "shock": float(shock),
            "total_return": float(total_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }

    return results


def risk_contribution(
    returns_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate risk contribution of each asset to portfolio variance.

    Args:
        returns_df: DataFrame with returns for each asset (columns = assets)
        weights: Portfolio weights (default: equal weight)

    Returns:
        Series with risk contribution for each asset
    """
    if weights is None:
        weights = pd.Series(1 / len(returns_df.columns), index=returns_df.columns)

    # Covariance matrix
    cov_matrix = returns_df.cov()

    # Portfolio variance
    portfolio_var = weights.T @ cov_matrix @ weights

    # Marginal contribution to risk (MCR)
    mcr = cov_matrix @ weights / np.sqrt(portfolio_var)

    # Risk contribution = weight * MCR
    risk_contrib = weights * mcr

    # Normalize to percentages
    risk_contrib_pct = risk_contrib / risk_contrib.sum()

    return risk_contrib_pct


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: Optional[int] = None,
) -> float:
    """Calculate beta of an asset relative to market.

    Args:
        asset_returns: Series of asset returns
        market_returns: Series of market/benchmark returns
        window: Rolling window size (None for full period)

    Returns:
        Beta value
    """
    # Align the two series
    aligned = pd.DataFrame({
        "asset": asset_returns,
        "market": market_returns,
    }).dropna()

    if window:
        # Rolling beta
        covariance = aligned["asset"].rolling(window).cov(aligned["market"])
        market_var = aligned["market"].rolling(window).var()
        beta = covariance / market_var
        return float(beta.iloc[-1])  # Return latest beta
    else:
        # Full period beta
        covariance = aligned["asset"].cov(aligned["market"])
        market_var = aligned["market"].var()
        beta = covariance / market_var
        return float(beta)


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualized: bool = True,
) -> float:
    """Calculate tracking error relative to benchmark.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        annualized: Whether to annualize the tracking error

    Returns:
        Tracking error value
    """
    # Align series
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    # Tracking difference
    tracking_diff = aligned["portfolio"] - aligned["benchmark"]

    # Tracking error = std dev of tracking difference
    te = tracking_diff.std()

    if annualized:
        te *= np.sqrt(252)

    return float(te)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate information ratio (excess return / tracking error).

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns

    Returns:
        Information ratio
    """
    # Align series
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    # Excess returns
    excess = aligned["portfolio"] - aligned["benchmark"]

    # Information ratio
    mean_excess = excess.mean() * 252  # Annualized
    te = tracking_error(portfolio_returns, benchmark_returns, annualized=True)

    ir = mean_excess / te if te > 0 else 0

    return float(ir)
