"""
Portfolio risk decomposition utilities.

Provides methods to decompose portfolio risk into contributions from individual
assets, factors, and systematic vs idiosyncratic components.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cholesky


def calculate_marginal_risk(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate marginal risk contribution for each asset.

    Marginal risk is the partial derivative of portfolio volatility
    with respect to each asset weight.

    Args:
        weights: Asset weights (must sum to 1)
        cov_matrix: Covariance matrix of asset returns

    Returns:
        Array of marginal risk contributions
    """
    portfolio_var = weights @ cov_matrix @ weights
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol == 0:
        return np.zeros_like(weights)

    marginal_risk = (cov_matrix @ weights) / portfolio_vol
    return marginal_risk


def calculate_component_risk(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate risk contribution and percentage risk contribution for each asset.

    Component risk shows how much each asset contributes to total portfolio risk.

    Args:
        weights: Asset weights (must sum to 1)
        cov_matrix: Covariance matrix of asset returns

    Returns:
        Tuple of (absolute_contributions, percentage_contributions)
    """
    marginal_risk = calculate_marginal_risk(weights, cov_matrix)
    component_contributions = weights * marginal_risk

    total_risk = np.sqrt(weights @ cov_matrix @ weights)
    if total_risk == 0:
        pct_contributions = np.zeros_like(weights)
    else:
        pct_contributions = component_contributions / total_risk

    return component_contributions, pct_contributions


def decompose_systematic_idiosyncratic(
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Decompose asset returns into systematic (market) and idiosyncratic components.

    Uses simple market model: R_i = alpha_i + beta_i * R_m + epsilon_i

    Args:
        returns: DataFrame of asset returns (assets as columns)
        market_returns: Series of market returns

    Returns:
        Dictionary with:
            - 'betas': Beta coefficients for each asset
            - 'systematic_var': Systematic variance for each asset
            - 'idiosyncratic_var': Idiosyncratic variance for each asset
            - 'total_var': Total variance for each asset
            - 'r_squared': R-squared values
    """
    assets = returns.columns
    results = {
        'asset': [],
        'beta': [],
        'systematic_var': [],
        'idiosyncratic_var': [],
        'total_var': [],
        'r_squared': [],
    }

    market_var = market_returns.var()

    for asset in assets:
        asset_returns = returns[asset].dropna()
        aligned_market = market_returns.loc[asset_returns.index]

        # Calculate beta
        cov = asset_returns.cov(aligned_market)
        beta = cov / market_var if market_var > 0 else 0

        # Decompose variance
        total_var = asset_returns.var()
        systematic_var = (beta ** 2) * market_var
        idiosyncratic_var = total_var - systematic_var

        # Calculate R-squared
        if total_var > 0:
            r_squared = systematic_var / total_var
        else:
            r_squared = 0

        results['asset'].append(asset)
        results['beta'].append(beta)
        results['systematic_var'].append(systematic_var)
        results['idiosyncratic_var'].append(idiosyncratic_var)
        results['total_var'].append(total_var)
        results['r_squared'].append(r_squared)

    return pd.DataFrame(results)


def calculate_diversification_ratio(
    weights: np.ndarray,
    volatilities: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Calculate portfolio diversification ratio.

    Diversification ratio = weighted average volatility / portfolio volatility
    A higher ratio indicates better diversification.

    Args:
        weights: Asset weights
        volatilities: Individual asset volatilities
        cov_matrix: Covariance matrix

    Returns:
        Diversification ratio
    """
    weighted_vol = weights @ volatilities
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

    if portfolio_vol == 0:
        return 0.0

    return weighted_vol / portfolio_vol


def calculate_concentration_metrics(weights: np.ndarray) -> Dict[str, float]:
    """
    Calculate portfolio concentration metrics.

    Args:
        weights: Asset weights (must sum to 1)

    Returns:
        Dictionary with concentration metrics:
            - 'herfindahl_index': Sum of squared weights
            - 'effective_n': Effective number of assets
            - 'max_weight': Maximum weight
            - 'top_3_weight': Sum of top 3 weights
    """
    weights = np.abs(weights)  # Handle short positions
    weights = weights / weights.sum()  # Normalize

    herfindahl = (weights ** 2).sum()
    effective_n = 1 / herfindahl if herfindahl > 0 else 0

    sorted_weights = np.sort(weights)[::-1]
    max_weight = sorted_weights[0] if len(sorted_weights) > 0 else 0
    top_3_weight = sorted_weights[:3].sum() if len(sorted_weights) >= 3 else sorted_weights.sum()

    return {
        'herfindahl_index': herfindahl,
        'effective_n': effective_n,
        'max_weight': max_weight,
        'top_3_weight': top_3_weight,
    }


def factor_risk_decomposition(
    weights: np.ndarray,
    factor_exposures: np.ndarray,
    factor_cov: np.ndarray,
    specific_var: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Decompose portfolio risk using a factor model.

    Assumes portfolio variance = weights' @ (B @ F @ B' + D) @ weights
    where B = factor exposures, F = factor covariance, D = diagonal specific variance

    Args:
        weights: Portfolio weights (N x 1)
        factor_exposures: Asset factor exposures (N x K)
        factor_cov: Factor covariance matrix (K x K)
        specific_var: Asset-specific variances (N x 1)

    Returns:
        Dictionary with:
            - 'total_risk': Total portfolio risk
            - 'factor_risk': Risk from common factors
            - 'specific_risk': Risk from asset-specific components
            - 'factor_contributions': Contribution from each factor
    """
    # Portfolio factor exposures
    portfolio_exposures = weights @ factor_exposures  # 1 x K

    # Factor risk
    factor_var = portfolio_exposures @ factor_cov @ portfolio_exposures.T
    factor_risk = np.sqrt(factor_var)

    # Specific risk
    specific_var_matrix = np.diag(specific_var)
    specific_var_portfolio = weights @ specific_var_matrix @ weights
    specific_risk = np.sqrt(specific_var_portfolio)

    # Total risk
    total_var = factor_var + specific_var_portfolio
    total_risk = np.sqrt(total_var)

    # Individual factor contributions
    factor_marginal_var = factor_cov @ portfolio_exposures.T  # K x 1
    factor_contributions = portfolio_exposures.T * factor_marginal_var  # K x 1

    return {
        'total_risk': float(total_risk),
        'factor_risk': float(factor_risk),
        'specific_risk': float(specific_risk),
        'factor_contributions': factor_contributions.flatten(),
    }


def calculate_risk_parity_weights(
    cov_matrix: np.ndarray,
    initial_weights: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Calculate risk parity portfolio weights.

    Risk parity ensures each asset contributes equally to portfolio risk.
    Uses iterative algorithm to find weights where all risk contributions are equal.

    Args:
        cov_matrix: Covariance matrix of asset returns
        initial_weights: Starting weights (default: equal weights)
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        Risk parity weights
    """
    n_assets = cov_matrix.shape[0]

    if initial_weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = initial_weights.copy()

    for _ in range(max_iterations):
        # Calculate current risk contributions
        _, pct_contributions = calculate_component_risk(weights, cov_matrix)

        # Target is equal contribution (1/N for each asset)
        target = 1.0 / n_assets

        # Adjust weights to move towards equal contributions
        # Simple iterative scaling
        adjustment = target / (pct_contributions + 1e-10)
        new_weights = weights * adjustment

        # Normalize
        new_weights = new_weights / new_weights.sum()

        # Check convergence
        if np.max(np.abs(new_weights - weights)) < tolerance:
            weights = new_weights
            break

        weights = new_weights

    return weights


def tail_risk_decomposition(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Decompose tail risk measures (VaR, CVaR) into asset contributions.

    Uses historical simulation to estimate tail risk contributions.

    Args:
        returns: DataFrame of asset returns
        weights: Portfolio weights
        confidence_level: Confidence level for VaR/CVaR (default 0.95)

    Returns:
        Dictionary with tail risk metrics and decomposition
    """
    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)

    # Calculate VaR
    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(portfolio_returns, var_percentile)

    # Calculate CVaR (expected shortfall)
    tail_returns = portfolio_returns[portfolio_returns <= var]
    cvar = tail_returns.mean() if len(tail_returns) > 0 else var

    # Component CVaR (contribution to tail risk)
    # For each asset, calculate its average return during tail events
    tail_mask = portfolio_returns <= var
    component_cvar = {}

    for col in returns.columns:
        asset_tail_returns = returns.loc[tail_mask, col]
        component_cvar[col] = weights[returns.columns.get_loc(col)] * asset_tail_returns.mean()

    return {
        'var': var,
        'cvar': cvar,
        'component_cvar': component_cvar,
    }
