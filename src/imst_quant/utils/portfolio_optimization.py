"""Portfolio optimization utilities for asset allocation.

Provides implementations of classic portfolio optimization techniques
including mean-variance optimization, risk parity, and Black-Litterman.

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample returns
    >>> returns = pd.DataFrame(
    ...     np.random.randn(252, 4) * 0.02,
    ...     columns=["AAPL", "GOOGL", "MSFT", "AMZN"]
    ... )
    >>> # Run mean-variance optimization
    >>> weights = mean_variance_optimize(returns, target_return=0.10)
    >>> print(f"Optimal weights: {weights}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OptimizationObjective(Enum):
    """Optimization objective for portfolio construction."""

    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"
    RISK_PARITY = "risk_parity"


@dataclass
class PortfolioStats:
    """Statistics for an optimized portfolio.

    Attributes:
        weights: Dict mapping asset names to weights.
        expected_return: Annualized expected return.
        volatility: Annualized volatility.
        sharpe_ratio: Sharpe ratio (assumes 0 risk-free rate if not specified).
        diversification_ratio: Sum of weighted vols / portfolio vol.
        effective_n: Effective number of bets (1 / sum(w^2)).
    """

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float


@dataclass
class EfficientFrontier:
    """Efficient frontier data points.

    Attributes:
        returns: List of returns along the frontier.
        volatilities: List of volatilities along the frontier.
        sharpe_ratios: List of Sharpe ratios along the frontier.
        weights: List of weight dicts for each point.
    """

    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    weights: List[Dict[str, float]]


def estimate_expected_returns(
    returns: pd.DataFrame,
    method: str = "mean",
    annual_factor: int = 252,
) -> pd.Series:
    """Estimate expected returns for portfolio optimization.

    Args:
        returns: DataFrame of daily returns (assets as columns).
        method: Estimation method ("mean", "ewma", "capm").
        annual_factor: Trading days per year for annualization.

    Returns:
        Series of annualized expected returns per asset.
    """
    if method == "mean":
        daily_mean = returns.mean()
    elif method == "ewma":
        # Exponentially weighted mean with half-life of 60 days
        daily_mean = returns.ewm(halflife=60).mean().iloc[-1]
    else:
        raise ValueError(f"Unknown method: {method}")

    annual_returns = daily_mean * annual_factor
    return annual_returns


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "sample",
    annual_factor: int = 252,
) -> pd.DataFrame:
    """Estimate covariance matrix for portfolio optimization.

    Args:
        returns: DataFrame of daily returns (assets as columns).
        method: Estimation method ("sample", "ewma", "ledoit_wolf").
        annual_factor: Trading days per year for annualization.

    Returns:
        Annualized covariance matrix.
    """
    if method == "sample":
        cov = returns.cov()
    elif method == "ewma":
        cov = returns.ewm(halflife=60).cov().iloc[-len(returns.columns):]
        cov.index = returns.columns
    elif method == "ledoit_wolf":
        # Shrinkage estimator
        cov = _ledoit_wolf_shrinkage(returns)
    else:
        raise ValueError(f"Unknown method: {method}")

    return cov * annual_factor


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance estimator.

    Shrinks sample covariance toward a structured target (scaled identity).
    """
    n, p = returns.shape
    sample_cov = returns.cov()

    # Shrinkage target: scaled identity matrix
    mu = np.trace(sample_cov.values) / p
    target = np.eye(p) * mu

    # Estimate optimal shrinkage intensity
    X = returns.values - returns.values.mean(axis=0)
    X2 = X ** 2

    # Sum of squared correlations
    sample = X.T @ X / n

    # Shrinkage intensity estimation (simplified)
    delta = sample_cov.values - target
    delta_sq_sum = (delta ** 2).sum()

    # Estimate variance of sample covariance elements
    phi_sum = 0
    for i in range(p):
        for j in range(p):
            phi_sum += np.var(X[:, i] * X[:, j])

    kappa = (phi_sum / n - delta_sq_sum) / delta_sq_sum if delta_sq_sum > 0 else 0
    shrinkage = max(0, min(1, kappa / n))

    # Shrunk covariance
    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov.values

    return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)


def portfolio_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """Calculate portfolio expected return."""
    return float(weights @ expected_returns)


def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Calculate portfolio volatility."""
    return float(np.sqrt(weights @ cov_matrix @ weights))


def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0.0


def mean_variance_optimize(
    returns: pd.DataFrame,
    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    risk_free_rate: float = 0.0,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    allow_short: bool = False,
    max_iterations: int = 1000,
) -> PortfolioStats:
    """Perform mean-variance portfolio optimization.

    Uses sequential least squares programming (SLSQP) to find optimal weights.

    Args:
        returns: DataFrame of daily returns.
        objective: Optimization objective (MAX_SHARPE, MIN_VARIANCE, etc.).
        target_return: Target annual return (for TARGET_RETURN objective).
        target_volatility: Target annual volatility (for TARGET_RISK objective).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        weight_bounds: (min, max) bounds for individual weights.
        allow_short: If True, allows negative weights.
        max_iterations: Maximum optimization iterations.

    Returns:
        PortfolioStats with optimal weights and metrics.

    Example:
        >>> returns = pd.DataFrame(np.random.randn(252, 3) * 0.02)
        >>> result = mean_variance_optimize(returns)
        >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """
    mu = estimate_expected_returns(returns).values
    cov = estimate_covariance(returns).values
    n_assets = len(mu)
    assets = returns.columns.tolist()

    if allow_short:
        bounds = (-1.0, 1.0)
    else:
        bounds = weight_bounds

    # Initial weights (equal weight)
    w0 = np.ones(n_assets) / n_assets

    # Simple gradient descent optimization
    w = w0.copy()
    lr = 0.01

    for iteration in range(max_iterations):
        if objective == OptimizationObjective.MAX_SHARPE:
            # Gradient of negative Sharpe
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            if vol > 1e-10:
                d_ret = mu
                d_vol = (cov @ w) / vol
                d_sharpe = (d_ret * vol - (ret - risk_free_rate) * d_vol) / (vol ** 2)
                gradient = -d_sharpe
            else:
                gradient = np.zeros(n_assets)

        elif objective == OptimizationObjective.MIN_VARIANCE:
            # Gradient of variance
            gradient = 2 * cov @ w

        elif objective == OptimizationObjective.TARGET_RETURN:
            if target_return is None:
                raise ValueError("target_return required")
            # Minimize variance subject to return constraint
            # Penalty method
            ret = w @ mu
            gradient = 2 * cov @ w + 100 * (ret - target_return) * mu

        elif objective == OptimizationObjective.TARGET_RISK:
            if target_volatility is None:
                raise ValueError("target_volatility required")
            # Maximize return subject to volatility constraint
            vol = np.sqrt(w @ cov @ w)
            gradient = -mu + 100 * max(0, vol - target_volatility) * (cov @ w) / vol

        else:
            gradient = np.zeros(n_assets)

        # Update weights
        w = w - lr * gradient

        # Project onto constraints
        w = np.clip(w, bounds[0], bounds[1])
        w = w / w.sum()  # Ensure weights sum to 1

    # Calculate final stats
    exp_ret = float(portfolio_return(w, mu))
    vol = float(portfolio_volatility(w, cov))
    sharpe = float(portfolio_sharpe(w, mu, cov, risk_free_rate))

    # Diversification metrics
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w * asset_vols).sum() / vol) if vol > 0 else 1.0
    effective_n = float(1.0 / (w ** 2).sum())

    weights_dict = {asset: float(w[i]) for i, asset in enumerate(assets)}

    return PortfolioStats(
        weights=weights_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
    )


def risk_parity_optimize(
    returns: pd.DataFrame,
    risk_budget: Optional[Dict[str, float]] = None,
    max_iterations: int = 1000,
) -> PortfolioStats:
    """Optimize portfolio for equal risk contribution (risk parity).

    Each asset contributes equally to portfolio risk.
    Risk contribution = weight * marginal risk = w_i * (Cov @ w)_i / vol

    Args:
        returns: DataFrame of daily returns.
        risk_budget: Optional dict of target risk contributions per asset.
        max_iterations: Maximum optimization iterations.

    Returns:
        PortfolioStats with risk parity weights.

    Example:
        >>> returns = pd.DataFrame(np.random.randn(252, 4) * 0.02)
        >>> result = risk_parity_optimize(returns)
    """
    mu = estimate_expected_returns(returns).values
    cov = estimate_covariance(returns).values
    n_assets = len(mu)
    assets = returns.columns.tolist()

    if risk_budget is None:
        target_risk = np.ones(n_assets) / n_assets
    else:
        target_risk = np.array([risk_budget.get(a, 1.0 / n_assets) for a in assets])
        target_risk = target_risk / target_risk.sum()

    # Start with inverse volatility weights
    vols = np.sqrt(np.diag(cov))
    w = (1 / vols) / (1 / vols).sum()

    lr = 0.001

    for iteration in range(max_iterations):
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-10:
            break

        # Marginal risk contribution
        mrc = (cov @ w) / vol

        # Risk contribution
        rc = w * mrc
        rc_normalized = rc / rc.sum() if rc.sum() > 0 else target_risk

        # Gradient: difference from target risk budget
        gradient = rc_normalized - target_risk

        # Update using gradient
        w = w * np.exp(-lr * gradient * 100)
        w = w / w.sum()
        w = np.clip(w, 0.01, 0.99)

    # Final stats
    exp_ret = float(portfolio_return(w, mu))
    vol = float(portfolio_volatility(w, cov))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w * asset_vols).sum() / vol) if vol > 0 else 1.0
    effective_n = float(1.0 / (w ** 2).sum())

    weights_dict = {asset: float(w[i]) for i, asset in enumerate(assets)}

    return PortfolioStats(
        weights=weights_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
    )


def black_litterman(
    returns: pd.DataFrame,
    market_caps: Dict[str, float],
    views: List[Dict],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> PortfolioStats:
    """Black-Litterman portfolio optimization.

    Combines market equilibrium with investor views to generate
    expected returns, then optimizes portfolio weights.

    Args:
        returns: DataFrame of daily returns.
        market_caps: Dict of market capitalizations per asset.
        views: List of view dicts, each with:
            - "assets": list of asset names in the view
            - "weights": list of weights (positive for long, negative for short)
            - "return": expected return of the view
            - "confidence": confidence level (0-1)
        tau: Uncertainty scaling factor for equilibrium returns.
        risk_aversion: Market risk aversion coefficient.

    Returns:
        PortfolioStats with Black-Litterman optimized weights.

    Example:
        >>> views = [
        ...     {"assets": ["AAPL"], "weights": [1], "return": 0.15, "confidence": 0.8}
        ... ]
        >>> market_caps = {"AAPL": 3e12, "GOOGL": 2e12}
        >>> result = black_litterman(returns, market_caps, views)
    """
    cov = estimate_covariance(returns).values
    n_assets = len(returns.columns)
    assets = returns.columns.tolist()

    # Market cap weights
    total_cap = sum(market_caps.values())
    w_mkt = np.array([market_caps.get(a, total_cap / n_assets) / total_cap for a in assets])

    # Equilibrium returns (implied by market cap weights)
    pi = risk_aversion * cov @ w_mkt

    # Process views
    if len(views) == 0:
        # No views - use equilibrium
        mu_bl = pi
    else:
        k = len(views)
        P = np.zeros((k, n_assets))  # Pick matrix
        Q = np.zeros(k)  # View returns
        omega_diag = []  # View uncertainty

        for i, view in enumerate(views):
            view_assets = view["assets"]
            view_weights = view["weights"]

            for j, asset in enumerate(view_assets):
                if asset in assets:
                    asset_idx = assets.index(asset)
                    P[i, asset_idx] = view_weights[j]

            Q[i] = view["return"]

            # View uncertainty based on confidence
            confidence = view.get("confidence", 0.5)
            omega_diag.append((1 - confidence) * 0.1)

        Omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_cov = tau * cov
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(Omega + 1e-10 * np.eye(k))

        # Posterior expected returns
        M1 = inv_tau_cov + P.T @ inv_omega @ P
        M2 = inv_tau_cov @ pi + P.T @ inv_omega @ Q
        mu_bl = np.linalg.solve(M1, M2)

    # Optimize using posterior returns
    w = np.linalg.solve(risk_aversion * cov, mu_bl)
    w = np.clip(w, 0, 1)
    w = w / w.sum() if w.sum() > 0 else np.ones(n_assets) / n_assets

    # Calculate stats
    exp_ret = float(portfolio_return(w, mu_bl))
    vol = float(portfolio_volatility(w, cov))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w * asset_vols).sum() / vol) if vol > 0 else 1.0
    effective_n = float(1.0 / (w ** 2).sum())

    weights_dict = {asset: float(w[i]) for i, asset in enumerate(assets)}

    return PortfolioStats(
        weights=weights_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
    )


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.0,
) -> EfficientFrontier:
    """Calculate the efficient frontier for a set of assets.

    Generates portfolio weights along the efficient frontier from
    minimum variance to maximum return.

    Args:
        returns: DataFrame of daily returns.
        n_points: Number of points along the frontier.
        risk_free_rate: Risk-free rate for Sharpe calculations.

    Returns:
        EfficientFrontier with return/vol/weight data for each point.
    """
    mu = estimate_expected_returns(returns).values
    cov = estimate_covariance(returns).values

    # Find min and max return portfolios
    min_var = mean_variance_optimize(
        returns, OptimizationObjective.MIN_VARIANCE
    )
    min_ret = min_var.expected_return

    max_ret = mu.max()

    # Generate frontier
    target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

    frontier_returns = []
    frontier_vols = []
    frontier_sharpes = []
    frontier_weights = []

    for target in target_returns:
        try:
            result = mean_variance_optimize(
                returns,
                OptimizationObjective.TARGET_RETURN,
                target_return=target,
            )
            frontier_returns.append(result.expected_return)
            frontier_vols.append(result.volatility)
            frontier_sharpes.append(result.sharpe_ratio)
            frontier_weights.append(result.weights)
        except Exception:
            continue

    return EfficientFrontier(
        returns=frontier_returns,
        volatilities=frontier_vols,
        sharpe_ratios=frontier_sharpes,
        weights=frontier_weights,
    )


def minimum_tracking_error(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    tracking_error_budget: float = 0.02,
) -> PortfolioStats:
    """Minimize tracking error relative to a benchmark.

    Useful for index tracking or enhanced indexing strategies.

    Args:
        returns: DataFrame of asset returns.
        benchmark_returns: Series of benchmark returns.
        tracking_error_budget: Maximum acceptable tracking error.

    Returns:
        PortfolioStats optimized for minimum tracking error.
    """
    # Align data
    common_idx = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_idx]
    benchmark = benchmark_returns.loc[common_idx]

    # Active returns
    n_assets = len(returns.columns)
    assets = returns.columns.tolist()

    # Covariance of active returns
    active_returns = returns.sub(benchmark, axis=0)
    active_cov = active_returns.cov().values * 252

    # Start with equal weights
    w = np.ones(n_assets) / n_assets

    # Minimize tracking error (active variance)
    for _ in range(500):
        gradient = 2 * active_cov @ w
        w = w - 0.01 * gradient
        w = np.clip(w, 0, 1)
        w = w / w.sum()

    # Calculate stats
    mu = estimate_expected_returns(returns).values
    cov = estimate_covariance(returns).values

    exp_ret = float(portfolio_return(w, mu))
    vol = float(portfolio_volatility(w, cov))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w * asset_vols).sum() / vol) if vol > 0 else 1.0
    effective_n = float(1.0 / (w ** 2).sum())

    weights_dict = {asset: float(w[i]) for i, asset in enumerate(assets)}

    return PortfolioStats(
        weights=weights_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
    )


def hierarchical_risk_parity(
    returns: pd.DataFrame,
) -> PortfolioStats:
    """Hierarchical Risk Parity (HRP) portfolio optimization.

    Uses hierarchical clustering to create a diversified portfolio
    without relying on expected returns (more robust to estimation error).

    Based on Lopez de Prado's HRP algorithm.

    Args:
        returns: DataFrame of daily returns.

    Returns:
        PortfolioStats with HRP weights.
    """
    cov = estimate_covariance(returns).values
    corr = returns.corr().values
    n = len(returns.columns)
    assets = returns.columns.tolist()

    # Step 1: Tree clustering using correlation distance
    dist = np.sqrt(0.5 * (1 - corr))

    # Simple hierarchical clustering (single linkage)
    clusters = {i: [i] for i in range(n)}
    cluster_order = list(range(n))

    while len(clusters) > 1:
        # Find closest pair
        min_dist = float("inf")
        merge_i, merge_j = 0, 1

        cluster_keys = list(clusters.keys())
        for i in range(len(cluster_keys)):
            for j in range(i + 1, len(cluster_keys)):
                ci, cj = cluster_keys[i], cluster_keys[j]
                # Average linkage
                d = np.mean([
                    dist[a, b]
                    for a in clusters[ci]
                    for b in clusters[cj]
                ])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = ci, cj

        # Merge clusters
        new_key = min(merge_i, merge_j)
        clusters[new_key] = clusters[merge_i] + clusters[merge_j]
        if merge_i != new_key:
            del clusters[merge_i]
        if merge_j != new_key:
            del clusters[merge_j]

    # Get ordered indices from clustering
    sorted_idx = clusters[list(clusters.keys())[0]]

    # Step 2: Recursive bisection
    w = np.ones(n)

    def get_cluster_var(indices):
        sub_cov = cov[np.ix_(indices, indices)]
        # Inverse variance weights within cluster
        inv_var = 1 / np.diag(sub_cov)
        weights = inv_var / inv_var.sum()
        return float(weights @ sub_cov @ weights)

    def bisect(indices, weights):
        if len(indices) == 1:
            return

        # Split in half
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]

        # Cluster variances
        var_left = get_cluster_var(left_idx)
        var_right = get_cluster_var(right_idx)

        # Allocate inversely to variance
        alpha = 1 - var_left / (var_left + var_right)

        for i in left_idx:
            weights[i] *= alpha
        for i in right_idx:
            weights[i] *= (1 - alpha)

        bisect(left_idx, weights)
        bisect(right_idx, weights)

    bisect(sorted_idx, w)
    w = w / w.sum()

    # Calculate stats
    mu = estimate_expected_returns(returns).values
    exp_ret = float(portfolio_return(w, mu))
    vol = float(portfolio_volatility(w, cov))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w * asset_vols).sum() / vol) if vol > 0 else 1.0
    effective_n = float(1.0 / (w ** 2).sum())

    weights_dict = {asset: float(w[i]) for i, asset in enumerate(assets)}

    return PortfolioStats(
        weights=weights_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        diversification_ratio=div_ratio,
        effective_n=effective_n,
    )
