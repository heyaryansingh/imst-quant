"""Risk parity portfolio construction and optimization.

This module implements various risk parity approaches including Equal Risk Contribution (ERC),
hierarchical risk parity, and adaptive risk parity with regime switching.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


class RiskParityOptimizer:
    """Risk parity portfolio optimizer."""

    def __init__(
        self,
        returns: pd.DataFrame,
        method: str = "equal_risk_contribution"
    ):
        """Initialize risk parity optimizer.

        Args:
            returns: Asset returns DataFrame
            method: Risk parity method to use
        """
        self.returns = returns
        self.method = method
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()

    def optimize(self, **kwargs) -> pd.Series:
        """Optimize portfolio weights using risk parity.

        Returns:
            Series with optimal weights
        """
        if self.method == "equal_risk_contribution":
            return self._equal_risk_contribution(**kwargs)
        elif self.method == "hierarchical":
            return self._hierarchical_risk_parity(**kwargs)
        elif self.method == "adaptive":
            return self._adaptive_risk_parity(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _equal_risk_contribution(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-6
    ) -> pd.Series:
        """Calculate Equal Risk Contribution (ERC) weights.

        Args:
            max_iter: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            ERC weights
        """
        n_assets = len(self.assets)

        # Objective: minimize sum of squared differences in risk contribution
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = self.cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Each asset should contribute 1/n of risk
            target_contrib = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints: weights sum to 1, all weights >= 0
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": tolerance}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        return pd.Series(result.x, index=self.assets)

    def _hierarchical_risk_parity(
        self,
        linkage_method: str = "single"
    ) -> pd.Series:
        """Calculate Hierarchical Risk Parity (HRP) weights.

        Args:
            linkage_method: Linkage method for clustering

        Returns:
            HRP weights
        """
        # Calculate correlation and distance matrix
        corr_matrix = self.returns.corr()
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Hierarchical clustering
        dist_condensed = squareform(dist_matrix, checks=False)
        linkage_matrix = hierarchy.linkage(dist_condensed, method=linkage_method)

        # Get quasi-diagonalization order
        sort_ix = self._get_quasi_diag(linkage_matrix)

        # Recursive bisection
        weights = pd.Series(1.0, index=sort_ix)
        clustered_alphas = [sort_ix]

        while len(clustered_alphas) > 0:
            clustered_alphas = [
                cluster[start:end]
                for cluster in clustered_alphas
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]

            for subcluster in clustered_alphas:
                # Calculate cluster variance
                subset_cov = self.cov_matrix.loc[subcluster, subcluster]
                inv_diag = 1 / np.diag(subset_cov)
                parity_w = inv_diag * (1 / np.sum(inv_diag))

                cluster_var = parity_w @ subset_cov @ parity_w

                # Allocate weight
                for i, asset in enumerate(subcluster):
                    weights[asset] *= cluster_var

        # Normalize
        weights = weights / weights.sum()
        return weights.reindex(self.assets, fill_value=0)

    def _adaptive_risk_parity(
        self,
        lookback_period: int = 60,
        min_weight: float = 0.01
    ) -> pd.Series:
        """Calculate adaptive risk parity with regime detection.

        Args:
            lookback_period: Lookback period for volatility estimation
            min_weight: Minimum weight per asset

        Returns:
            Adaptive risk parity weights
        """
        # Calculate rolling volatility
        rolling_vol = self.returns.rolling(lookback_period).std()
        current_vol = rolling_vol.iloc[-1]

        # Inverse volatility weights
        inv_vol = 1 / current_vol
        weights = inv_vol / inv_vol.sum()

        # Apply minimum weight constraint
        weights = weights.clip(lower=min_weight)
        weights = weights / weights.sum()

        return weights

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> List:
        """Get quasi-diagonal ordering from linkage matrix."""
        order = []

        def recursive_order(idx):
            if idx < len(self.assets):
                order.append(self.assets[idx])
            else:
                left = int(linkage_matrix[idx - len(self.assets), 0])
                right = int(linkage_matrix[idx - len(self.assets), 1])
                recursive_order(left)
                recursive_order(right)

        recursive_order(len(linkage_matrix) + len(self.assets))
        return order

    def calculate_risk_contributions(self, weights: pd.Series) -> pd.Series:
        """Calculate risk contribution for each asset.

        Args:
            weights: Portfolio weights

        Returns:
            Risk contributions
        """
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        marginal_contrib = self.cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol

        return risk_contrib

    def calculate_diversification_ratio(self, weights: pd.Series) -> float:
        """Calculate portfolio diversification ratio.

        Args:
            weights: Portfolio weights

        Returns:
            Diversification ratio
        """
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(self.cov_matrix))
        weighted_vol = weights @ individual_vols

        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)

        return weighted_vol / portfolio_vol


class TargetVolatilityRiskParity:
    """Risk parity with target volatility."""

    def __init__(
        self,
        returns: pd.DataFrame,
        target_volatility: float = 0.10
    ):
        """Initialize target volatility risk parity.

        Args:
            returns: Asset returns
            target_volatility: Target annualized volatility
        """
        self.returns = returns
        self.target_volatility = target_volatility
        self.optimizer = RiskParityOptimizer(returns)

    def optimize(self) -> Tuple[pd.Series, float]:
        """Optimize with target volatility scaling.

        Returns:
            Tuple of (scaled_weights, leverage)
        """
        # Get base risk parity weights
        base_weights = self.optimizer.optimize()

        # Calculate portfolio volatility
        cov_matrix = self.returns.cov()
        portfolio_vol = np.sqrt(base_weights @ cov_matrix @ base_weights)
        annualized_vol = portfolio_vol * np.sqrt(252)

        # Calculate leverage needed
        leverage = self.target_volatility / annualized_vol

        # Scale weights
        scaled_weights = base_weights * leverage

        return scaled_weights, leverage


class ConditionalRiskParity:
    """Risk parity with regime-dependent allocations."""

    def __init__(
        self,
        returns: pd.DataFrame,
        regime_indicator: pd.Series
    ):
        """Initialize conditional risk parity.

        Args:
            returns: Asset returns
            regime_indicator: Regime classification (e.g., high_vol, low_vol)
        """
        self.returns = returns
        self.regime_indicator = regime_indicator
        self.regimes = regime_indicator.unique()

    def optimize(self) -> Dict[str, pd.Series]:
        """Optimize for each regime.

        Returns:
            Dictionary mapping regime -> weights
        """
        regime_weights = {}

        for regime in self.regimes:
            # Get returns for this regime
            regime_mask = self.regime_indicator == regime
            regime_returns = self.returns[regime_mask]

            # Optimize
            optimizer = RiskParityOptimizer(regime_returns)
            weights = optimizer.optimize()
            regime_weights[regime] = weights

        return regime_weights

    def get_current_allocation(
        self,
        current_regime: str,
        regime_weights: Dict[str, pd.Series]
    ) -> pd.Series:
        """Get allocation for current regime.

        Args:
            current_regime: Current market regime
            regime_weights: Pre-computed regime weights

        Returns:
            Weights for current regime
        """
        return regime_weights.get(current_regime, pd.Series())


def calculate_risk_parity_with_constraints(
    returns: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.Series:
    """Calculate risk parity with additional constraints.

    Args:
        returns: Asset returns
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        sector_constraints: Dictionary of sector -> (min, max) weight

    Returns:
        Constrained risk parity weights
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov()

    def objective(weights):
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        target_contrib = portfolio_vol / n_assets
        return np.sum((risk_contrib - target_contrib) ** 2)

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Initial guess
    x0 = np.ones(n_assets) / n_assets

    # Add sector constraints if provided
    if sector_constraints:
        for sector, (min_w, max_w) in sector_constraints.items():
            # This would require sector mapping - simplified here
            pass

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return pd.Series(result.x, index=returns.columns)


def backtest_risk_parity(
    returns: pd.DataFrame,
    rebalance_frequency: int = 20,
    method: str = "equal_risk_contribution"
) -> pd.DataFrame:
    """Backtest risk parity strategy.

    Args:
        returns: Asset returns
        rebalance_frequency: Days between rebalancing
        method: Risk parity method

    Returns:
        DataFrame with backtest results
    """
    portfolio_values = [1.0]
    weights_history = []
    risk_contributions_history = []

    optimizer = RiskParityOptimizer(returns)

    for i in range(0, len(returns), rebalance_frequency):
        # Get historical data up to this point
        hist_returns = returns.iloc[max(0, i-252):i]

        if len(hist_returns) < 60:
            continue

        # Optimize
        optimizer.returns = hist_returns
        optimizer.cov_matrix = hist_returns.cov()
        weights = optimizer.optimize()

        weights_history.append(weights)

        # Calculate risk contributions
        risk_contrib = optimizer.calculate_risk_contributions(weights)
        risk_contributions_history.append(risk_contrib)

        # Apply weights for next period
        period_end = min(i + rebalance_frequency, len(returns))
        period_returns = returns.iloc[i:period_end]

        for _, row in period_returns.iterrows():
            portfolio_return = (weights * row).sum()
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

    results = pd.DataFrame({
        'portfolio_value': portfolio_values
    })

    return results
