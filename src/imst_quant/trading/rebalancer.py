"""Portfolio rebalancing optimizer with risk constraints.

Provides intelligent portfolio rebalancing strategies with risk management:
- Mean-variance optimization
- Risk parity
- Equal weight with drift threshold
- Minimum variance
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger()


class PortfolioRebalancer:
    """Optimize and rebalance portfolio weights based on various strategies."""

    def __init__(
        self,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        """Initialize the portfolio rebalancer.

        Args:
            rebalance_threshold: Minimum drift from target to trigger rebalance (e.g., 0.05 = 5%)
            transaction_cost: Transaction cost as decimal (e.g., 0.001 = 0.1%)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        """
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
        self.min_weight = min_weight
        self.max_weight = max_weight

        logger.info("Portfolio rebalancer initialized",
                   threshold=rebalance_threshold,
                   transaction_cost=transaction_cost)

    def mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
    ) -> pd.Series:
        """Optimize portfolio using mean-variance optimization.

        Args:
            returns: DataFrame with asset returns (columns = assets)
            target_return: Target expected return (None = maximize Sharpe)
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Series with optimal weights for each asset
        """
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        n_assets = len(returns.columns)

        # Objective: minimize variance - risk_aversion * expected_return
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return portfolio_variance - risk_aversion * portfolio_return

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, mean_returns) - target_return,
            })

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess (equal weight)
        init_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("Optimization failed, using equal weights",
                          message=result.message)
            weights = init_weights
        else:
            weights = result.x

        logger.info("Mean-variance optimization complete",
                   target_return=target_return,
                   risk_aversion=risk_aversion,
                   max_weight=weights.max(),
                   min_weight=weights.min())

        return pd.Series(weights, index=returns.columns)

    def risk_parity(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Calculate risk parity weights (equal risk contribution).

        Args:
            returns: DataFrame with asset returns

        Returns:
            Series with risk parity weights
        """
        cov_matrix = returns.cov() * 252  # Annualized
        n_assets = len(returns.columns)

        # Objective: minimize sum of squared differences in risk contribution
        def objective(weights):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib
            target_risk = portfolio_var / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        init_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            weights = init_weights

        logger.info("Risk parity optimization complete",
                   weights_std=weights.std())

        return pd.Series(weights, index=returns.columns)

    def minimum_variance(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Calculate minimum variance portfolio weights.

        Args:
            returns: DataFrame with asset returns

        Returns:
            Series with minimum variance weights
        """
        cov_matrix = returns.cov() * 252  # Annualized
        n_assets = len(returns.columns)

        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        init_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Minimum variance optimization failed")
            weights = init_weights

        return pd.Series(weights, index=returns.columns)

    def check_rebalance_needed(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> Tuple[bool, pd.Series]:
        """Check if rebalancing is needed based on drift threshold.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Tuple of (rebalance_needed, weight_drift)
        """
        # Calculate drift
        drift = (current_weights - target_weights).abs()

        # Check if any asset exceeds threshold
        max_drift = drift.max()
        rebalance_needed = max_drift > self.rebalance_threshold

        logger.info("Rebalance check",
                   max_drift=max_drift,
                   threshold=self.rebalance_threshold,
                   rebalance_needed=rebalance_needed)

        return rebalance_needed, drift

    def calculate_rebalance_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """Calculate trades needed to rebalance to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with trade details (asset, current_value, target_value, trade_value)
        """
        # Calculate values
        current_values = current_weights * portfolio_value
        target_values = target_weights * portfolio_value
        trade_values = target_values - current_values

        # Calculate transaction costs
        total_trade_value = trade_values.abs().sum()
        estimated_cost = total_trade_value * self.transaction_cost

        trades = pd.DataFrame({
            "asset": current_weights.index,
            "current_weight": current_weights.values,
            "target_weight": target_weights.values,
            "current_value": current_values.values,
            "target_value": target_values.values,
            "trade_value": trade_values.values,
            "trade_pct": (trade_values / current_values).values,
        })

        logger.info("Rebalance trades calculated",
                   total_trades=len(trades),
                   total_trade_value=total_trade_value,
                   estimated_cost=estimated_cost,
                   cost_pct=estimated_cost / portfolio_value)

        return trades

    def equal_weight(
        self,
        assets: List[str],
    ) -> pd.Series:
        """Generate equal weight portfolio.

        Args:
            assets: List of asset names

        Returns:
            Series with equal weights
        """
        n = len(assets)
        weights = [1 / n] * n
        return pd.Series(weights, index=assets)

    def market_cap_weight(
        self,
        market_caps: pd.Series,
    ) -> pd.Series:
        """Generate market cap weighted portfolio.

        Args:
            market_caps: Series with market cap for each asset

        Returns:
            Series with market cap weights
        """
        total_cap = market_caps.sum()
        weights = market_caps / total_cap
        return weights

    def should_rebalance_with_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        expected_alpha: float = 0.0,
    ) -> Tuple[bool, Dict[str, float]]:
        """Determine if rebalancing is worthwhile considering transaction costs.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            expected_alpha: Expected excess return from rebalancing (annualized)

        Returns:
            Tuple of (should_rebalance, analysis_dict)
        """
        # Calculate trades
        trades_df = self.calculate_rebalance_trades(
            current_weights, target_weights, portfolio_value
        )

        # Total cost
        total_trade_value = trades_df["trade_value"].abs().sum()
        total_cost = total_trade_value * self.transaction_cost

        # Expected benefit
        expected_benefit = portfolio_value * expected_alpha

        # Net benefit
        net_benefit = expected_benefit - total_cost

        should_rebalance = net_benefit > 0

        analysis = {
            "total_cost": total_cost,
            "cost_pct": total_cost / portfolio_value,
            "expected_benefit": expected_benefit,
            "net_benefit": net_benefit,
            "should_rebalance": should_rebalance,
        }

        logger.info("Cost-benefit analysis",
                   **analysis)

        return should_rebalance, analysis
