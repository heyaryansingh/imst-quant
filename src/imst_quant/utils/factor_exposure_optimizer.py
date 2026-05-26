"""Factor Exposure Optimizer - Optimize portfolio factor exposures and tilts.

This module provides factor-based portfolio optimization including:
- Factor exposure calculation and attribution
- Factor tilt optimization
- Factor risk budgeting
- Pure factor portfolios construction
- Factor timing strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class FactorExposures:
    """Portfolio factor exposures."""

    exposures: Dict[str, float]
    active_exposures: Dict[str, float]
    factor_contributions: Dict[str, float]
    total_risk: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'exposures': self.exposures,
            'active_exposures': self.active_exposures,
            'factor_contributions': self.factor_contributions,
            'total_risk': self.total_risk
        }


class FactorExposureOptimizer:
    """Optimizes portfolio factor exposures."""

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        asset_factor_loadings: pd.DataFrame,
        benchmark_weights: Optional[pd.Series] = None
    ):
        """Initialize factor exposure optimizer.

        Args:
            factor_returns: DataFrame with factor returns (columns=factors, index=dates)
            asset_factor_loadings: DataFrame with factor loadings (rows=assets, columns=factors)
            benchmark_weights: Optional benchmark weights for active management
        """
        self.factor_returns = factor_returns
        self.factor_loadings = asset_factor_loadings
        self.benchmark_weights = benchmark_weights

        # Calculate factor covariance
        self.factor_cov = factor_returns.cov()

        # Asset names and factors
        self.assets = list(asset_factor_loadings.index)
        self.factors = list(asset_factor_loadings.columns)
        self.num_assets = len(self.assets)
        self.num_factors = len(self.factors)

    def calculate_exposures(
        self,
        portfolio_weights: pd.Series
    ) -> FactorExposures:
        """Calculate factor exposures for a portfolio.

        Args:
            portfolio_weights: Portfolio weights (must sum to 1)

        Returns:
            FactorExposures with detailed breakdown
        """
        # Align weights with factor loadings
        weights = portfolio_weights.reindex(self.assets, fill_value=0.0)

        # Calculate exposures
        exposures = {}
        for factor in self.factors:
            exposure = np.sum(
                weights.values * self.factor_loadings[factor].values
            )
            exposures[factor] = exposure

        # Active exposures (vs benchmark)
        active_exposures = {}
        if self.benchmark_weights is not None:
            bench_weights = self.benchmark_weights.reindex(self.assets, fill_value=0.0)
            for factor in self.factors:
                bench_exposure = np.sum(
                    bench_weights.values * self.factor_loadings[factor].values
                )
                active_exposures[factor] = exposures[factor] - bench_exposure
        else:
            active_exposures = exposures.copy()

        # Factor contributions to risk
        exp_vector = np.array([exposures[f] for f in self.factors])
        factor_var = exp_vector @ self.factor_cov.values @ exp_vector
        total_risk = np.sqrt(factor_var)

        # Individual factor contributions
        factor_contributions = {}
        if total_risk > 0:
            for i, factor in enumerate(self.factors):
                contrib = (exp_vector[i] * (self.factor_cov.values[i] @ exp_vector)) / factor_var
                factor_contributions[factor] = contrib
        else:
            factor_contributions = {f: 0.0 for f in self.factors}

        return FactorExposures(
            exposures=exposures,
            active_exposures=active_exposures,
            factor_contributions=factor_contributions,
            total_risk=total_risk
        )

    def optimize_factor_tilt(
        self,
        target_tilts: Dict[str, float],
        max_tracking_error: float = 0.05,
        position_limits: Tuple[float, float] = (0.0, 0.1),
        turnover_constraint: Optional[float] = None,
        current_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """Optimize portfolio to achieve target factor tilts.

        Args:
            target_tilts: Dictionary of target active factor exposures
            max_tracking_error: Maximum tracking error vs benchmark
            position_limits: (min, max) position size limits
            turnover_constraint: Optional maximum turnover
            current_weights: Current portfolio weights for turnover calculation

        Returns:
            Optimized portfolio weights
        """
        if self.benchmark_weights is None:
            raise ValueError("Benchmark weights required for factor tilt optimization")

        # Objective: minimize squared deviations from target tilts
        def objective(weights):
            port_weights = pd.Series(weights, index=self.assets)
            exposures = self.calculate_exposures(port_weights)

            deviations = 0.0
            for factor, target in target_tilts.items():
                actual = exposures.active_exposures.get(factor, 0.0)
                deviations += (actual - target) ** 2

            return deviations

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
        ]

        # Tracking error constraint
        def tracking_error_constraint(weights):
            active_weights = weights - self.benchmark_weights.values
            # Simplified TE calculation
            te = np.std(active_weights) * np.sqrt(252)
            return max_tracking_error - te

        constraints.append({'type': 'ineq', 'fun': tracking_error_constraint})

        # Turnover constraint
        if turnover_constraint and current_weights is not None:
            def turnover_limit(weights):
                turnover = np.sum(np.abs(weights - current_weights.values))
                return turnover_constraint - turnover

            constraints.append({'type': 'ineq', 'fun': turnover_limit})

        # Bounds
        bounds = [position_limits] * self.num_assets

        # Initial guess (benchmark weights)
        x0 = self.benchmark_weights.values

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            print(f"Optimization warning: {result.message}")

        return pd.Series(result.x, index=self.assets)

    def construct_pure_factor_portfolio(
        self,
        factor: str,
        target_exposure: float = 1.0,
        position_limits: Tuple[float, float] = (-0.2, 0.2)
    ) -> pd.Series:
        """Construct pure factor portfolio (long-short).

        Args:
            factor: Target factor name
            target_exposure: Target exposure to the factor
            position_limits: Position size limits

        Returns:
            Long-short portfolio weights
        """
        if factor not in self.factors:
            raise ValueError(f"Factor {factor} not found")

        # Objective: minimize exposure to other factors + transaction costs
        def objective(weights):
            port_weights = pd.Series(weights, index=self.assets)
            exposures = self.calculate_exposures(port_weights)

            # Minimize other factor exposures
            other_exposures = sum(
                exposures.exposures[f] ** 2
                for f in self.factors if f != factor
            )

            # Add weight penalty for transaction costs
            weight_penalty = 0.001 * np.sum(np.abs(weights))

            return other_exposures + weight_penalty

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w)},  # Dollar neutral
            {
                'type': 'eq',
                'fun': lambda w: np.sum(
                    w * self.factor_loadings[factor].values
                ) - target_exposure
            }  # Target factor exposure
        ]

        # Bounds
        bounds = [position_limits] * self.num_assets

        # Initial guess
        x0 = np.zeros(self.num_assets)

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return pd.Series(result.x, index=self.assets)

    def optimize_factor_risk_budget(
        self,
        factor_risk_budgets: Dict[str, float],
        total_volatility_target: float = 0.15,
        position_limits: Tuple[float, float] = (0.0, 0.15)
    ) -> pd.Series:
        """Optimize portfolio to achieve factor risk budgets.

        Args:
            factor_risk_budgets: Target risk contribution by factor (must sum to 1)
            total_volatility_target: Target portfolio volatility
            position_limits: Position size limits

        Returns:
            Optimized portfolio weights
        """
        # Validate risk budgets
        if not np.isclose(sum(factor_risk_budgets.values()), 1.0):
            raise ValueError("Factor risk budgets must sum to 1")

        # Objective: minimize squared deviations from target risk budgets
        def objective(weights):
            port_weights = pd.Series(weights, index=self.assets)
            exposures = self.calculate_exposures(port_weights)

            if exposures.total_risk == 0:
                return 1e10

            deviations = 0.0
            for factor, target_budget in factor_risk_budgets.items():
                actual_contrib = exposures.factor_contributions.get(factor, 0.0)
                deviations += (actual_contrib - target_budget) ** 2

            # Penalty for deviation from volatility target
            vol_penalty = 100 * (exposures.total_risk - total_volatility_target) ** 2

            return deviations + vol_penalty

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
        ]

        # Bounds
        bounds = [position_limits] * self.num_assets

        # Initial guess (equal weight)
        x0 = np.ones(self.num_assets) / self.num_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return pd.Series(result.x, index=self.assets)

    def analyze_factor_timing_signals(
        self,
        lookback_periods: List[int] = [20, 60, 120]
    ) -> pd.DataFrame:
        """Analyze factor momentum for timing strategies.

        Args:
            lookback_periods: List of lookback periods for momentum

        Returns:
            DataFrame with factor timing signals
        """
        signals = []

        for factor in self.factors:
            factor_rets = self.factor_returns[factor]

            signal_data = {'factor': factor}

            for period in lookback_periods:
                if len(factor_rets) < period:
                    continue

                # Momentum
                momentum = factor_rets.tail(period).mean()
                signal_data[f'momentum_{period}d'] = momentum

                # Volatility
                vol = factor_rets.tail(period).std() * np.sqrt(252)
                signal_data[f'vol_{period}d'] = vol

                # Sharpe
                sharpe = (momentum * 252) / (vol + 1e-6)
                signal_data[f'sharpe_{period}d'] = sharpe

            # Recent performance
            signal_data['ret_1m'] = factor_rets.tail(21).sum()
            signal_data['ret_3m'] = factor_rets.tail(63).sum()
            signal_data['ret_6m'] = factor_rets.tail(126).sum()

            signals.append(signal_data)

        df = pd.DataFrame(signals)

        # Composite timing score
        if 'sharpe_60d' in df.columns:
            df['timing_score'] = (
                0.4 * df['sharpe_60d'] +
                0.3 * df['ret_3m'] / df['vol_60d'] +
                0.3 * df['momentum_120d'] / df['vol_120d']
            )

        return df.sort_values('timing_score', ascending=False) if 'timing_score' in df.columns else df

    def decompose_performance(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights_ts: pd.DataFrame
    ) -> pd.DataFrame:
        """Decompose portfolio performance into factor contributions.

        Args:
            portfolio_returns: Time series of portfolio returns
            portfolio_weights_ts: Time series of portfolio weights

        Returns:
            DataFrame with factor performance attribution
        """
        # Align dates
        common_dates = portfolio_returns.index.intersection(
            self.factor_returns.index
        ).intersection(portfolio_weights_ts.index)

        if len(common_dates) == 0:
            raise ValueError("No common dates found")

        results = []

        for date in common_dates:
            weights = portfolio_weights_ts.loc[date]
            exposures = self.calculate_exposures(weights)

            factor_returns_date = self.factor_returns.loc[date]

            for factor in self.factors:
                factor_contrib = (
                    exposures.exposures[factor] *
                    factor_returns_date[factor]
                )

                results.append({
                    'date': date,
                    'factor': factor,
                    'exposure': exposures.exposures[factor],
                    'factor_return': factor_returns_date[factor],
                    'contribution': factor_contrib
                })

        df = pd.DataFrame(results)

        # Aggregate by factor
        summary = df.groupby('factor').agg({
            'exposure': 'mean',
            'factor_return': 'sum',
            'contribution': 'sum'
        })

        summary['contribution_pct'] = (
            summary['contribution'] / summary['contribution'].sum() * 100
        )

        return summary.sort_values('contribution', ascending=False)

    def calculate_factor_diversification(
        self,
        portfolio_weights: pd.Series
    ) -> float:
        """Calculate factor diversification ratio.

        Args:
            portfolio_weights: Portfolio weights

        Returns:
            Diversification ratio (higher = more diversified)
        """
        exposures = self.calculate_exposures(portfolio_weights)

        # Sum of individual factor volatilities
        factor_vols = []
        for factor in self.factors:
            factor_vol = np.sqrt(
                self.factor_cov.loc[factor, factor]
            ) * abs(exposures.exposures[factor])
            factor_vols.append(factor_vol)

        sum_factor_vols = sum(factor_vols)

        # Portfolio volatility (from factor model)
        portfolio_vol = exposures.total_risk

        # Diversification ratio
        if portfolio_vol > 0:
            div_ratio = sum_factor_vols / portfolio_vol
        else:
            div_ratio = 0.0

        return div_ratio


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)

    # Generate sample data
    n_assets = 30
    n_factors = 5
    n_days = 500

    factors = [f'Factor_{i}' for i in range(n_factors)]
    assets = [f'Asset_{i}' for i in range(n_assets)]

    # Factor returns
    factor_returns = pd.DataFrame(
        np.random.randn(n_days, n_factors) * 0.01,
        columns=factors
    )

    # Factor loadings
    factor_loadings = pd.DataFrame(
        np.random.randn(n_assets, n_factors) * 0.5,
        index=assets,
        columns=factors
    )

    # Benchmark weights
    benchmark = pd.Series(
        np.ones(n_assets) / n_assets,
        index=assets
    )

    # Create optimizer
    optimizer = FactorExposureOptimizer(
        factor_returns=factor_returns,
        asset_factor_loadings=factor_loadings,
        benchmark_weights=benchmark
    )

    # Calculate exposures
    exposures = optimizer.calculate_exposures(benchmark)
    print("Benchmark Factor Exposures:")
    print(exposures.to_dict())

    # Optimize for factor tilt
    target_tilts = {'Factor_0': 0.5, 'Factor_1': -0.3}
    tilted_portfolio = optimizer.optimize_factor_tilt(
        target_tilts=target_tilts,
        max_tracking_error=0.05
    )
    print("\nTilted Portfolio Top Holdings:")
    print(tilted_portfolio.nlargest(5))

    # Timing signals
    timing = optimizer.analyze_factor_timing_signals()
    print("\nFactor Timing Signals:")
    print(timing)
