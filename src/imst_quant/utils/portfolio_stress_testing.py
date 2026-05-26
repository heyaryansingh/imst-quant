"""Portfolio stress testing for scenario analysis and risk assessment.

This module provides comprehensive stress testing capabilities for evaluating
portfolio performance under various market scenarios, historical crisis events,
and custom shock scenarios.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Types of stress test scenarios."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    FACTOR_SHOCK = "factor_shock"
    CORRELATION_BREAKDOWN = "correlation_breakdown"


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    scenario_type: ScenarioType
    returns_shock: Dict[str, float]  # Asset -> return shock (%)
    volatility_multiplier: float = 1.0
    correlation_shock: Optional[np.ndarray] = None
    duration_days: int = 1


class HistoricalScenarios:
    """Pre-defined historical crisis scenarios."""

    BLACK_MONDAY_1987 = StressScenario(
        name="Black Monday (Oct 1987)",
        scenario_type=ScenarioType.HISTORICAL,
        returns_shock={"SPY": -20.5, "TLT": 2.0, "GLD": 0.5},
        volatility_multiplier=3.0,
        duration_days=1
    )

    DOTCOM_CRASH_2000 = StressScenario(
        name="Dot-com Crash (2000-2002)",
        scenario_type=ScenarioType.HISTORICAL,
        returns_shock={"SPY": -45.0, "QQQ": -78.0, "TLT": 15.0},
        volatility_multiplier=2.0,
        duration_days=252
    )

    FINANCIAL_CRISIS_2008 = StressScenario(
        name="Financial Crisis (2008)",
        scenario_type=ScenarioType.HISTORICAL,
        returns_shock={"SPY": -38.5, "TLT": 20.0, "GLD": 5.5},
        volatility_multiplier=2.5,
        duration_days=252
    )

    COVID_CRASH_2020 = StressScenario(
        name="COVID-19 Crash (Mar 2020)",
        scenario_type=ScenarioType.HISTORICAL,
        returns_shock={"SPY": -34.0, "TLT": 11.0, "GLD": -0.5},
        volatility_multiplier=4.0,
        duration_days=30
    )

    RATE_SHOCK_2022 = StressScenario(
        name="Rate Shock (2022)",
        scenario_type=ScenarioType.HISTORICAL,
        returns_shock={"SPY": -18.0, "TLT": -30.0, "GLD": -1.0},
        volatility_multiplier=1.5,
        duration_days=252
    )


class PortfolioStressTester:
    """Comprehensive portfolio stress testing framework."""

    def __init__(
        self,
        positions: pd.Series,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None
    ):
        """Initialize stress tester.

        Args:
            positions: Current portfolio positions (shares or weights)
            returns: Historical returns for each asset
            prices: Current prices for position valuation
        """
        self.positions = positions
        self.returns = returns
        self.prices = prices

        # Calculate portfolio value
        if prices is not None:
            self.portfolio_value = (positions * prices).sum()
        else:
            self.portfolio_value = positions.sum()

    def apply_scenario(
        self,
        scenario: StressScenario
    ) -> Dict[str, Union[float, pd.Series]]:
        """Apply stress scenario to portfolio.

        Args:
            scenario: Stress scenario to apply

        Returns:
            Dictionary with stress test results
        """
        results = {
            'scenario_name': scenario.name,
            'scenario_type': scenario.scenario_type.value,
        }

        # Calculate portfolio shock
        portfolio_shock = 0.0
        for asset, shock in scenario.returns_shock.items():
            if asset in self.positions.index:
                weight = self.positions[asset] / self.positions.sum()
                portfolio_shock += weight * (shock / 100)

        results['portfolio_return'] = portfolio_shock
        results['portfolio_loss'] = self.portfolio_value * portfolio_shock

        # Calculate VaR and CVaR under stress
        stressed_returns = self._apply_volatility_shock(
            self.returns,
            scenario.volatility_multiplier
        )

        results['stressed_var_95'] = np.percentile(stressed_returns.sum(axis=1), 5)
        results['stressed_cvar_95'] = stressed_returns.sum(axis=1)[
            stressed_returns.sum(axis=1) <= results['stressed_var_95']
        ].mean()

        # Maximum drawdown under stress
        cumulative = (1 + stressed_returns.sum(axis=1)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        results['max_drawdown_stressed'] = drawdown.min()

        return results

    def run_historical_scenarios(self) -> pd.DataFrame:
        """Run all pre-defined historical scenarios.

        Returns:
            DataFrame with results for each scenario
        """
        scenarios = [
            HistoricalScenarios.BLACK_MONDAY_1987,
            HistoricalScenarios.DOTCOM_CRASH_2000,
            HistoricalScenarios.FINANCIAL_CRISIS_2008,
            HistoricalScenarios.COVID_CRASH_2020,
            HistoricalScenarios.RATE_SHOCK_2022,
        ]

        results = []
        for scenario in scenarios:
            result = self.apply_scenario(scenario)
            results.append(result)

        return pd.DataFrame(results)

    def factor_shock_analysis(
        self,
        factor_shocks: Dict[str, float],
        factor_exposures: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze portfolio impact of factor shocks.

        Args:
            factor_shocks: Factor -> shock magnitude (%)
            factor_exposures: Asset x Factor exposure matrix

        Returns:
            Dictionary with factor shock results
        """
        # Calculate portfolio factor exposures
        portfolio_exposures = factor_exposures.T @ self.positions / self.positions.sum()

        # Calculate impact
        total_impact = 0.0
        factor_impacts = {}

        for factor, shock in factor_shocks.items():
            if factor in portfolio_exposures.index:
                impact = portfolio_exposures[factor] * (shock / 100)
                factor_impacts[factor] = impact
                total_impact += impact

        return {
            'total_impact': total_impact,
            'factor_impacts': factor_impacts,
            'portfolio_loss': self.portfolio_value * total_impact
        }

    def monte_carlo_stress(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        shock_magnitude: float = 2.0
    ) -> Dict[str, float]:
        """Monte Carlo stress testing with tail scenarios.

        Args:
            n_simulations: Number of simulations to run
            confidence_level: Confidence level for VaR/CVaR
            shock_magnitude: Standard deviations for shock

        Returns:
            Dictionary with Monte Carlo stress results
        """
        # Calculate covariance matrix
        cov_matrix = self.returns.cov()

        # Generate shocked scenarios
        mean_returns = self.returns.mean()
        shocked_mean = mean_returns - shock_magnitude * self.returns.std()

        simulated_returns = np.random.multivariate_normal(
            shocked_mean,
            cov_matrix * (shock_magnitude ** 2),
            n_simulations
        )

        # Calculate portfolio returns
        weights = self.positions / self.positions.sum()
        portfolio_returns = simulated_returns @ weights

        # Calculate metrics
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(portfolio_returns, var_percentile)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return {
            'mean_stressed_return': portfolio_returns.mean(),
            'stressed_volatility': portfolio_returns.std(),
            f'var_{int(confidence_level*100)}': var,
            f'cvar_{int(confidence_level*100)}': cvar,
            'worst_case': portfolio_returns.min(),
            'expected_shortfall': self.portfolio_value * cvar
        }

    def correlation_breakdown_test(
        self,
        breakdown_magnitude: float = 0.5
    ) -> Dict[str, float]:
        """Test portfolio under correlation breakdown scenarios.

        Args:
            breakdown_magnitude: Magnitude of correlation increase (0-1)

        Returns:
            Dictionary with correlation breakdown results
        """
        # Calculate original correlation
        corr_matrix = self.returns.corr()

        # Create stressed correlation (increase all correlations)
        stressed_corr = corr_matrix + breakdown_magnitude * (1 - corr_matrix)
        np.fill_diagonal(stressed_corr.values, 1.0)

        # Convert to covariance
        std_devs = self.returns.std()
        stressed_cov = stressed_corr * np.outer(std_devs, std_devs)

        # Calculate portfolio volatility under stress
        weights = self.positions / self.positions.sum()
        original_vol = np.sqrt(weights @ self.returns.cov() @ weights)
        stressed_vol = np.sqrt(weights @ stressed_cov @ weights)

        return {
            'original_volatility': original_vol,
            'stressed_volatility': stressed_vol,
            'volatility_increase': (stressed_vol / original_vol - 1) * 100,
            'diversification_loss': (stressed_vol - original_vol) / original_vol * 100
        }

    def _apply_volatility_shock(
        self,
        returns: pd.DataFrame,
        multiplier: float
    ) -> pd.DataFrame:
        """Apply volatility shock to returns."""
        mean_returns = returns.mean()
        demeaned = returns - mean_returns
        shocked = demeaned * multiplier + mean_returns
        return shocked

    def generate_stress_report(self) -> pd.DataFrame:
        """Generate comprehensive stress testing report.

        Returns:
            DataFrame with all stress test results
        """
        report = []

        # Historical scenarios
        historical = self.run_historical_scenarios()
        report.append(('Historical Scenarios', historical))

        # Monte Carlo stress
        mc_results = self.monte_carlo_stress()
        mc_df = pd.DataFrame([mc_results])
        report.append(('Monte Carlo Stress', mc_df))

        # Correlation breakdown
        corr_results = self.correlation_breakdown_test()
        corr_df = pd.DataFrame([corr_results])
        report.append(('Correlation Breakdown', corr_df))

        return report


def calculate_tail_risk_contribution(
    positions: pd.Series,
    returns: pd.DataFrame,
    confidence_level: float = 0.95
) -> pd.Series:
    """Calculate tail risk contribution of each asset.

    Args:
        positions: Portfolio positions
        returns: Asset returns
        confidence_level: Confidence level for tail risk

    Returns:
        Series with tail risk contribution by asset
    """
    weights = positions / positions.sum()
    portfolio_returns = (returns * weights).sum(axis=1)

    # Calculate tail threshold
    var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    tail_returns = returns[portfolio_returns <= var_threshold]

    # Calculate contribution
    tail_contributions = (tail_returns * weights).mean()
    return tail_contributions / tail_contributions.sum()


def liquidity_stress_test(
    positions: pd.Series,
    volumes: pd.DataFrame,
    liquidation_horizon: int = 5,
    participation_rate: float = 0.1
) -> Dict[str, float]:
    """Test portfolio liquidation under stressed liquidity.

    Args:
        positions: Current positions
        volumes: Historical volume data
        liquidation_horizon: Days to liquidate
        participation_rate: Maximum % of volume we can trade

    Returns:
        Dictionary with liquidation analysis
    """
    results = {}

    for asset in positions.index:
        if asset not in volumes.columns:
            continue

        position = positions[asset]
        avg_volume = volumes[asset].mean()

        # Can we liquidate within horizon?
        max_daily_trade = avg_volume * participation_rate
        days_to_liquidate = abs(position) / max_daily_trade

        results[asset] = {
            'position': position,
            'avg_daily_volume': avg_volume,
            'days_to_liquidate': days_to_liquidate,
            'can_liquidate_in_horizon': days_to_liquidate <= liquidation_horizon,
            'market_impact_risk': 'HIGH' if days_to_liquidate > liquidation_horizon else 'LOW'
        }

    return results
