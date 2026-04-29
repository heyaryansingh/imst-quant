"""Value at Risk (VaR) and Conditional VaR calculations for risk management.

Provides parametric, historical, and Monte Carlo VaR methods plus CVaR calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Optional, Dict, Any
import structlog

logger = structlog.get_logger()


class VaRCalculator:
    """Calculate Value at Risk and Conditional Value at Risk metrics."""

    def __init__(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: Literal["parametric", "historical", "monte_carlo"] = "historical"
    ):
        """Initialize VaR calculator.

        Args:
            returns: Time series of portfolio returns
            confidence_level: Confidence level for VaR (default: 0.95 for 95%)
            method: VaR calculation method
        """
        self.returns = returns
        self.confidence_level = confidence_level
        self.method = method
        self.alpha = 1 - confidence_level

    def parametric_var(self) -> float:
        """Calculate parametric VaR assuming normal distribution.

        Returns:
            VaR value (positive number representing potential loss)
        """
        mean = self.returns.mean()
        std = self.returns.std()
        z_score = stats.norm.ppf(self.alpha)
        var = -(mean + z_score * std)

        logger.info(
            "parametric_var_calculated",
            var=var,
            mean=mean,
            std=std,
            confidence=self.confidence_level
        )
        return var

    def historical_var(self) -> float:
        """Calculate historical VaR using empirical distribution.

        Returns:
            VaR value (positive number representing potential loss)
        """
        var = -np.percentile(self.returns.dropna(), self.alpha * 100)

        logger.info(
            "historical_var_calculated",
            var=var,
            observations=len(self.returns.dropna()),
            confidence=self.confidence_level
        )
        return var

    def monte_carlo_var(self, num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR through simulation.

        Args:
            num_simulations: Number of Monte Carlo simulations

        Returns:
            VaR value (positive number representing potential loss)
        """
        mean = self.returns.mean()
        std = self.returns.std()

        simulated_returns = np.random.normal(mean, std, num_simulations)
        var = -np.percentile(simulated_returns, self.alpha * 100)

        logger.info(
            "monte_carlo_var_calculated",
            var=var,
            simulations=num_simulations,
            confidence=self.confidence_level
        )
        return var

    def calculate_var(self) -> float:
        """Calculate VaR using the configured method.

        Returns:
            VaR value
        """
        if self.method == "parametric":
            return self.parametric_var()
        elif self.method == "historical":
            return self.historical_var()
        elif self.method == "monte_carlo":
            return self.monte_carlo_var()
        else:
            raise ValueError(f"Unknown VaR method: {self.method}")

    def conditional_var(self) -> float:
        """Calculate Conditional VaR (Expected Shortfall/CVaR).

        Returns the expected loss given that VaR threshold has been exceeded.

        Returns:
            CVaR value (positive number)
        """
        if self.method == "parametric":
            # Analytical CVaR for normal distribution
            mean = self.returns.mean()
            std = self.returns.std()
            z_score = stats.norm.ppf(self.alpha)
            cvar = -(mean - std * stats.norm.pdf(z_score) / self.alpha)
        else:
            # Historical CVaR
            var = -np.percentile(self.returns.dropna(), self.alpha * 100)
            tail_losses = self.returns[self.returns <= -var]
            cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0.0

        logger.info(
            "conditional_var_calculated",
            cvar=cvar,
            method=self.method,
            confidence=self.confidence_level
        )
        return cvar

    def rolling_var(
        self,
        window: int = 252,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling VaR over time.

        Args:
            window: Rolling window size (default: 252 trading days = 1 year)
            min_periods: Minimum observations required

        Returns:
            Time series of VaR values
        """
        if min_periods is None:
            min_periods = window

        def calc_var(returns_slice):
            if len(returns_slice) < min_periods:
                return np.nan
            calculator = VaRCalculator(
                returns_slice,
                self.confidence_level,
                self.method
            )
            return calculator.calculate_var()

        rolling_var = self.returns.rolling(
            window=window,
            min_periods=min_periods
        ).apply(calc_var, raw=False)

        logger.info(
            "rolling_var_calculated",
            window=window,
            observations=len(rolling_var.dropna())
        )
        return rolling_var

    def var_report(self) -> Dict[str, Any]:
        """Generate comprehensive VaR report with multiple methods and metrics.

        Returns:
            Dictionary with VaR metrics and statistics
        """
        parametric_calculator = VaRCalculator(
            self.returns,
            self.confidence_level,
            "parametric"
        )
        historical_calculator = VaRCalculator(
            self.returns,
            self.confidence_level,
            "historical"
        )

        report = {
            "confidence_level": self.confidence_level,
            "observations": len(self.returns.dropna()),
            "var": {
                "parametric": parametric_calculator.parametric_var(),
                "historical": historical_calculator.historical_var(),
                "monte_carlo": historical_calculator.monte_carlo_var(),
            },
            "cvar": {
                "parametric": parametric_calculator.conditional_var(),
                "historical": historical_calculator.conditional_var(),
            },
            "return_stats": {
                "mean": float(self.returns.mean()),
                "std": float(self.returns.std()),
                "min": float(self.returns.min()),
                "max": float(self.returns.max()),
                "skew": float(self.returns.skew()),
                "kurtosis": float(self.returns.kurtosis()),
            }
        }

        logger.info("var_report_generated", report=report)
        return report


def calculate_portfolio_var(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    method: Literal["parametric", "historical", "monte_carlo"] = "historical"
) -> Dict[str, float]:
    """Calculate VaR for a portfolio with multiple positions.

    Args:
        positions: DataFrame with position sizes (columns = assets)
        returns: DataFrame with asset returns (columns = assets)
        confidence_level: Confidence level for VaR
        method: VaR calculation method

    Returns:
        Dictionary with total VaR and marginal VaR by position
    """
    # Calculate portfolio returns
    aligned_positions = positions.reindex(returns.index, method='ffill')
    portfolio_returns = (aligned_positions * returns).sum(axis=1)

    # Calculate total VaR
    calculator = VaRCalculator(portfolio_returns, confidence_level, method)
    total_var = calculator.calculate_var()

    # Calculate marginal VaR for each position
    marginal_vars = {}
    for asset in positions.columns:
        asset_contribution = aligned_positions[asset] * returns[asset]
        asset_calculator = VaRCalculator(asset_contribution, confidence_level, method)
        marginal_vars[asset] = asset_calculator.calculate_var()

    logger.info(
        "portfolio_var_calculated",
        total_var=total_var,
        num_positions=len(positions.columns)
    )

    return {
        "total_var": total_var,
        "marginal_var": marginal_vars
    }


def stress_test_var(
    returns: pd.Series,
    stress_scenarios: Dict[str, float],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Perform VaR stress testing under different scenarios.

    Args:
        returns: Historical returns
        stress_scenarios: Dict of scenario_name -> return_shock (e.g., {"market_crash": -0.20})
        confidence_level: Confidence level for VaR

    Returns:
        VaR under each stress scenario
    """
    results = {}

    for scenario_name, shock in stress_scenarios.items():
        stressed_returns = returns + shock
        calculator = VaRCalculator(stressed_returns, confidence_level, "historical")
        results[scenario_name] = calculator.calculate_var()

    logger.info(
        "stress_test_completed",
        scenarios=list(stress_scenarios.keys()),
        results=results
    )
    return results
