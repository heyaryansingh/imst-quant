"""Alpha generation and attribution metrics for strategy evaluation.

This module provides comprehensive metrics for measuring alpha generation,
attribution analysis, and information ratio decomposition for quantitative
trading strategies.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class AlphaDecomposition:
    """Decomposition of alpha sources."""
    total_alpha: float
    selection_alpha: float  # Stock selection
    timing_alpha: float  # Market timing
    interaction_alpha: float  # Selection x Timing
    residual_alpha: float


class AlphaMetrics:
    """Comprehensive alpha generation metrics."""

    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ):
        """Initialize alpha metrics calculator.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.excess_returns = strategy_returns - risk_free_rate / 252

    def calculate_jensens_alpha(self) -> float:
        """Calculate Jensen's Alpha (CAPM alpha).

        Returns:
            Annualized Jensen's alpha
        """
        # Run CAPM regression
        excess_benchmark = self.benchmark_returns - self.risk_free_rate / 252

        X = np.column_stack([np.ones(len(excess_benchmark)), excess_benchmark])
        y = self.excess_returns

        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha = beta[0]

        # Annualize
        return alpha * 252

    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio.

        Returns:
            Annualized information ratio
        """
        active_returns = self.strategy_returns - self.benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        return (active_returns.mean() * 252) / tracking_error

    def calculate_treynor_ratio(self) -> float:
        """Calculate Treynor Ratio.

        Returns:
            Annualized Treynor ratio
        """
        beta = self._calculate_beta()
        if beta == 0:
            return 0.0

        excess_return = self.strategy_returns.mean() * 252 - self.risk_free_rate
        return excess_return / beta

    def calculate_appraisal_ratio(self) -> float:
        """Calculate Appraisal Ratio (alpha / unsystematic risk).

        Returns:
            Appraisal ratio
        """
        alpha = self.calculate_jensens_alpha()
        unsystematic_risk = self._calculate_unsystematic_risk()

        if unsystematic_risk == 0:
            return 0.0

        return alpha / unsystematic_risk

    def calculate_m2_alpha(self) -> float:
        """Calculate M-squared alpha (Modigliani-Modigliani measure).

        Returns:
            M2 alpha in percentage
        """
        # Calculate strategy Sharpe
        strategy_sharpe = (
            (self.strategy_returns.mean() * 252 - self.risk_free_rate) /
            (self.strategy_returns.std() * np.sqrt(252))
        )

        # Calculate benchmark volatility
        benchmark_vol = self.benchmark_returns.std() * np.sqrt(252)

        # M2 alpha
        m2_return = self.risk_free_rate + strategy_sharpe * benchmark_vol
        benchmark_return = self.benchmark_returns.mean() * 252

        return (m2_return - benchmark_return) * 100

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_returns: pd.DataFrame
    ) -> AlphaDecomposition:
        """Perform Brinson attribution analysis.

        Args:
            portfolio_weights: Portfolio weights by sector over time
            benchmark_weights: Benchmark weights by sector over time
            sector_returns: Sector returns over time

        Returns:
            AlphaDecomposition with attribution breakdown
        """
        # Allocation effect: (w_p - w_b) * (r_s - r_b)
        weight_diff = portfolio_weights - benchmark_weights
        benchmark_total = (benchmark_weights * sector_returns).sum(axis=1)
        sector_excess = sector_returns.sub(benchmark_total, axis=0)
        allocation = (weight_diff * sector_excess).sum(axis=1).mean()

        # Selection effect: w_b * (r_p - r_s)
        # Approximated by active returns weighted by benchmark
        selection = ((portfolio_weights - benchmark_weights) * sector_returns).sum(axis=1).mean()

        # Interaction effect
        interaction = (weight_diff * (sector_returns.sub(sector_returns.mean(), axis=0))).sum(axis=1).mean()

        # Total alpha
        total = allocation + selection + interaction

        return AlphaDecomposition(
            total_alpha=total,
            selection_alpha=selection,
            timing_alpha=allocation,
            interaction_alpha=interaction,
            residual_alpha=0.0
        )

    def calculate_alpha_decay(
        self,
        holding_periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """Calculate alpha decay over different holding periods.

        Args:
            holding_periods: List of holding periods (days)

        Returns:
            DataFrame with alpha metrics by holding period
        """
        results = []

        for period in holding_periods:
            # Calculate returns over holding period
            period_returns = self.strategy_returns.rolling(period).sum()
            period_benchmark = self.benchmark_returns.rolling(period).sum()

            # Calculate alpha for this period
            excess = period_returns - self.risk_free_rate * period / 252
            excess_benchmark = period_benchmark - self.risk_free_rate * period / 252

            X = np.column_stack([
                np.ones(len(excess_benchmark.dropna())),
                excess_benchmark.dropna()
            ])
            y = excess.dropna()

            if len(X) > 0 and len(y) > 0:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha = beta[0] * 252 / period

                results.append({
                    'holding_period': period,
                    'alpha': alpha,
                    'alpha_tstat': self._calculate_tstat(y - X @ beta),
                    'observations': len(y)
                })

        return pd.DataFrame(results)

    def calculate_skill_vs_luck(
        self,
        n_simulations: int = 10000
    ) -> Dict[str, float]:
        """Assess whether returns are driven by skill vs luck.

        Args:
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with skill assessment metrics
        """
        # Calculate observed Sharpe ratio
        observed_sharpe = (
            (self.strategy_returns.mean() * 252 - self.risk_free_rate) /
            (self.strategy_returns.std() * np.sqrt(252))
        )

        # Simulate random strategies with same volatility
        vol = self.strategy_returns.std()
        simulated_sharpes = []

        for _ in range(n_simulations):
            random_returns = np.random.normal(0, vol, len(self.strategy_returns))
            random_sharpe = (
                (random_returns.mean() * 252 - self.risk_free_rate) /
                (random_returns.std() * np.sqrt(252))
            )
            simulated_sharpes.append(random_sharpe)

        simulated_sharpes = np.array(simulated_sharpes)

        # Calculate p-value
        p_value = (simulated_sharpes >= observed_sharpe).mean()

        # Calculate probabilistic Sharpe ratio (PSR)
        psr = stats.norm.cdf(
            (observed_sharpe - 0) /
            (self.strategy_returns.std() / np.sqrt(len(self.strategy_returns)))
        )

        return {
            'observed_sharpe': observed_sharpe,
            'p_value': p_value,
            'probabilistic_sharpe_ratio': psr,
            'skill_probability': 1 - p_value,
            'median_random_sharpe': np.median(simulated_sharpes)
        }

    def calculate_capture_ratios(self) -> Dict[str, float]:
        """Calculate upside and downside capture ratios.

        Returns:
            Dictionary with capture ratios
        """
        # Split into up and down markets
        up_mask = self.benchmark_returns > 0
        down_mask = self.benchmark_returns < 0

        # Upside capture
        strategy_up = self.strategy_returns[up_mask].mean() * 252
        benchmark_up = self.benchmark_returns[up_mask].mean() * 252
        upside_capture = (strategy_up / benchmark_up * 100) if benchmark_up != 0 else 0

        # Downside capture
        strategy_down = self.strategy_returns[down_mask].mean() * 252
        benchmark_down = self.benchmark_returns[down_mask].mean() * 252
        downside_capture = (strategy_down / benchmark_down * 100) if benchmark_down != 0 else 0

        # Capture ratio
        capture_ratio = upside_capture / abs(downside_capture) if downside_capture != 0 else 0

        return {
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'capture_ratio': capture_ratio,
            'up_periods': up_mask.sum(),
            'down_periods': down_mask.sum()
        }

    def calculate_active_share(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series
    ) -> float:
        """Calculate active share metric.

        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights

        Returns:
            Active share (0-100%)
        """
        # Align indices
        all_assets = portfolio_weights.index.union(benchmark_weights.index)
        port = portfolio_weights.reindex(all_assets, fill_value=0)
        bench = benchmark_weights.reindex(all_assets, fill_value=0)

        # Active share is half the sum of absolute differences
        active_share = 0.5 * np.abs(port - bench).sum()
        return active_share * 100

    def calculate_transfer_coefficient(
        self,
        forecasts: pd.DataFrame,
        realized_returns: pd.DataFrame
    ) -> float:
        """Calculate transfer coefficient (forecast realization).

        Args:
            forecasts: Forecasted returns or rankings
            realized_returns: Realized returns

        Returns:
            Transfer coefficient (correlation between forecast and realized)
        """
        # Flatten and calculate correlation
        flat_forecasts = forecasts.values.flatten()
        flat_realized = realized_returns.values.flatten()

        # Remove NaN
        mask = ~(np.isnan(flat_forecasts) | np.isnan(flat_realized))
        flat_forecasts = flat_forecasts[mask]
        flat_realized = flat_realized[mask]

        if len(flat_forecasts) < 2:
            return 0.0

        return np.corrcoef(flat_forecasts, flat_realized)[0, 1]

    def _calculate_beta(self) -> float:
        """Calculate beta relative to benchmark."""
        cov = np.cov(self.strategy_returns, self.benchmark_returns)[0, 1]
        var = np.var(self.benchmark_returns)
        return cov / var if var > 0 else 0.0

    def _calculate_unsystematic_risk(self) -> float:
        """Calculate unsystematic (idiosyncratic) risk."""
        beta = self._calculate_beta()
        systematic_returns = beta * self.benchmark_returns
        residuals = self.strategy_returns - systematic_returns
        return residuals.std() * np.sqrt(252)

    def _calculate_tstat(self, residuals: pd.Series) -> float:
        """Calculate t-statistic for residuals."""
        if len(residuals) < 2:
            return 0.0
        return residuals.mean() / (residuals.std() / np.sqrt(len(residuals)))

    def generate_alpha_report(self) -> pd.DataFrame:
        """Generate comprehensive alpha metrics report.

        Returns:
            DataFrame with all alpha metrics
        """
        metrics = {
            'Jensens Alpha': self.calculate_jensens_alpha(),
            'Information Ratio': self.calculate_information_ratio(),
            'Treynor Ratio': self.calculate_treynor_ratio(),
            'Appraisal Ratio': self.calculate_appraisal_ratio(),
            'M2 Alpha': self.calculate_m2_alpha(),
            'Beta': self._calculate_beta(),
            'Unsystematic Risk': self._calculate_unsystematic_risk(),
        }

        # Add capture ratios
        capture = self.calculate_capture_ratios()
        metrics.update(capture)

        # Add skill assessment
        skill = self.calculate_skill_vs_luck()
        metrics.update(skill)

        return pd.DataFrame([metrics])


def calculate_fundamental_law_alpha(
    information_coefficient: float,
    breadth: int,
    transfer_coefficient: float = 1.0
) -> float:
    """Calculate expected alpha using Grinold's Fundamental Law.

    Args:
        information_coefficient: IC (correlation between forecast and outcome)
        breadth: Number of independent bets per year
        transfer_coefficient: Ability to implement forecasts

    Returns:
        Expected information ratio
    """
    return information_coefficient * np.sqrt(breadth) * transfer_coefficient


def decompose_information_ratio(
    ic: float,
    breadth: int,
    tc: float,
    actual_ir: float
) -> Dict[str, float]:
    """Decompose Information Ratio into components.

    Args:
        ic: Information coefficient
        breadth: Number of bets
        tc: Transfer coefficient
        actual_ir: Actual information ratio achieved

    Returns:
        Dictionary with IR decomposition
    """
    expected_ir = calculate_fundamental_law_alpha(ic, breadth, tc)

    return {
        'expected_ir': expected_ir,
        'actual_ir': actual_ir,
        'ic_contribution': ic * np.sqrt(breadth),
        'tc_impact': tc,
        'breadth_benefit': np.sqrt(breadth),
        'implementation_gap': actual_ir - expected_ir
    }
