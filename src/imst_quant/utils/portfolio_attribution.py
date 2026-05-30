"""Portfolio attribution analysis utilities.

Decomposes portfolio returns into contribution sources: asset allocation,
security selection, interaction effects, and factor exposures.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls


class BrinsonAttribution:
    """Brinson-Fachler attribution model.

    Decomposes active return (portfolio vs benchmark) into:
    - Allocation effect: return from over/underweighting sectors
    - Selection effect: return from picking winning stocks within sectors
    - Interaction effect: combined allocation and selection
    """

    def __init__(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None
    ):
        """Initialize attribution analyzer.

        Args:
            portfolio_weights: DataFrame with dates as index, assets as columns
            benchmark_weights: DataFrame with dates as index, assets as columns
            portfolio_returns: DataFrame with dates as index, assets as columns
            benchmark_returns: DataFrame with dates as index, assets as columns
            sector_map: Optional mapping of asset -> sector for grouping
        """
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.sector_map = sector_map or {}

    def compute_attribution(self, period: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Compute Brinson attribution for given period.

        Args:
            period: Optional (start_date, end_date) tuple, defaults to full history

        Returns:
            DataFrame with columns: allocation, selection, interaction, total_active
        """
        if period:
            start, end = period
            pw = self.portfolio_weights.loc[start:end]
            bw = self.benchmark_weights.loc[start:end]
            pr = self.portfolio_returns.loc[start:end]
            br = self.benchmark_returns.loc[start:end]
        else:
            pw, bw, pr, br = (
                self.portfolio_weights,
                self.benchmark_weights,
                self.portfolio_returns,
                self.benchmark_returns
            )

        # Align indices
        common_idx = pw.index.intersection(bw.index).intersection(
            pr.index).intersection(br.index)
        pw = pw.loc[common_idx]
        bw = bw.loc[common_idx]
        pr = pr.loc[common_idx]
        br = br.loc[common_idx]

        # Allocation effect: (wp - wb) * rb
        allocation = (pw - bw).multiply(br, axis=0)

        # Selection effect: wb * (rp - rb)
        selection = bw.multiply(pr - br, axis=0)

        # Interaction effect: (wp - wb) * (rp - rb)
        interaction = (pw - bw).multiply(pr - br, axis=0)

        results = pd.DataFrame({
            'allocation': allocation.sum(axis=1),
            'selection': selection.sum(axis=1),
            'interaction': interaction.sum(axis=1)
        })
        results['total_active'] = results.sum(axis=1)

        return results

    def sector_attribution(self) -> pd.DataFrame:
        """Compute attribution grouped by sectors.

        Returns:
            DataFrame with sector-level attribution breakdown
        """
        if not self.sector_map:
            raise ValueError("sector_map required for sector attribution")

        sector_results = []
        for sector in set(self.sector_map.values()):
            sector_assets = [a for a, s in self.sector_map.items() if s == sector]
            sector_assets = [a for a in sector_assets if a in self.portfolio_weights.columns]

            if not sector_assets:
                continue

            # Sum weights and returns within sector
            pw_sector = self.portfolio_weights[sector_assets].sum(axis=1)
            bw_sector = self.benchmark_weights[sector_assets].sum(axis=1)
            pr_sector = (
                self.portfolio_weights[sector_assets]
                .multiply(self.portfolio_returns[sector_assets], axis=0)
                .sum(axis=1)
            ) / pw_sector.replace(0, np.nan)
            br_sector = (
                self.benchmark_weights[sector_assets]
                .multiply(self.benchmark_returns[sector_assets], axis=0)
                .sum(axis=1)
            ) / bw_sector.replace(0, np.nan)

            allocation = (pw_sector - bw_sector) * br_sector
            selection = bw_sector * (pr_sector - br_sector)
            interaction = (pw_sector - bw_sector) * (pr_sector - br_sector)

            sector_results.append({
                'sector': sector,
                'allocation': allocation.sum(),
                'selection': selection.sum(),
                'interaction': interaction.sum(),
                'total': allocation.sum() + selection.sum() + interaction.sum()
            })

        return pd.DataFrame(sector_results).set_index('sector')


class FactorAttribution:
    """Factor-based return attribution using regression.

    Attributes portfolio returns to systematic factor exposures
    and stock-specific (idiosyncratic) returns.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None
    ):
        """Initialize factor attribution.

        Args:
            portfolio_returns: Series of portfolio returns
            factor_returns: DataFrame with factors as columns (e.g., Fama-French)
            risk_free_rate: Optional risk-free rate series
        """
        self.portfolio_returns = portfolio_returns
        self.factor_returns = factor_returns
        self.risk_free_rate = risk_free_rate or pd.Series(0, index=portfolio_returns.index)

    def run_regression(self) -> Tuple[pd.Series, float, float]:
        """Run factor regression.

        Returns:
            Tuple of (factor_loadings, alpha, r_squared)
        """
        # Align data
        common_idx = self.portfolio_returns.index.intersection(
            self.factor_returns.index).intersection(self.risk_free_rate.index)

        y = (self.portfolio_returns.loc[common_idx] -
             self.risk_free_rate.loc[common_idx]).values
        X = self.factor_returns.loc[common_idx].values

        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha = beta[0]
        factor_betas = pd.Series(beta[1:], index=self.factor_returns.columns)

        # R-squared
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return factor_betas, alpha, r_squared

    def decompose_returns(self) -> pd.DataFrame:
        """Decompose returns into factor contributions.

        Returns:
            DataFrame with factor contributions per period
        """
        factor_betas, alpha, _ = self.run_regression()

        # Align data
        common_idx = self.portfolio_returns.index.intersection(
            self.factor_returns.index)

        # Factor contributions
        factor_contrib = self.factor_returns.loc[common_idx].multiply(
            factor_betas, axis=1
        )
        factor_contrib['alpha'] = alpha
        factor_contrib['total'] = factor_contrib.sum(axis=1)

        return factor_contrib

    def rolling_attribution(
        self,
        window: int = 60,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute rolling factor loadings.

        Args:
            window: Rolling window size in periods
            min_periods: Minimum periods required, defaults to window

        Returns:
            DataFrame with rolling factor loadings over time
        """
        min_periods = min_periods or window
        common_idx = self.portfolio_returns.index.intersection(
            self.factor_returns.index)

        loadings_list = []
        for i in range(len(common_idx)):
            if i < min_periods - 1:
                continue

            start_idx = max(0, i - window + 1)
            end_idx = i + 1

            window_idx = common_idx[start_idx:end_idx]
            y = (self.portfolio_returns.loc[window_idx] -
                 self.risk_free_rate.loc[window_idx]).values
            X = self.factor_returns.loc[window_idx].values
            X = np.column_stack([np.ones(len(X)), X])

            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            loadings = {'date': common_idx[i], 'alpha': beta[0]}
            for j, col in enumerate(self.factor_returns.columns):
                loadings[col] = beta[j + 1]
            loadings_list.append(loadings)

        return pd.DataFrame(loadings_list).set_index('date')


class PerformanceContribution:
    """Calculate contribution of individual positions to portfolio return.

    Shows which positions helped/hurt performance over time.
    """

    @staticmethod
    def position_contribution(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_costs: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate position-level contribution to portfolio return.

        Args:
            weights: Position weights (dates x assets)
            returns: Position returns (dates x assets)
            transaction_costs: Optional transaction costs (dates x assets)

        Returns:
            DataFrame with position contribution to total return
        """
        # Contribution = weight * return
        contribution = weights.multiply(returns, axis=0)

        if transaction_costs is not None:
            contribution = contribution - transaction_costs

        # Add total column
        result = contribution.copy()
        result['total_return'] = contribution.sum(axis=1)

        return result

    @staticmethod
    def cumulative_contribution(
        weights: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate cumulative contribution of each position.

        Args:
            weights: Position weights (dates x assets)
            returns: Position returns (dates x assets)

        Returns:
            DataFrame with cumulative contribution over time
        """
        contribution = weights.multiply(returns, axis=0)
        cumulative = contribution.cumsum()
        cumulative['total_return'] = cumulative.sum(axis=1)

        return cumulative

    @staticmethod
    def top_contributors(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        n: int = 10,
        period: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """Identify top contributing positions.

        Args:
            weights: Position weights (dates x assets)
            returns: Position returns (dates x assets)
            n: Number of top contributors to return
            period: Optional (start, end) date range

        Returns:
            DataFrame with top N contributors ranked by contribution
        """
        if period:
            start, end = period
            weights = weights.loc[start:end]
            returns = returns.loc[start:end]

        contribution = weights.multiply(returns, axis=0).sum(axis=0)
        top_n = contribution.nlargest(n)

        result = pd.DataFrame({
            'asset': top_n.index,
            'total_contribution': top_n.values,
            'avg_weight': weights[top_n.index].mean(axis=0).values,
            'avg_return': returns[top_n.index].mean(axis=0).values
        })

        return result.sort_values('total_contribution', ascending=False)


def attribution_summary(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive attribution summary.

    Args:
        portfolio_weights: Portfolio weights over time
        benchmark_weights: Benchmark weights over time
        portfolio_returns: Portfolio returns over time
        benchmark_returns: Benchmark returns over time
        factor_returns: Optional factor returns for factor attribution

    Returns:
        Dictionary with attribution results from multiple methods
    """
    results = {}

    # Brinson attribution
    brinson = BrinsonAttribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_returns,
        benchmark_returns
    )
    results['brinson'] = brinson.compute_attribution()

    # Position contribution
    results['position_contrib'] = PerformanceContribution.position_contribution(
        portfolio_weights, portfolio_returns
    )

    # Factor attribution (if factors provided)
    if factor_returns is not None:
        portfolio_total_return = portfolio_weights.multiply(
            portfolio_returns, axis=0
        ).sum(axis=1)
        factor_attr = FactorAttribution(portfolio_total_return, factor_returns)
        results['factor_loadings'], results['alpha'], results['r_squared'] = (
            factor_attr.run_regression()
        )
        results['factor_decomp'] = factor_attr.decompose_returns()

    return results
