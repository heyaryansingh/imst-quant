"""Performance attribution analysis for trading strategies.

This module provides tools for understanding the sources of portfolio returns
through factor decomposition, sector attribution, and timing analysis.

Features:
- Brinson attribution (allocation, selection, interaction effects)
- Factor-based attribution (market, size, value, momentum)
- Timing vs selection decomposition
- Contribution analysis by asset
- Rolling attribution windows

Example:
    >>> from imst_quant.utils.attribution import PerformanceAttributor
    >>> attributor = PerformanceAttributor(portfolio_df, benchmark_df)
    >>> brinson = attributor.brinson_attribution()
    >>> print(f"Selection effect: {brinson['selection']:.2%}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class BrinsonAttribution:
    """Results from Brinson attribution analysis.

    Attributes:
        allocation_effect: Return from over/underweighting sectors.
        selection_effect: Return from stock selection within sectors.
        interaction_effect: Combined allocation and selection effect.
        total_active_return: Sum of all effects.
        sector_details: Per-sector breakdown of effects.
    """

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    sector_details: Dict[str, Dict[str, float]]


@dataclass
class FactorAttribution:
    """Results from factor-based attribution analysis.

    Attributes:
        factor_contributions: Dict mapping factor names to return contributions.
        residual: Return not explained by factors (alpha).
        factor_exposures: Dict mapping factor names to exposures (betas).
        r_squared: Percentage of return explained by factors.
    """

    factor_contributions: Dict[str, float]
    residual: float
    factor_exposures: Dict[str, float]
    r_squared: float


class PerformanceAttributor:
    """Analyzer for decomposing portfolio performance into attribution factors.

    Provides Brinson-style attribution, factor analysis, and contribution
    breakdowns to understand sources of return.

    Example:
        >>> # Prepare data with portfolio and benchmark weights/returns
        >>> attributor = PerformanceAttributor(portfolio_df, benchmark_df)
        >>> brinson = attributor.brinson_attribution(sector_col="sector")
    """

    def __init__(
        self,
        portfolio_df: pl.DataFrame,
        benchmark_df: Optional[pl.DataFrame] = None,
        asset_col: str = "asset_id",
        return_col: str = "return",
        weight_col: str = "weight",
        date_col: str = "date",
    ):
        """Initialize performance attributor.

        Args:
            portfolio_df: DataFrame with portfolio holdings, weights, and returns.
            benchmark_df: Optional DataFrame with benchmark weights and returns.
            asset_col: Column name for asset IDs. Default "asset_id".
            return_col: Column name for returns. Default "return".
            weight_col: Column name for weights. Default "weight".
            date_col: Column name for dates. Default "date".
        """
        self.portfolio_df = portfolio_df
        self.benchmark_df = benchmark_df
        self.asset_col = asset_col
        self.return_col = return_col
        self.weight_col = weight_col
        self.date_col = date_col

    def brinson_attribution(
        self,
        sector_col: str = "sector",
    ) -> BrinsonAttribution:
        """Perform Brinson attribution analysis.

        Decomposes active return into allocation, selection, and interaction
        effects at the sector level.

        Formulas:
        - Allocation = sum((wp_s - wb_s) * rb_s)
        - Selection = sum(wb_s * (rp_s - rb_s))
        - Interaction = sum((wp_s - wb_s) * (rp_s - rb_s))

        Args:
            sector_col: Column name for sector classification.

        Returns:
            BrinsonAttribution with all effects and sector details.

        Example:
            >>> brinson = attributor.brinson_attribution(sector_col="sector")
            >>> print(f"Allocation: {brinson.allocation_effect:.4f}")
        """
        if self.benchmark_df is None:
            raise ValueError("Benchmark data required for Brinson attribution")

        # Aggregate to sector level
        portfolio_sectors = (
            self.portfolio_df
            .group_by(sector_col)
            .agg([
                pl.col(self.weight_col).sum().alias("portfolio_weight"),
                (pl.col(self.weight_col) * pl.col(self.return_col)).sum().alias("weighted_return"),
            ])
            .with_columns(
                (pl.col("weighted_return") / pl.col("portfolio_weight")).alias("portfolio_return")
            )
        )

        benchmark_sectors = (
            self.benchmark_df
            .group_by(sector_col)
            .agg([
                pl.col(self.weight_col).sum().alias("benchmark_weight"),
                (pl.col(self.weight_col) * pl.col(self.return_col)).sum().alias("weighted_return"),
            ])
            .with_columns(
                (pl.col("weighted_return") / pl.col("benchmark_weight")).alias("benchmark_return")
            )
        )

        # Join portfolio and benchmark
        combined = portfolio_sectors.join(
            benchmark_sectors,
            on=sector_col,
            how="outer",
            suffix="_b",
        ).fill_null(0)

        # Calculate effects for each sector
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        sector_details = {}

        for row in combined.iter_rows(named=True):
            sector = row[sector_col]
            wp = row["portfolio_weight"]
            wb = row["benchmark_weight"]
            rp = row.get("portfolio_return", 0) or 0
            rb = row.get("benchmark_return", 0) or 0

            # Brinson effects
            alloc = (wp - wb) * rb
            select = wb * (rp - rb)
            interact = (wp - wb) * (rp - rb)

            allocation_effect += alloc
            selection_effect += select
            interaction_effect += interact

            sector_details[sector] = {
                "allocation": alloc,
                "selection": select,
                "interaction": interact,
                "total": alloc + select + interact,
                "portfolio_weight": wp,
                "benchmark_weight": wb,
                "portfolio_return": rp,
                "benchmark_return": rb,
            }

        total_active = allocation_effect + selection_effect + interaction_effect

        return BrinsonAttribution(
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_active_return=total_active,
            sector_details=sector_details,
        )

    def contribution_analysis(self) -> pl.DataFrame:
        """Calculate return contribution by asset.

        Contribution = weight * return for each asset.

        Returns:
            DataFrame with asset contributions sorted by magnitude.

        Example:
            >>> contributions = attributor.contribution_analysis()
            >>> top_contributors = contributions.head(10)
        """
        contributions = self.portfolio_df.with_columns(
            (pl.col(self.weight_col) * pl.col(self.return_col)).alias("contribution")
        )

        # Aggregate by asset
        summary = (
            contributions
            .group_by(self.asset_col)
            .agg([
                pl.col(self.weight_col).mean().alias("avg_weight"),
                pl.col(self.return_col).sum().alias("total_return"),
                pl.col("contribution").sum().alias("total_contribution"),
            ])
            .sort("total_contribution", descending=True)
        )

        return summary

    def timing_vs_selection(self) -> Dict[str, float]:
        """Decompose return into market timing and security selection.

        Timing measures whether the portfolio was more/less invested
        during up/down markets. Selection measures stock picking skill.

        Returns:
            Dict with timing_effect, selection_effect, and total_return.
        """
        if self.benchmark_df is None:
            raise ValueError("Benchmark required for timing analysis")

        # Aggregate portfolio returns by period
        portfolio_returns = (
            self.portfolio_df
            .group_by(self.date_col)
            .agg(
                (pl.col(self.weight_col) * pl.col(self.return_col)).sum().alias("portfolio_return")
            )
            .sort(self.date_col)
        )

        benchmark_returns = (
            self.benchmark_df
            .group_by(self.date_col)
            .agg(
                (pl.col(self.weight_col) * pl.col(self.return_col)).sum().alias("benchmark_return")
            )
            .sort(self.date_col)
        )

        combined = portfolio_returns.join(benchmark_returns, on=self.date_col)

        # Calculate market exposure (beta proxy)
        port_arr = combined["portfolio_return"].to_numpy()
        bench_arr = combined["benchmark_return"].to_numpy()

        # Simple decomposition
        # Timing: correlation of portfolio beta with market direction
        # Selection: average alpha

        # Calculate rolling beta as exposure measure
        if len(bench_arr) >= 20:
            # Use a simple regression approach
            beta = np.cov(port_arr, bench_arr)[0, 1] / np.var(bench_arr) if np.var(bench_arr) > 0 else 1.0
            alpha = np.mean(port_arr) - beta * np.mean(bench_arr)
        else:
            beta = 1.0
            alpha = np.mean(port_arr) - np.mean(bench_arr)

        # Timing effect: exposure changes relative to benchmark
        timing_effect = (beta - 1) * np.sum(bench_arr)

        # Selection effect: stock picking alpha
        selection_effect = alpha * len(port_arr)

        total_return = np.sum(port_arr)

        return {
            "timing_effect": float(timing_effect),
            "selection_effect": float(selection_effect),
            "total_return": float(total_return),
            "beta": float(beta),
            "alpha": float(alpha),
        }


def factor_attribution(
    portfolio_returns: pl.Series | np.ndarray,
    factor_returns: Dict[str, pl.Series | np.ndarray],
    annualization: int = 252,
) -> FactorAttribution:
    """Perform factor-based performance attribution.

    Regresses portfolio returns on factor returns to determine
    exposure (beta) and contribution from each factor.

    Args:
        portfolio_returns: Strategy return series.
        factor_returns: Dict mapping factor names to return series.
        annualization: Periods per year for annualizing. Default 252.

    Returns:
        FactorAttribution with contributions and exposures.

    Example:
        >>> factors = {"market": market_returns, "smb": smb, "hml": hml}
        >>> attr = factor_attribution(strategy_returns, factors)
        >>> print(f"Market contribution: {attr.factor_contributions['market']:.2%}")
    """
    # Convert to numpy
    if isinstance(portfolio_returns, pl.Series):
        y = portfolio_returns.to_numpy()
    else:
        y = np.asarray(portfolio_returns)

    factor_names = list(factor_returns.keys())
    factor_matrix = []

    for name in factor_names:
        f = factor_returns[name]
        if isinstance(f, pl.Series):
            factor_matrix.append(f.to_numpy())
        else:
            factor_matrix.append(np.asarray(f))

    # Align lengths
    min_len = min(len(y), min(len(f) for f in factor_matrix))
    y = y[:min_len]
    X = np.column_stack([f[:min_len] for f in factor_matrix])

    # Add intercept
    X_with_const = np.column_stack([np.ones(min_len), X])

    # OLS regression
    result = np.linalg.lstsq(X_with_const, y, rcond=None)
    coefficients = result[0]

    alpha = coefficients[0]
    betas = coefficients[1:]

    # Calculate contributions (beta * mean factor return * periods)
    contributions = {}
    exposures = {}

    for i, name in enumerate(factor_names):
        exposures[name] = float(betas[i])
        mean_factor = float(np.mean(factor_matrix[i][:min_len]))
        contributions[name] = float(betas[i] * mean_factor * min_len)

    # R-squared
    predicted = X_with_const @ coefficients
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual (annualized alpha)
    residual = float(alpha * annualization)

    return FactorAttribution(
        factor_contributions=contributions,
        residual=residual,
        factor_exposures=exposures,
        r_squared=r_squared,
    )


def calculate_contribution_to_return(
    returns: pl.Series | np.ndarray,
    weights: pl.Series | np.ndarray,
    names: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Calculate each position's contribution to total return.

    Args:
        returns: Return for each position.
        weights: Weight of each position.
        names: Optional position names.

    Returns:
        DataFrame with position contributions.

    Example:
        >>> contrib = calculate_contribution_to_return(
        ...     returns=[0.05, -0.02, 0.03],
        ...     weights=[0.4, 0.3, 0.3],
        ...     names=["AAPL", "GOOGL", "MSFT"]
        ... )
    """
    if isinstance(returns, pl.Series):
        returns = returns.to_numpy()
    else:
        returns = np.asarray(returns)

    if isinstance(weights, pl.Series):
        weights = weights.to_numpy()
    else:
        weights = np.asarray(weights)

    contributions = returns * weights
    total_return = np.sum(contributions)

    if names is None:
        names = [f"Position_{i}" for i in range(len(returns))]

    return pl.DataFrame({
        "position": names,
        "weight": weights.tolist(),
        "return": returns.tolist(),
        "contribution": contributions.tolist(),
        "pct_of_total": (contributions / total_return * 100 if total_return != 0 else [0] * len(contributions)),
    }).sort("contribution", descending=True)


def rolling_attribution(
    portfolio_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    window: int = 20,
    sector_col: str = "sector",
    date_col: str = "date",
    return_col: str = "return",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Calculate rolling Brinson attribution over time.

    Args:
        portfolio_df: Portfolio data with weights and returns.
        benchmark_df: Benchmark data.
        window: Rolling window size in periods.
        sector_col: Column for sector classification.
        date_col: Column for dates.
        return_col: Column for returns.
        weight_col: Column for weights.

    Returns:
        DataFrame with rolling allocation, selection, interaction effects.
    """
    dates = portfolio_df.select(date_col).unique().sort(date_col)
    date_list = dates[date_col].to_list()

    results = []

    for i in range(window - 1, len(date_list)):
        window_dates = date_list[i - window + 1 : i + 1]

        # Filter to window
        port_window = portfolio_df.filter(pl.col(date_col).is_in(window_dates))
        bench_window = benchmark_df.filter(pl.col(date_col).is_in(window_dates))

        if port_window.height == 0 or bench_window.height == 0:
            continue

        # Run attribution
        try:
            attributor = PerformanceAttributor(
                port_window,
                bench_window,
                return_col=return_col,
                weight_col=weight_col,
                date_col=date_col,
            )
            brinson = attributor.brinson_attribution(sector_col=sector_col)

            results.append({
                "date": date_list[i],
                "allocation_effect": brinson.allocation_effect,
                "selection_effect": brinson.selection_effect,
                "interaction_effect": brinson.interaction_effect,
                "total_active_return": brinson.total_active_return,
            })
        except Exception:
            # Skip periods with insufficient data
            continue

    return pl.DataFrame(results)
