"""Portfolio exposure analysis utilities.

Analyzes portfolio exposure across sectors, asset types, geographies,
and custom groupings to identify concentration risk.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import polars as pl


@dataclass
class ExposureMetrics:
    """Container for exposure analysis metrics."""

    by_sector: Dict[str, float]
    by_asset_type: Dict[str, float]
    by_geography: Optional[Dict[str, float]] = None
    herfindahl_index: float = 0.0
    max_single_exposure: float = 0.0
    concentration_ratio_top5: float = 0.0


class ExposureAnalyzer:
    """Analyzes portfolio concentration and exposure across dimensions."""

    def __init__(self, portfolio_df: pl.DataFrame):
        """Initialize with portfolio holdings.

        Args:
            portfolio_df: DataFrame with columns [symbol, weight, sector, asset_type, geography]
        """
        self.portfolio = portfolio_df
        self._validate_portfolio()

    def _validate_portfolio(self) -> None:
        """Validate portfolio DataFrame has required columns."""
        required = {"symbol", "weight"}
        if not required.issubset(self.portfolio.columns):
            missing = required - set(self.portfolio.columns)
            raise ValueError(f"Portfolio missing required columns: {missing}")

        # Ensure weights sum to ~1.0
        total_weight = self.portfolio["weight"].sum()
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights sum to {total_weight}, expected ~1.0")

    def analyze(
        self,
        include_geography: bool = False,
        custom_grouping: Optional[str] = None,
    ) -> ExposureMetrics:
        """Compute exposure metrics across all dimensions.

        Args:
            include_geography: Whether to include geographic exposure
            custom_grouping: Optional column name for custom grouping

        Returns:
            ExposureMetrics with all computed exposures
        """
        # Sector exposure
        by_sector = self._compute_exposure_by_column("sector")

        # Asset type exposure
        by_asset_type = self._compute_exposure_by_column("asset_type")

        # Geographic exposure (optional)
        by_geography = None
        if include_geography and "geography" in self.portfolio.columns:
            by_geography = self._compute_exposure_by_column("geography")

        # Concentration metrics
        weights = self.portfolio["weight"].to_list()
        hhi = self._compute_herfindahl_index(weights)
        max_exp = max(weights)
        top5_ratio = self._compute_top_n_ratio(weights, n=5)

        return ExposureMetrics(
            by_sector=by_sector,
            by_asset_type=by_asset_type,
            by_geography=by_geography,
            herfindahl_index=hhi,
            max_single_exposure=max_exp,
            concentration_ratio_top5=top5_ratio,
        )

    def _compute_exposure_by_column(self, column: str) -> Dict[str, float]:
        """Compute exposure percentages grouped by a column.

        Args:
            column: Column name to group by

        Returns:
            Dict mapping group to exposure percentage
        """
        if column not in self.portfolio.columns:
            return {}

        grouped = (
            self.portfolio.group_by(column)
            .agg(pl.col("weight").sum().alias("total_weight"))
            .sort("total_weight", descending=True)
        )

        return {
            row[column]: float(row["total_weight"]) for row in grouped.iter_rows(named=True)
        }

    def _compute_herfindahl_index(self, weights: List[float]) -> float:
        """Compute Herfindahl-Hirschman Index (HHI) for concentration.

        HHI ranges from 1/N (perfectly diversified) to 1.0 (single holding).
        Values > 0.25 indicate high concentration.

        Args:
            weights: List of position weights

        Returns:
            HHI value between 0 and 1
        """
        return sum(w**2 for w in weights)

    def _compute_top_n_ratio(self, weights: List[float], n: int = 5) -> float:
        """Compute concentration ratio of top N holdings.

        Args:
            weights: List of position weights
            n: Number of top holdings to consider

        Returns:
            Sum of top N weights
        """
        sorted_weights = sorted(weights, reverse=True)
        return sum(sorted_weights[:n])

    def get_diversification_score(self) -> float:
        """Compute a simple diversification score (0-100).

        Returns:
            Score where 100 is perfectly diversified, 0 is concentrated
        """
        metrics = self.analyze()
        hhi = metrics.herfindahl_index

        # Convert HHI to score: 1/N → 100, 1.0 → 0
        n_holdings = len(self.portfolio)
        ideal_hhi = 1.0 / n_holdings
        score = 100 * (1.0 - (hhi - ideal_hhi) / (1.0 - ideal_hhi))

        return max(0.0, min(100.0, score))

    def identify_concentration_risks(
        self, sector_threshold: float = 0.3, single_threshold: float = 0.1
    ) -> List[str]:
        """Identify concentration risks exceeding thresholds.

        Args:
            sector_threshold: Max acceptable sector exposure (default 30%)
            single_threshold: Max acceptable single position (default 10%)

        Returns:
            List of risk warnings
        """
        metrics = self.analyze()
        risks = []

        # Check sector concentration
        for sector, exposure in metrics.by_sector.items():
            if exposure > sector_threshold:
                risks.append(
                    f"Sector '{sector}' over-concentrated: {exposure:.1%} "
                    f"(threshold: {sector_threshold:.1%})"
                )

        # Check single position concentration
        if metrics.max_single_exposure > single_threshold:
            risks.append(
                f"Largest position is {metrics.max_single_exposure:.1%} "
                f"(threshold: {single_threshold:.1%})"
            )

        # Check overall concentration via HHI
        if metrics.herfindahl_index > 0.25:
            risks.append(
                f"Portfolio highly concentrated (HHI={metrics.herfindahl_index:.3f})"
            )

        return risks


def format_exposure_report(metrics: ExposureMetrics) -> str:
    """Format exposure metrics into a readable report.

    Args:
        metrics: ExposureMetrics to format

    Returns:
        Formatted multi-line string report
    """
    lines = ["Portfolio Exposure Analysis", "=" * 50, ""]

    # Sector exposure
    lines.append("Sector Exposure:")
    for sector, pct in sorted(metrics.by_sector.items(), key=lambda x: -x[1]):
        lines.append(f"  {sector:20s} {pct:6.2%}")
    lines.append("")

    # Asset type exposure
    lines.append("Asset Type Exposure:")
    for asset_type, pct in sorted(metrics.by_asset_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {asset_type:20s} {pct:6.2%}")
    lines.append("")

    # Geographic exposure (if available)
    if metrics.by_geography:
        lines.append("Geographic Exposure:")
        for geo, pct in sorted(metrics.by_geography.items(), key=lambda x: -x[1]):
            lines.append(f"  {geo:20s} {pct:6.2%}")
        lines.append("")

    # Concentration metrics
    lines.append("Concentration Metrics:")
    lines.append(f"  Herfindahl Index:       {metrics.herfindahl_index:.4f}")
    lines.append(f"  Max Single Position:    {metrics.max_single_exposure:.2%}")
    lines.append(f"  Top 5 Concentration:    {metrics.concentration_ratio_top5:.2%}")

    return "\n".join(lines)
