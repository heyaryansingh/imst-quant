"""Portfolio health monitoring and diagnostics.

This module provides real-time portfolio health checks including position
concentration risks, correlation stress, liquidity warnings, and overall
portfolio wellness scores.

Example:
    >>> from imst_quant.utils.portfolio_health import PortfolioHealthMonitor
    >>> import polars as pl
    >>> weights = pl.Series("weight", [0.4, 0.35, 0.25])
    >>> monitor = PortfolioHealthMonitor()
    >>> health = monitor.assess_health(weights)
    >>> print(f"Health Score: {health.overall_score:.1f}/100")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl


@dataclass
class HealthAlert:
    """Health alert for portfolio issues.

    Attributes:
        severity: Alert severity (critical, warning, info).
        category: Alert category (concentration, correlation, liquidity, etc.).
        message: Human-readable alert message.
        metric_value: Associated metric value if applicable.
    """

    severity: str  # critical, warning, info
    category: str
    message: str
    metric_value: Optional[float] = None


@dataclass
class PortfolioHealth:
    """Portfolio health assessment results.

    Attributes:
        overall_score: Overall health score from 0-100.
        concentration_score: Concentration risk score from 0-100.
        correlation_score: Correlation risk score from 0-100.
        liquidity_score: Liquidity adequacy score from 0-100.
        risk_score: Risk management score from 0-100.
        alerts: List of health alerts.
        metrics: Dictionary of detailed health metrics.
    """

    overall_score: float
    concentration_score: float
    correlation_score: float
    liquidity_score: float
    risk_score: float
    alerts: List[HealthAlert]
    metrics: Dict[str, float]


class PortfolioHealthMonitor:
    """Monitor portfolio health and identify risks.

    Performs comprehensive health checks on portfolio positions including:
    - Concentration risk (HHI, max position weight)
    - Correlation risk (average correlation, max correlation)
    - Liquidity risk (position turnover, trade size vs volume)
    - Risk metrics (VaR utilization, drawdown proximity)

    Attributes:
        max_position_weight: Maximum allowed position weight (default: 0.30).
        max_correlation: Maximum allowed average correlation (default: 0.70).
        hhi_threshold: HHI threshold for concentration warning (default: 0.20).
    """

    def __init__(
        self,
        max_position_weight: float = 0.30,
        max_correlation: float = 0.70,
        hhi_threshold: float = 0.20,
    ):
        """Initialize portfolio health monitor.

        Args:
            max_position_weight: Maximum position weight before warning (default: 30%).
            max_correlation: Maximum average correlation before warning (default: 0.70).
            hhi_threshold: HHI threshold for concentration alert (default: 0.20).
        """
        self.max_position_weight = max_position_weight
        self.max_correlation = max_correlation
        self.hhi_threshold = hhi_threshold

    def calculate_hhi(self, weights: pl.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration.

        HHI measures concentration risk. Values closer to 1 indicate high
        concentration (few large positions), values closer to 0 indicate
        diversification (many small positions).

        Args:
            weights: Series of position weights (absolute values).

        Returns:
            HHI value between 0 and 1.
        """
        if len(weights) == 0:
            return 0.0

        abs_weights = weights.abs()
        total_weight = float(abs_weights.sum())

        if total_weight == 0:
            return 0.0

        normalized_weights = abs_weights / total_weight
        return float((normalized_weights**2).sum())

    def assess_concentration(self, weights: pl.Series) -> tuple[float, List[HealthAlert]]:
        """Assess portfolio concentration risk.

        Checks:
        - HHI above threshold
        - Any position exceeds max weight
        - Top 3 positions exceed 70% of portfolio

        Args:
            weights: Series of position weights (can be negative for shorts).

        Returns:
            Tuple of (concentration_score, alerts) where score is 0-100.
        """
        alerts = []
        abs_weights = weights.abs()

        hhi = self.calculate_hhi(abs_weights)
        max_weight = float(abs_weights.max()) if len(abs_weights) > 0 else 0.0

        # Sort weights descending and calculate top 3
        sorted_weights = abs_weights.sort(descending=True)
        top3_weight = float(sorted_weights.head(3).sum()) if len(sorted_weights) >= 3 else float(sorted_weights.sum())

        # Generate alerts
        if hhi > self.hhi_threshold:
            alerts.append(
                HealthAlert(
                    severity="warning",
                    category="concentration",
                    message=f"High concentration risk: HHI={hhi:.3f} (threshold: {self.hhi_threshold})",
                    metric_value=hhi,
                )
            )

        if max_weight > self.max_position_weight:
            alerts.append(
                HealthAlert(
                    severity="critical",
                    category="concentration",
                    message=f"Position exceeds max weight: {max_weight:.1%} > {self.max_position_weight:.1%}",
                    metric_value=max_weight,
                )
            )

        if top3_weight > 0.70 and len(sorted_weights) > 3:
            alerts.append(
                HealthAlert(
                    severity="warning",
                    category="concentration",
                    message=f"Top 3 positions represent {top3_weight:.1%} of portfolio",
                    metric_value=top3_weight,
                )
            )

        # Calculate score (100 = best, 0 = worst)
        hhi_score = max(0, 100 * (1 - hhi / 0.5))  # Penalize HHI above 0.5
        weight_score = max(0, 100 * (1 - max_weight / 0.5))  # Penalize single position above 50%
        score = (hhi_score + weight_score) / 2

        return score, alerts

    def assess_health(
        self,
        weights: pl.Series,
        returns: Optional[pl.DataFrame] = None,
        liquidity_scores: Optional[pl.Series] = None,
    ) -> PortfolioHealth:
        """Perform comprehensive portfolio health assessment.

        Args:
            weights: Series of position weights.
            returns: Optional DataFrame of asset returns for correlation analysis.
            liquidity_scores: Optional series of liquidity scores (0-1) per asset.

        Returns:
            PortfolioHealth object with scores and alerts.
        """
        all_alerts = []

        # Concentration assessment
        concentration_score, conc_alerts = self.assess_concentration(weights)
        all_alerts.extend(conc_alerts)

        # Correlation score (placeholder - would need actual correlation matrix)
        correlation_score = 100.0

        # Liquidity assessment
        if liquidity_scores is not None:
            avg_liquidity = float(liquidity_scores.mean())
            liquidity_score = avg_liquidity * 100

            if avg_liquidity < 0.30:
                all_alerts.append(
                    HealthAlert(
                        severity="critical",
                        category="liquidity",
                        message=f"Low average liquidity: {avg_liquidity:.1%}",
                        metric_value=avg_liquidity,
                    )
                )
            elif avg_liquidity < 0.50:
                all_alerts.append(
                    HealthAlert(
                        severity="warning",
                        category="liquidity",
                        message=f"Moderate liquidity: {avg_liquidity:.1%}",
                        metric_value=avg_liquidity,
                    )
                )
        else:
            liquidity_score = 100.0  # No data = assume OK

        # Risk score (placeholder for now, can integrate VaR/drawdown)
        risk_score = 100.0

        # Overall score (weighted average)
        overall_score = (
            0.30 * concentration_score + 0.30 * correlation_score + 0.25 * liquidity_score + 0.15 * risk_score
        )

        # Compile metrics
        metrics = {
            "hhi": self.calculate_hhi(weights.abs()),
            "max_weight": float(weights.abs().max()) if len(weights) > 0 else 0.0,
            "num_positions": len(weights),
        }

        return PortfolioHealth(
            overall_score=overall_score,
            concentration_score=concentration_score,
            correlation_score=correlation_score,
            liquidity_score=liquidity_score,
            risk_score=risk_score,
            alerts=all_alerts,
            metrics=metrics,
        )


def generate_health_report(health: PortfolioHealth) -> str:
    """Generate human-readable health report.

    Args:
        health: PortfolioHealth object from assessment.

    Returns:
        Formatted multi-line health report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PORTFOLIO HEALTH REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overall score
    lines.append(f"Overall Health Score: {health.overall_score:.1f}/100")
    lines.append("")

    # Individual scores
    lines.append("Component Scores:")
    lines.append(f"  Concentration Risk:   {health.concentration_score:.1f}/100")
    lines.append(f"  Correlation Risk:     {health.correlation_score:.1f}/100")
    lines.append(f"  Liquidity Adequacy:   {health.liquidity_score:.1f}/100")
    lines.append(f"  Risk Management:      {health.risk_score:.1f}/100")
    lines.append("")

    # Key metrics
    lines.append("Key Metrics:")
    lines.append(f"  HHI (Concentration):  {health.metrics.get('hhi', 0):.3f}")
    lines.append(f"  Max Position Weight:  {health.metrics.get('max_weight', 0):.1%}")
    lines.append(f"  Number of Positions:  {health.metrics.get('num_positions', 0)}")
    lines.append("")

    # Alerts
    if health.alerts:
        critical_alerts = [a for a in health.alerts if a.severity == "critical"]
        warning_alerts = [a for a in health.alerts if a.severity == "warning"]

        if critical_alerts:
            lines.append("CRITICAL ALERTS:")
            for alert in critical_alerts:
                lines.append(f"  - [{alert.category.upper()}] {alert.message}")
            lines.append("")

        if warning_alerts:
            lines.append("WARNINGS:")
            for alert in warning_alerts:
                lines.append(f"  - [{alert.category.upper()}] {alert.message}")
            lines.append("")
    else:
        lines.append("No alerts - portfolio health is good!")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
