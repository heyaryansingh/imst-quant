"""Portfolio Risk Dashboard - Comprehensive real-time risk monitoring and visualization.

Provides unified risk metrics dashboard combining VaR, drawdowns, correlations,
concentration, and tail risk for real-time portfolio monitoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_to_benchmark: float
    concentration_hhi: float
    tail_ratio: float
    downside_deviation: float
    ulcer_index: float


@dataclass
class PositionRisk:
    """Risk metrics for individual position."""

    symbol: str
    weight: float
    contribution_to_var: float
    marginal_var: float
    beta: float
    volatility: float
    max_drawdown: float
    liquidity_score: float


class RiskDashboard:
    """Comprehensive portfolio risk monitoring dashboard."""

    def __init__(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ):
        """Initialize risk dashboard.

        Args:
            returns: Portfolio returns time series
            positions: DataFrame with columns [symbol, weight, returns]
            benchmark_returns: Benchmark returns for beta calculation
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.positions = positions
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    def calculate_var(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk using historical simulation.

        Args:
            confidence: Confidence level (0.95 or 0.99)

        Returns:
            Tuple of (VaR, CVaR)
        """
        var = np.percentile(self.returns, (1 - confidence) * 100)
        cvar = self.returns[self.returns <= var].mean()
        return var, cvar

    def calculate_drawdown(self) -> Tuple[float, float, pd.Series]:
        """Calculate maximum and current drawdown.

        Returns:
            Tuple of (max_drawdown, current_drawdown, drawdown_series)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]

        return max_dd, current_dd, drawdown

    def calculate_performance_ratios(self) -> Dict[str, float]:
        """Calculate Sharpe, Sortino, and other performance ratios.

        Returns:
            Dictionary of performance metrics
        """
        excess_returns = self.returns - self.risk_free_rate / 252
        volatility = self.returns.std() * np.sqrt(252)

        sharpe = (excess_returns.mean() * 252) / volatility if volatility > 0 else 0

        downside_returns = self.returns[self.returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)
        sortino = (excess_returns.mean() * 252) / downside_dev if downside_dev > 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'volatility_annual': volatility,
            'downside_deviation': downside_dev
        }

    def calculate_beta(self) -> float:
        """Calculate portfolio beta vs benchmark.

        Returns:
            Beta coefficient
        """
        if self.benchmark_returns is None:
            return 1.0

        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)

        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

    def calculate_concentration(self) -> float:
        """Calculate portfolio concentration using HHI.

        Returns:
            Herfindahl-Hirschman Index (0-1)
        """
        weights = self.positions['weight'].values
        hhi = np.sum(weights ** 2)
        return hhi

    def calculate_tail_ratio(self) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile).

        Returns:
            Tail ratio
        """
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)

        return abs(p95 / p5) if p5 != 0 else 0

    def calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index - measure of downside volatility.

        Returns:
            Ulcer Index
        """
        _, _, drawdown = self.calculate_drawdown()
        ulcer = np.sqrt(np.mean(drawdown ** 2))
        return ulcer

    def calculate_position_risk(self) -> List[PositionRisk]:
        """Calculate risk metrics for each position.

        Returns:
            List of PositionRisk objects
        """
        position_risks = []

        for _, pos in self.positions.iterrows():
            symbol = pos['symbol']
            weight = pos['weight']
            pos_returns = pos['returns'] if 'returns' in pos else None

            if pos_returns is None or len(pos_returns) == 0:
                continue

            # Position-specific metrics
            pos_vol = pos_returns.std() * np.sqrt(252)
            pos_var = np.percentile(pos_returns, 5)

            # Marginal VaR contribution
            portfolio_var, _ = self.calculate_var(0.95)
            marginal_var = (pos_var - portfolio_var) / weight if weight > 0 else 0

            # Beta if benchmark available
            pos_beta = 1.0
            if self.benchmark_returns is not None:
                cov = np.cov(pos_returns, self.benchmark_returns)[0, 1]
                bench_var = np.var(self.benchmark_returns)
                pos_beta = cov / bench_var if bench_var > 0 else 1.0

            # Drawdown
            pos_cumulative = (1 + pos_returns).cumprod()
            pos_running_max = pos_cumulative.expanding().max()
            pos_dd = ((pos_cumulative - pos_running_max) / pos_running_max).min()

            # Simple liquidity score (can be enhanced with volume data)
            liquidity_score = 1.0 - abs(pos_vol - 0.2)  # Normalized around 20% vol

            position_risks.append(PositionRisk(
                symbol=symbol,
                weight=weight,
                contribution_to_var=marginal_var * weight,
                marginal_var=marginal_var,
                beta=pos_beta,
                volatility=pos_vol,
                max_drawdown=pos_dd,
                liquidity_score=max(0, min(1, liquidity_score))
            ))

        return position_risks

    def generate_dashboard(self) -> RiskMetrics:
        """Generate complete risk dashboard.

        Returns:
            RiskMetrics object with all metrics
        """
        var_95, cvar_95 = self.calculate_var(0.95)
        var_99, cvar_99 = self.calculate_var(0.99)
        max_dd, current_dd, _ = self.calculate_drawdown()

        perf_ratios = self.calculate_performance_ratios()
        beta = self.calculate_beta()

        correlation = 0.0
        if self.benchmark_returns is not None:
            correlation = np.corrcoef(self.returns, self.benchmark_returns)[0, 1]

        concentration = self.calculate_concentration()
        tail_ratio = self.calculate_tail_ratio()
        ulcer = self.calculate_ulcer_index()

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            volatility_annual=perf_ratios['volatility_annual'],
            sharpe_ratio=perf_ratios['sharpe_ratio'],
            sortino_ratio=perf_ratios['sortino_ratio'],
            beta=beta,
            correlation_to_benchmark=correlation,
            concentration_hhi=concentration,
            tail_ratio=tail_ratio,
            downside_deviation=perf_ratios['downside_deviation'],
            ulcer_index=ulcer
        )

    def get_risk_alerts(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk alerts based on thresholds.

        Args:
            metrics: RiskMetrics object

        Returns:
            List of alert messages
        """
        alerts = []

        if metrics.var_95 < -0.03:
            alerts.append(f"HIGH RISK: 95% VaR exceeds 3% ({metrics.var_95:.2%})")

        if metrics.current_drawdown < -0.15:
            alerts.append(f"HIGH DRAWDOWN: Current drawdown {metrics.current_drawdown:.2%}")

        if metrics.concentration_hhi > 0.5:
            alerts.append(f"HIGH CONCENTRATION: HHI {metrics.concentration_hhi:.2f}")

        if metrics.sharpe_ratio < 0.5:
            alerts.append(f"LOW SHARPE: {metrics.sharpe_ratio:.2f}")

        if metrics.volatility_annual > 0.40:
            alerts.append(f"HIGH VOLATILITY: {metrics.volatility_annual:.2%}")

        if metrics.tail_ratio > 3.0:
            alerts.append(f"FAT TAILS: Tail ratio {metrics.tail_ratio:.2f}")

        return alerts

    def export_summary(self) -> pd.DataFrame:
        """Export risk dashboard as DataFrame for reporting.

        Returns:
            DataFrame with risk metrics
        """
        metrics = self.generate_dashboard()
        position_risks = self.calculate_position_risk()

        summary_data = {
            'Metric': [
                'VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%',
                'Max Drawdown', 'Current Drawdown',
                'Annual Volatility', 'Downside Deviation',
                'Sharpe Ratio', 'Sortino Ratio',
                'Beta', 'Correlation to Benchmark',
                'Concentration (HHI)', 'Tail Ratio', 'Ulcer Index'
            ],
            'Value': [
                f"{metrics.var_95:.2%}", f"{metrics.var_99:.2%}",
                f"{metrics.cvar_95:.2%}", f"{metrics.cvar_99:.2%}",
                f"{metrics.max_drawdown:.2%}", f"{metrics.current_drawdown:.2%}",
                f"{metrics.volatility_annual:.2%}", f"{metrics.downside_deviation:.2%}",
                f"{metrics.sharpe_ratio:.2f}", f"{metrics.sortino_ratio:.2f}",
                f"{metrics.beta:.2f}", f"{metrics.correlation_to_benchmark:.2f}",
                f"{metrics.concentration_hhi:.2f}", f"{metrics.tail_ratio:.2f}",
                f"{metrics.ulcer_index:.2f}"
            ]
        }

        return pd.DataFrame(summary_data)


def create_risk_heatmap(
    positions: List[PositionRisk],
    metrics: List[str] = None
) -> pd.DataFrame:
    """Create risk heatmap for positions.

    Args:
        positions: List of PositionRisk objects
        metrics: List of metric names to include

    Returns:
        DataFrame suitable for heatmap visualization
    """
    if metrics is None:
        metrics = ['weight', 'volatility', 'beta', 'max_drawdown', 'contribution_to_var']

    data = []
    for pos in positions:
        row = {'symbol': pos.symbol}
        if 'weight' in metrics:
            row['weight'] = pos.weight
        if 'volatility' in metrics:
            row['volatility'] = pos.volatility
        if 'beta' in metrics:
            row['beta'] = pos.beta
        if 'max_drawdown' in metrics:
            row['max_drawdown'] = pos.max_drawdown
        if 'contribution_to_var' in metrics:
            row['contribution_to_var'] = pos.contribution_to_var
        if 'liquidity_score' in metrics:
            row['liquidity_score'] = pos.liquidity_score

        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index('symbol')

    return df
