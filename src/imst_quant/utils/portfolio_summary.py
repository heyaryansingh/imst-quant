"""Portfolio summary utilities for comprehensive position and performance analysis.

This module provides utilities for generating comprehensive portfolio summaries including:
- Current position details with P&L
- Risk metrics (VaR, CVaR, Beta, Sharpe)
- Sector/asset allocation breakdown
- Recent performance statistics

Example:
    >>> from imst_quant.utils.portfolio_summary import PortfolioSummary
    >>> summary = PortfolioSummary(portfolio_data)
    >>> report = summary.generate_summary()
    >>> summary.display_console()
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class PositionMetrics:
    """Metrics for a single portfolio position.

    Attributes:
        symbol: Ticker symbol
        quantity: Number of shares/units
        entry_price: Average entry price
        current_price: Current market price
        market_value: Current position value
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L percentage
        weight: Portfolio weight (0-1)
        sector: Asset sector/category
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    sector: Optional[str] = None


@dataclass
class PortfolioMetrics:
    """Overall portfolio performance and risk metrics.

    Attributes:
        total_value: Total portfolio value
        cash: Available cash
        invested: Total invested capital
        total_pnl: Total P&L
        total_pnl_pct: Total P&L percentage
        daily_return: Return for today
        weekly_return: Return for past 7 days
        monthly_return: Return for past 30 days
        sharpe_ratio: Sharpe ratio (annualized)
        max_drawdown: Maximum drawdown
        var_95: Value at Risk (95% confidence)
        cvar_95: Conditional VaR (95%)
        beta: Portfolio beta vs benchmark
        win_rate: Percentage of winning positions
    """
    total_value: float
    cash: float
    invested: float
    total_pnl: float
    total_pnl_pct: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    win_rate: float


class PortfolioSummary:
    """Generate comprehensive portfolio summaries and reports.

    Args:
        positions: List of current positions with entry/current prices
        historical_values: Time series of portfolio values
        benchmark_returns: Optional benchmark returns for beta calculation
        risk_free_rate: Annual risk-free rate (default: 0.03)

    Example:
        >>> positions = [
        ...     {'symbol': 'AAPL', 'quantity': 100, 'entry': 150, 'current': 160},
        ...     {'symbol': 'GOOGL', 'quantity': 50, 'entry': 2800, 'current': 2900}
        ... ]
        >>> history = pd.Series([100000, 102000, 105000, 103000],
        ...                      index=pd.date_range('2024-01-01', periods=4))
        >>> summary = PortfolioSummary(positions, history)
        >>> report = summary.generate_summary()
    """

    def __init__(
        self,
        positions: List[Dict[str, Any]],
        historical_values: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.03
    ):
        self.positions = positions
        self.historical_values = historical_values
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    def calculate_position_metrics(self) -> List[PositionMetrics]:
        """Calculate metrics for each position.

        Returns:
            List of PositionMetrics objects
        """
        total_value = sum(
            pos['quantity'] * pos['current']
            for pos in self.positions
        )

        metrics = []
        for pos in self.positions:
            market_value = pos['quantity'] * pos['current']
            cost_basis = pos['quantity'] * pos['entry']
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis else 0
            weight = market_value / total_value if total_value else 0

            metrics.append(PositionMetrics(
                symbol=pos['symbol'],
                quantity=pos['quantity'],
                entry_price=pos['entry'],
                current_price=pos['current'],
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                weight=weight,
                sector=pos.get('sector')
            ))

        return metrics

    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate overall portfolio metrics.

        Returns:
            PortfolioMetrics object with comprehensive stats
        """
        # Calculate returns
        returns = self.historical_values.pct_change().dropna()

        # Time-based returns
        daily_return = returns.iloc[-1] if len(returns) > 0 else 0
        weekly_return = (
            (self.historical_values.iloc[-1] / self.historical_values.iloc[-5] - 1)
            if len(self.historical_values) >= 5 else 0
        )
        monthly_return = (
            (self.historical_values.iloc[-1] / self.historical_values.iloc[-20] - 1)
            if len(self.historical_values) >= 20 else 0
        )

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown()
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0

        # Beta calculation
        beta = 1.0
        if self.benchmark_returns is not None and len(returns) > 1:
            beta = self._calculate_beta(returns)

        # Position stats
        position_metrics = self.calculate_position_metrics()
        total_value = sum(pm.market_value for pm in position_metrics)
        total_cost = sum(pm.quantity * pm.entry_price for pm in position_metrics)
        total_pnl = sum(pm.unrealized_pnl for pm in position_metrics)
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

        winning_positions = sum(1 for pm in position_metrics if pm.unrealized_pnl > 0)
        win_rate = (winning_positions / len(position_metrics) * 100) if position_metrics else 0

        return PortfolioMetrics(
            total_value=total_value,
            cash=0,  # Would come from portfolio data
            invested=total_cost,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_return=daily_return * 100,
            weekly_return=weekly_return * 100,
            monthly_return=monthly_return * 100,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,
            var_95=var_95 * 100,
            cvar_95=cvar_95 * 100,
            beta=beta,
            win_rate=win_rate
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (self.risk_free_rate / 252)
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.historical_values) < 2:
            return 0.0
        cummax = self.historical_values.cummax()
        drawdown = (self.historical_values - cummax) / cummax
        return drawdown.min()

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate portfolio beta vs benchmark."""
        aligned_returns = returns.align(self.benchmark_returns, join='inner')
        if len(aligned_returns[0]) < 2:
            return 1.0
        covariance = np.cov(aligned_returns[0], aligned_returns[1])[0, 1]
        benchmark_variance = np.var(aligned_returns[1])
        return covariance / benchmark_variance if benchmark_variance else 1.0

    def generate_summary(self) -> Dict[str, Any]:
        """Generate complete portfolio summary.

        Returns:
            Dictionary with positions and portfolio metrics
        """
        position_metrics = self.calculate_position_metrics()
        portfolio_metrics = self.calculate_portfolio_metrics()

        return {
            'positions': [
                {
                    'symbol': pm.symbol,
                    'quantity': pm.quantity,
                    'entry_price': pm.entry_price,
                    'current_price': pm.current_price,
                    'market_value': pm.market_value,
                    'unrealized_pnl': pm.unrealized_pnl,
                    'unrealized_pnl_pct': pm.unrealized_pnl_pct,
                    'weight': pm.weight,
                    'sector': pm.sector
                }
                for pm in position_metrics
            ],
            'portfolio': {
                'total_value': portfolio_metrics.total_value,
                'cash': portfolio_metrics.cash,
                'invested': portfolio_metrics.invested,
                'total_pnl': portfolio_metrics.total_pnl,
                'total_pnl_pct': portfolio_metrics.total_pnl_pct,
                'daily_return': portfolio_metrics.daily_return,
                'weekly_return': portfolio_metrics.weekly_return,
                'monthly_return': portfolio_metrics.monthly_return,
                'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                'max_drawdown': portfolio_metrics.max_drawdown,
                'var_95': portfolio_metrics.var_95,
                'cvar_95': portfolio_metrics.cvar_95,
                'beta': portfolio_metrics.beta,
                'win_rate': portfolio_metrics.win_rate
            },
            'timestamp': datetime.now().isoformat()
        }

    def display_console(self):
        """Display portfolio summary in console-friendly format."""
        summary = self.generate_summary()

        print("\n" + "=" * 80)
        print("PORTFOLIO SUMMARY".center(80))
        print("=" * 80)

        # Portfolio overview
        portfolio = summary['portfolio']
        print(f"\nTotal Value: ${portfolio['total_value']:,.2f}")
        print(f"Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2f}%)")
        print(f"Win Rate: {portfolio['win_rate']:.1f}%")

        # Returns
        print(f"\nReturns:")
        print(f"  Daily:   {portfolio['daily_return']:+.2f}%")
        print(f"  Weekly:  {portfolio['weekly_return']:+.2f}%")
        print(f"  Monthly: {portfolio['monthly_return']:+.2f}%")

        # Risk metrics
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:    {portfolio['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:    {portfolio['max_drawdown']:.2f}%")
        print(f"  VaR (95%):       {portfolio['var_95']:.2f}%")
        print(f"  CVaR (95%):      {portfolio['cvar_95']:.2f}%")
        print(f"  Beta:            {portfolio['beta']:.2f}")

        # Positions
        print(f"\nPositions ({len(summary['positions'])}):")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'Weight':>8}")
        print("-" * 80)

        for pos in summary['positions']:
            pnl_str = f"${pos['unrealized_pnl']:,.0f}"
            pnl_pct = f"({pos['unrealized_pnl_pct']:+.1f}%)"
            print(
                f"{pos['symbol']:<10} {pos['quantity']:>8.0f} "
                f"${pos['entry_price']:>9.2f} ${pos['current_price']:>9.2f} "
                f"{pnl_str:>7} {pnl_pct:>6} {pos['weight']*100:>6.1f}%"
            )

        print("=" * 80 + "\n")


def get_sector_allocation(positions: List[PositionMetrics]) -> Dict[str, float]:
    """Calculate portfolio allocation by sector.

    Args:
        positions: List of PositionMetrics objects

    Returns:
        Dictionary mapping sector to allocation percentage
    """
    sector_values = {}
    total_value = sum(p.market_value for p in positions)

    for pos in positions:
        sector = pos.sector or 'Unknown'
        sector_values[sector] = sector_values.get(sector, 0) + pos.market_value

    return {
        sector: (value / total_value * 100) if total_value else 0
        for sector, value in sector_values.items()
    }
