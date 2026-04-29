"""Real-time portfolio performance tracking and monitoring.

Provides live tracking of portfolio metrics with alerts and thresholds.
Useful for paper trading and production monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()


class PerformanceTracker:
    """Track portfolio performance metrics in real-time with alert thresholds."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize the performance tracker.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            alert_thresholds: Dict of metric names to threshold values
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "max_drawdown": -0.15,  # Alert if DD exceeds 15%
            "min_sharpe": 0.5,  # Alert if Sharpe drops below 0.5
            "max_daily_loss": -0.05,  # Alert if daily loss exceeds 5%
        }

        # Historical tracking
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = [datetime.now(timezone.utc)]
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []

        logger.info("Performance tracker initialized",
                   initial_capital=initial_capital,
                   thresholds=self.alert_thresholds)

    def update_position(
        self,
        new_capital: float,
        timestamp: Optional[datetime] = None,
        trade_details: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Update current position and recalculate metrics.

        Args:
            new_capital: Updated portfolio value
            timestamp: Timestamp of update (default: now)
            trade_details: Optional trade information for tracking

        Returns:
            Dict with current metrics and any triggered alerts
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Calculate return since last update
        prev_capital = self.current_capital
        period_return = (new_capital - prev_capital) / prev_capital

        # Update state
        self.current_capital = new_capital
        self.equity_curve.append(new_capital)
        self.timestamps.append(timestamp)
        self.daily_returns.append(period_return)

        if trade_details:
            self.trades.append({
                **trade_details,
                "timestamp": timestamp,
                "capital_after": new_capital,
            })

        # Calculate current metrics
        metrics = self.get_current_metrics()

        # Check for alert triggers
        alerts = self._check_alerts(metrics, period_return)
        if alerts:
            metrics["alerts"] = alerts
            logger.warning("Performance alerts triggered",
                          alerts=alerts,
                          metrics=metrics)

        return metrics

    def get_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics.

        Returns:
            Dict with all current performance metrics
        """
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "current_capital": self.current_capital,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
            }

        # Total return
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # Maximum drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (annualized)
        returns_series = pd.Series(self.daily_returns)
        if len(returns_series) > 1 and returns_series.std() > 0:
            excess_returns = returns_series.mean() - (self.risk_free_rate / 252)
            sharpe = (excess_returns / returns_series.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Win rate from trades
        win_rate = 0.0
        if self.trades:
            profitable_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
            win_rate = profitable_trades / len(self.trades)

        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino = (returns_series.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0.0

        # Volatility (annualized)
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0

        # Calmar ratio (return / max drawdown)
        annual_return = total_return * (252 / len(self.daily_returns)) if self.daily_returns else 0
        calmar = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0

        return {
            "total_return": float(total_return),
            "current_capital": float(self.current_capital),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "total_trades": len(self.trades),
            "days_active": len(self.daily_returns),
        }

    def _check_alerts(
        self,
        metrics: Dict[str, float],
        period_return: float,
    ) -> List[str]:
        """Check if any alert thresholds are breached.

        Args:
            metrics: Current performance metrics
            period_return: Most recent period return

        Returns:
            List of triggered alert messages
        """
        alerts = []

        # Check drawdown threshold
        if metrics["max_drawdown"] < self.alert_thresholds["max_drawdown"]:
            alerts.append(
                f"Max drawdown {metrics['max_drawdown']:.2%} exceeds threshold "
                f"{self.alert_thresholds['max_drawdown']:.2%}"
            )

        # Check Sharpe threshold
        if (metrics["sharpe_ratio"] < self.alert_thresholds["min_sharpe"]
            and len(self.daily_returns) > 20):  # Only check after 20 periods
            alerts.append(
                f"Sharpe ratio {metrics['sharpe_ratio']:.2f} below threshold "
                f"{self.alert_thresholds['min_sharpe']:.2f}"
            )

        # Check daily loss threshold
        if period_return < self.alert_thresholds["max_daily_loss"]:
            alerts.append(
                f"Period loss {period_return:.2%} exceeds threshold "
                f"{self.alert_thresholds['max_daily_loss']:.2%}"
            )

        return alerts

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame.

        Returns:
            DataFrame with timestamps and portfolio values
        """
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "portfolio_value": self.equity_curve,
        })

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame.

        Returns:
            DataFrame with all recorded trades
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def reset(self, initial_capital: Optional[float] = None):
        """Reset tracker to initial state.

        Args:
            initial_capital: New initial capital (default: keep existing)
        """
        if initial_capital:
            self.initial_capital = initial_capital

        self.current_capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.timestamps = [datetime.now(timezone.utc)]
        self.trades = []
        self.daily_returns = []

        logger.info("Performance tracker reset", initial_capital=self.initial_capital)

    def summary_report(self) -> str:
        """Generate a formatted summary report.

        Returns:
            Multi-line string with performance summary
        """
        metrics = self.get_current_metrics()

        report = f"""
        ═══════════════════════════════════════════════════
                    PORTFOLIO PERFORMANCE SUMMARY
        ═══════════════════════════════════════════════════

        Capital:
        • Initial:          ${self.initial_capital:,.2f}
        • Current:          ${self.current_capital:,.2f}
        • Total Return:     {metrics['total_return']:.2%}

        Risk Metrics:
        • Max Drawdown:     {metrics['max_drawdown']:.2%}
        • Volatility:       {metrics['volatility']:.2%}
        • Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}
        • Sortino Ratio:    {metrics['sortino_ratio']:.3f}
        • Calmar Ratio:     {metrics['calmar_ratio']:.3f}

        Trading:
        • Total Trades:     {metrics['total_trades']}
        • Win Rate:         {metrics['win_rate']:.2%}
        • Days Active:      {metrics['days_active']}

        ═══════════════════════════════════════════════════
        """

        return report
