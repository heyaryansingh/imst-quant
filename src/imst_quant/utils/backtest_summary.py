"""Backtesting summary report generation and visualization.

This module provides utilities to generate comprehensive backtest summary
reports including performance metrics, equity curves, drawdown analysis,
and trade statistics.

Functions:
    generate_summary_report: Create comprehensive backtest summary
    calculate_trade_stats: Calculate trade-level statistics
    format_report_table: Format metrics as ASCII table
    export_report_json: Export report as JSON
    export_report_html: Export report as HTML (future)

Example:
    >>> from imst_quant.utils.backtest_summary import generate_summary_report
    >>> report = generate_summary_report(returns, trades, initial_capital=100000)
    >>> print(report.to_string())
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl

from imst_quant.utils.risk_metrics import calculate_all_metrics


@dataclass
class TradeStats:
    """Trade-level statistics for backtesting.

    Attributes:
        total_trades: Total number of trades executed.
        winning_trades: Number of profitable trades.
        losing_trades: Number of losing trades.
        win_rate: Percentage of winning trades.
        avg_win: Average profit on winning trades.
        avg_loss: Average loss on losing trades.
        avg_win_loss_ratio: Ratio of avg win to avg loss.
        largest_win: Largest single trade profit.
        largest_loss: Largest single trade loss.
        avg_trade: Average profit/loss per trade.
        avg_holding_period: Average number of periods per trade.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    avg_holding_period: float = 0.0


@dataclass
class BacktestReport:
    """Comprehensive backtest summary report.

    Attributes:
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        final_capital: Ending capital.
        total_return: Total portfolio return.
        cagr: Compound Annual Growth Rate.
        risk_metrics: Dictionary of risk metrics (Sharpe, Sortino, etc.).
        trade_stats: Trade-level statistics.
        monthly_returns: Monthly return series.
        equity_curve: DataFrame with equity values over time.
    """

    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    cagr: float
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    trade_stats: Optional[TradeStats] = None
    monthly_returns: Optional[pl.DataFrame] = None
    equity_curve: Optional[pl.DataFrame] = None

    def to_string(self) -> str:
        """Format report as human-readable string.

        Returns:
            Formatted report string with sections for overview,
            risk metrics, and trade statistics.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST SUMMARY REPORT".center(70))
        lines.append("=" * 70)
        lines.append("")

        # Overview section
        lines.append("OVERVIEW")
        lines.append("-" * 70)
        lines.append(f"Period:                {self.start_date} to {self.end_date}")
        lines.append(f"Initial Capital:       ${self.initial_capital:,.2f}")
        lines.append(f"Final Capital:         ${self.final_capital:,.2f}")
        lines.append(f"Total Return:          {self.total_return:.2%}")
        lines.append(f"CAGR:                  {self.cagr:.2%}")
        lines.append("")

        # Risk metrics section
        lines.append("RISK METRICS")
        lines.append("-" * 70)
        for metric, value in self.risk_metrics.items():
            label = metric.replace("_", " ").title()
            if "return" in metric or "drawdown" in metric or "var" in metric:
                lines.append(f"{label:<25} {value:.2%}")
            else:
                lines.append(f"{label:<25} {value:.4f}")
        lines.append("")

        # Trade statistics section
        if self.trade_stats:
            ts = self.trade_stats
            lines.append("TRADE STATISTICS")
            lines.append("-" * 70)
            lines.append(f"Total Trades:          {ts.total_trades}")
            lines.append(f"Winning Trades:        {ts.winning_trades}")
            lines.append(f"Losing Trades:         {ts.losing_trades}")
            lines.append(f"Win Rate:              {ts.win_rate:.2%}")
            lines.append(f"Average Win:           ${ts.avg_win:,.2f}")
            lines.append(f"Average Loss:          ${ts.avg_loss:,.2f}")
            lines.append(f"Avg Win/Loss Ratio:    {ts.avg_win_loss_ratio:.2f}")
            lines.append(f"Largest Win:           ${ts.largest_win:,.2f}")
            lines.append(f"Largest Loss:          ${ts.largest_loss:,.2f}")
            lines.append(f"Average Trade:         ${ts.avg_trade:,.2f}")
            lines.append(f"Avg Holding Period:    {ts.avg_holding_period:.1f} days")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON export.

        Returns:
            Dictionary representation of the report.
        """
        report_dict = {
            "overview": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": self.initial_capital,
                "final_capital": self.final_capital,
                "total_return": self.total_return,
                "cagr": self.cagr,
            },
            "risk_metrics": self.risk_metrics,
        }

        if self.trade_stats:
            report_dict["trade_stats"] = {
                "total_trades": self.trade_stats.total_trades,
                "winning_trades": self.trade_stats.winning_trades,
                "losing_trades": self.trade_stats.losing_trades,
                "win_rate": self.trade_stats.win_rate,
                "avg_win": self.trade_stats.avg_win,
                "avg_loss": self.trade_stats.avg_loss,
                "avg_win_loss_ratio": self.trade_stats.avg_win_loss_ratio,
                "largest_win": self.trade_stats.largest_win,
                "largest_loss": self.trade_stats.largest_loss,
                "avg_trade": self.trade_stats.avg_trade,
                "avg_holding_period": self.trade_stats.avg_holding_period,
            }

        return report_dict


def calculate_trade_stats(
    trades: pl.DataFrame,
    pnl_col: str = "pnl",
    holding_period_col: str = "holding_period",
) -> TradeStats:
    """Calculate trade-level statistics from trades DataFrame.

    Args:
        trades: DataFrame containing individual trades with PnL.
        pnl_col: Column name for profit/loss (default: "pnl").
        holding_period_col: Column name for holding period (default: "holding_period").

    Returns:
        TradeStats object with calculated statistics.

    Example:
        >>> trades = pl.DataFrame({
        ...     "pnl": [100, -50, 200, -30, 150],
        ...     "holding_period": [5, 3, 7, 2, 6],
        ... })
        >>> stats = calculate_trade_stats(trades)
        >>> print(f"Win rate: {stats.win_rate:.2%}")
    """
    if trades.is_empty():
        return TradeStats()

    total_trades = len(trades)
    winning_trades_df = trades.filter(pl.col(pnl_col) > 0)
    losing_trades_df = trades.filter(pl.col(pnl_col) < 0)

    winning_trades = len(winning_trades_df)
    losing_trades = len(losing_trades_df)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades_df[pnl_col].mean() if winning_trades > 0 else 0.0
    avg_loss = abs(losing_trades_df[pnl_col].mean()) if losing_trades > 0 else 0.0

    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    largest_win = trades[pnl_col].max() or 0.0
    largest_loss = abs(trades[pnl_col].min() or 0.0)
    avg_trade = trades[pnl_col].mean() or 0.0

    avg_holding_period = 0.0
    if holding_period_col in trades.columns:
        avg_holding_period = trades[holding_period_col].mean() or 0.0

    return TradeStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        avg_win_loss_ratio=avg_win_loss_ratio,
        largest_win=float(largest_win),
        largest_loss=float(largest_loss),
        avg_trade=float(avg_trade),
        avg_holding_period=avg_holding_period,
    )


def generate_summary_report(
    returns: Union[pl.Series, pl.DataFrame],
    initial_capital: float = 100000.0,
    trades: Optional[pl.DataFrame] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    return_col: str = "returns",
) -> BacktestReport:
    """Generate comprehensive backtest summary report.

    Args:
        returns: Series or DataFrame containing daily returns.
        initial_capital: Starting capital amount (default: 100000).
        trades: Optional DataFrame with individual trades for trade stats.
        start_date: Backtest start date (default: first date in returns).
        end_date: Backtest end date (default: last date in returns).
        risk_free_rate: Daily risk-free rate for Sharpe/Sortino.
        periods_per_year: Trading periods per year (default: 252).
        return_col: Column name if returns is DataFrame.

    Returns:
        BacktestReport with all calculated metrics.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, 0.005, -0.01])
        >>> report = generate_summary_report(returns, initial_capital=100000)
        >>> print(report.to_string())
    """
    if isinstance(returns, pl.DataFrame):
        ret_series = returns[return_col]
        if "date" in returns.columns:
            start_date = start_date or str(returns["date"].min())
            end_date = end_date or str(returns["date"].max())
    else:
        ret_series = returns

    # Default dates if not provided
    start_date = start_date or "N/A"
    end_date = end_date or "N/A"

    # Calculate risk metrics
    risk_metrics = calculate_all_metrics(
        ret_series,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        var_confidence=0.95,
    )

    # Calculate final capital and total return
    total_return = risk_metrics["total_return"]
    final_capital = initial_capital * (1 + total_return)

    # Calculate CAGR
    n_periods = len(ret_series)
    years = n_periods / periods_per_year if periods_per_year > 0 else 1
    cagr = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    # Calculate trade statistics if trades provided
    trade_stats = None
    if trades is not None and not trades.is_empty():
        trade_stats = calculate_trade_stats(trades)

    # Generate equity curve
    equity_curve = None
    if isinstance(returns, pl.DataFrame):
        equity_values = initial_capital * (1 + ret_series).cum_prod()
        equity_curve = returns.with_columns(equity_values.alias("equity"))

    return BacktestReport(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        cagr=cagr,
        risk_metrics=risk_metrics,
        trade_stats=trade_stats,
        equity_curve=equity_curve,
    )
