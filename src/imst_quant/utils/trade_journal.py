"""Trade journal for logging, reviewing, and analyzing trading decisions.

This module provides utilities for maintaining a structured trade journal that
helps traders track their entries/exits, analyze performance by various
dimensions, and identify patterns in their trading behavior.

Example:
    >>> from imst_quant.utils.trade_journal import TradeJournal, TradeEntry
    >>> journal = TradeJournal()
    >>> entry = TradeEntry(
    ...     symbol="AAPL",
    ...     direction="long",
    ...     entry_price=150.0,
    ...     quantity=100,
    ...     entry_reason="Bullish breakout above 20-day high",
    ... )
    >>> journal.log_entry(entry)
    >>> journal.close_trade(entry.trade_id, exit_price=155.0, exit_reason="Target hit")
    >>> stats = journal.get_statistics()
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class TradeEntry:
    """Represents a single trade entry in the journal.

    Attributes:
        trade_id: Unique identifier for the trade.
        symbol: Trading symbol/ticker.
        direction: Trade direction ('long' or 'short').
        entry_price: Price at which position was entered.
        quantity: Number of shares/contracts.
        entry_time: Timestamp of entry.
        entry_reason: Rationale for entering the trade.
        setup_type: Trading setup category (e.g., 'breakout', 'pullback').
        timeframe: Trading timeframe (e.g., 'intraday', 'swing').
        risk_amount: Dollar amount risked on the trade.
        stop_loss: Stop loss price level.
        take_profit: Take profit price level.
        exit_price: Price at which position was closed.
        exit_time: Timestamp of exit.
        exit_reason: Rationale for exiting the trade.
        pnl: Profit/loss in dollars.
        pnl_percent: Profit/loss as percentage of entry.
        r_multiple: Profit/loss expressed as multiple of risk.
        notes: Additional notes or observations.
        tags: List of tags for categorization.
        emotions: Trader's emotional state during the trade.
        status: Trade status ('open', 'closed', 'cancelled').
    """

    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    quantity: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_reason: str = ""
    setup_type: str = ""
    timeframe: str = ""
    risk_amount: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float | None = None
    pnl_percent: float | None = None
    r_multiple: float | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    emotions: str = ""
    status: Literal["open", "closed", "cancelled"] = "open"
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def calculate_pnl(self) -> float | None:
        """Calculate profit/loss if trade is closed.

        Returns:
            PnL in dollars, or None if trade is not closed.
        """
        if self.exit_price is None:
            return None

        if self.direction == "long":
            pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - self.exit_price) * self.quantity

        return pnl

    def calculate_pnl_percent(self) -> float | None:
        """Calculate PnL as percentage of entry price.

        Returns:
            PnL percentage, or None if trade is not closed.
        """
        if self.exit_price is None:
            return None

        if self.direction == "long":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    def calculate_r_multiple(self) -> float | None:
        """Calculate R-multiple (risk-adjusted return).

        Returns:
            R-multiple, or None if risk_amount is not set or trade not closed.
        """
        if self.risk_amount is None or self.risk_amount == 0:
            return None

        pnl = self.calculate_pnl()
        if pnl is None:
            return None

        return pnl / self.risk_amount

    def to_dict(self) -> dict[str, Any]:
        """Convert trade entry to dictionary for serialization.

        Returns:
            Dictionary representation of the trade.
        """
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.entry_time:
            data["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            data["exit_time"] = self.exit_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeEntry:
        """Create TradeEntry from dictionary.

        Args:
            data: Dictionary with trade entry data.

        Returns:
            TradeEntry instance.
        """
        # Convert ISO format strings back to datetime
        if data.get("entry_time") and isinstance(data["entry_time"], str):
            data["entry_time"] = datetime.fromisoformat(data["entry_time"])
        if data.get("exit_time") and isinstance(data["exit_time"], str):
            data["exit_time"] = datetime.fromisoformat(data["exit_time"])
        return cls(**data)


@dataclass
class JournalStatistics:
    """Aggregated statistics from the trade journal.

    Attributes:
        total_trades: Total number of trades.
        winning_trades: Number of profitable trades.
        losing_trades: Number of unprofitable trades.
        win_rate: Percentage of winning trades.
        total_pnl: Total profit/loss.
        avg_pnl: Average profit/loss per trade.
        avg_winner: Average profit on winning trades.
        avg_loser: Average loss on losing trades.
        profit_factor: Ratio of gross profits to gross losses.
        expectancy: Expected value per trade.
        avg_r_multiple: Average R-multiple across trades.
        largest_win: Largest single winning trade.
        largest_loss: Largest single losing trade.
        avg_holding_time: Average time in trade (hours).
        consecutive_wins: Current or max consecutive wins.
        consecutive_losses: Current or max consecutive losses.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_r_multiple: float | None = None
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


class TradeJournal:
    """Trade journal for logging and analyzing trades.

    Maintains a collection of trade entries and provides methods for
    analysis, statistics, and persistence.

    Example:
        >>> journal = TradeJournal(journal_path="my_trades.json")
        >>> # Log a new trade
        >>> entry = TradeEntry(symbol="AAPL", direction="long", entry_price=150, quantity=100)
        >>> journal.log_entry(entry)
        >>> # Close the trade
        >>> journal.close_trade(entry.trade_id, exit_price=155, exit_reason="Target")
        >>> # Get statistics
        >>> stats = journal.get_statistics()
        >>> print(f"Win rate: {stats.win_rate:.1%}")
    """

    def __init__(self, journal_path: str | Path | None = None):
        """Initialize trade journal.

        Args:
            journal_path: Path to persist journal data. If None, journal is memory-only.
        """
        self.journal_path = Path(journal_path) if journal_path else None
        self.trades: list[TradeEntry] = []

        if self.journal_path and self.journal_path.exists():
            self.load()

    def log_entry(self, trade: TradeEntry) -> str:
        """Log a new trade entry.

        Args:
            trade: TradeEntry to add to the journal.

        Returns:
            Trade ID of the logged entry.
        """
        self.trades.append(trade)
        self._auto_save()
        return trade.trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "",
        exit_time: datetime | None = None,
    ) -> TradeEntry | None:
        """Close an open trade.

        Args:
            trade_id: ID of the trade to close.
            exit_price: Price at which position was closed.
            exit_reason: Rationale for closing the trade.
            exit_time: Timestamp of exit (defaults to now).

        Returns:
            Updated TradeEntry, or None if trade not found.
        """
        trade = self.get_trade(trade_id)
        if trade is None:
            return None

        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.status = "closed"

        # Calculate metrics
        trade.pnl = trade.calculate_pnl()
        trade.pnl_percent = trade.calculate_pnl_percent()
        trade.r_multiple = trade.calculate_r_multiple()

        self._auto_save()
        return trade

    def cancel_trade(self, trade_id: str, reason: str = "") -> TradeEntry | None:
        """Cancel an open trade without execution.

        Args:
            trade_id: ID of the trade to cancel.
            reason: Reason for cancellation.

        Returns:
            Updated TradeEntry, or None if trade not found.
        """
        trade = self.get_trade(trade_id)
        if trade is None:
            return None

        trade.status = "cancelled"
        trade.notes = f"{trade.notes} [Cancelled: {reason}]".strip()
        self._auto_save()
        return trade

    def get_trade(self, trade_id: str) -> TradeEntry | None:
        """Get a trade by ID.

        Args:
            trade_id: Trade identifier.

        Returns:
            TradeEntry if found, None otherwise.
        """
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_open_trades(self) -> list[TradeEntry]:
        """Get all open trades.

        Returns:
            List of open TradeEntry objects.
        """
        return [t for t in self.trades if t.status == "open"]

    def get_closed_trades(self) -> list[TradeEntry]:
        """Get all closed trades.

        Returns:
            List of closed TradeEntry objects.
        """
        return [t for t in self.trades if t.status == "closed"]

    def filter_trades(
        self,
        symbol: str | None = None,
        direction: str | None = None,
        setup_type: str | None = None,
        tags: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_pnl: float | None = None,
        max_pnl: float | None = None,
    ) -> list[TradeEntry]:
        """Filter trades by various criteria.

        Args:
            symbol: Filter by trading symbol.
            direction: Filter by direction ('long' or 'short').
            setup_type: Filter by setup type.
            tags: Filter by tags (any match).
            start_date: Filter trades after this date.
            end_date: Filter trades before this date.
            min_pnl: Filter trades with PnL >= this value.
            max_pnl: Filter trades with PnL <= this value.

        Returns:
            List of matching TradeEntry objects.
        """
        filtered = self.trades.copy()

        if symbol:
            filtered = [t for t in filtered if t.symbol.upper() == symbol.upper()]

        if direction:
            filtered = [t for t in filtered if t.direction == direction]

        if setup_type:
            filtered = [t for t in filtered if t.setup_type == setup_type]

        if tags:
            filtered = [t for t in filtered if any(tag in t.tags for tag in tags)]

        if start_date:
            filtered = [t for t in filtered if t.entry_time >= start_date]

        if end_date:
            filtered = [t for t in filtered if t.entry_time <= end_date]

        if min_pnl is not None:
            filtered = [t for t in filtered if t.pnl is not None and t.pnl >= min_pnl]

        if max_pnl is not None:
            filtered = [t for t in filtered if t.pnl is not None and t.pnl <= max_pnl]

        return filtered

    def get_statistics(
        self, trades: list[TradeEntry] | None = None
    ) -> JournalStatistics:
        """Calculate statistics for trades.

        Args:
            trades: List of trades to analyze. If None, uses all closed trades.

        Returns:
            JournalStatistics with aggregated metrics.
        """
        if trades is None:
            trades = self.get_closed_trades()

        stats = JournalStatistics()

        if not trades:
            return stats

        stats.total_trades = len(trades)

        # Categorize trades
        pnls = [t.pnl for t in trades if t.pnl is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if p == 0]

        stats.winning_trades = len(winners)
        stats.losing_trades = len(losers)
        stats.breakeven_trades = len(breakeven)

        if pnls:
            stats.win_rate = len(winners) / len(pnls)
            stats.total_pnl = sum(pnls)
            stats.avg_pnl = stats.total_pnl / len(pnls)

            if winners:
                stats.avg_winner = sum(winners) / len(winners)
                stats.largest_win = max(winners)

            if losers:
                stats.avg_loser = sum(losers) / len(losers)
                stats.largest_loss = min(losers)

            # Profit factor
            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 0
            if gross_loss > 0:
                stats.profit_factor = gross_profit / gross_loss

            # Expectancy
            if stats.win_rate > 0:
                stats.expectancy = (
                    stats.win_rate * stats.avg_winner
                    + (1 - stats.win_rate) * stats.avg_loser
                )

        # R-multiple stats
        r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
        if r_multiples:
            stats.avg_r_multiple = sum(r_multiples) / len(r_multiples)

        # Holding time
        holding_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                holding_times.append(delta.total_seconds() / 3600)  # hours
        if holding_times:
            stats.avg_holding_time_hours = sum(holding_times) / len(holding_times)

        # Consecutive wins/losses
        if pnls:
            stats.max_consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
            stats.max_consecutive_losses = self._max_consecutive(pnls, lambda x: x < 0)

        return stats

    def _max_consecutive(
        self, values: list[float], condition: callable
    ) -> int:
        """Find maximum consecutive values meeting a condition.

        Args:
            values: List of values to check.
            condition: Function returning True for matching values.

        Returns:
            Maximum consecutive count.
        """
        max_count = 0
        current_count = 0

        for v in values:
            if condition(v):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def get_performance_by_symbol(self) -> dict[str, JournalStatistics]:
        """Get statistics broken down by symbol.

        Returns:
            Dictionary mapping symbol to JournalStatistics.
        """
        by_symbol: dict[str, list[TradeEntry]] = {}
        for trade in self.get_closed_trades():
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)

        return {sym: self.get_statistics(trades) for sym, trades in by_symbol.items()}

    def get_performance_by_setup(self) -> dict[str, JournalStatistics]:
        """Get statistics broken down by setup type.

        Returns:
            Dictionary mapping setup type to JournalStatistics.
        """
        by_setup: dict[str, list[TradeEntry]] = {}
        for trade in self.get_closed_trades():
            setup = trade.setup_type or "unclassified"
            if setup not in by_setup:
                by_setup[setup] = []
            by_setup[setup].append(trade)

        return {setup: self.get_statistics(trades) for setup, trades in by_setup.items()}

    def get_performance_by_direction(self) -> dict[str, JournalStatistics]:
        """Get statistics broken down by direction (long/short).

        Returns:
            Dictionary mapping direction to JournalStatistics.
        """
        long_trades = [t for t in self.get_closed_trades() if t.direction == "long"]
        short_trades = [t for t in self.get_closed_trades() if t.direction == "short"]

        return {
            "long": self.get_statistics(long_trades),
            "short": self.get_statistics(short_trades),
        }

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        """Calculate cumulative equity curve from closed trades.

        Returns:
            List of (datetime, cumulative_pnl) tuples.
        """
        closed = sorted(
            [t for t in self.get_closed_trades() if t.exit_time],
            key=lambda t: t.exit_time,  # type: ignore
        )

        curve = []
        cumulative = 0.0

        for trade in closed:
            if trade.pnl is not None:
                cumulative += trade.pnl
                curve.append((trade.exit_time, cumulative))  # type: ignore

        return curve

    def save(self, path: str | Path | None = None) -> None:
        """Save journal to JSON file.

        Args:
            path: Path to save to. If None, uses journal_path.

        Raises:
            ValueError: If no path is specified.
        """
        save_path = Path(path) if path else self.journal_path
        if save_path is None:
            raise ValueError("No path specified for saving journal")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "trades": [t.to_dict() for t in self.trades],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path | None = None) -> None:
        """Load journal from JSON file.

        Args:
            path: Path to load from. If None, uses journal_path.

        Raises:
            ValueError: If no path is specified.
        """
        load_path = Path(path) if path else self.journal_path
        if load_path is None:
            raise ValueError("No path specified for loading journal")

        if not load_path.exists():
            return

        with open(load_path) as f:
            data = json.load(f)

        self.trades = [TradeEntry.from_dict(t) for t in data.get("trades", [])]

    def _auto_save(self) -> None:
        """Automatically save if a path is configured."""
        if self.journal_path:
            self.save()

    def export_to_csv(self, path: str | Path) -> None:
        """Export journal to CSV format.

        Args:
            path: Path to save CSV file.
        """
        import csv

        fields = [
            "trade_id", "symbol", "direction", "entry_price", "exit_price",
            "quantity", "entry_time", "exit_time", "pnl", "pnl_percent",
            "r_multiple", "entry_reason", "exit_reason", "setup_type",
            "timeframe", "stop_loss", "take_profit", "status", "tags", "notes",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for trade in self.trades:
                row = trade.to_dict()
                row["tags"] = ";".join(row.get("tags", []))
                writer.writerow({k: row.get(k, "") for k in fields})

    def generate_summary_report(self) -> str:
        """Generate a text summary report of the journal.

        Returns:
            Formatted string with journal summary.
        """
        stats = self.get_statistics()
        open_count = len(self.get_open_trades())

        lines = [
            "=" * 50,
            "TRADE JOURNAL SUMMARY",
            "=" * 50,
            "",
            f"Total Trades:        {stats.total_trades}",
            f"Open Positions:      {open_count}",
            f"Closed Trades:       {stats.total_trades}",
            "",
            f"Winning Trades:      {stats.winning_trades}",
            f"Losing Trades:       {stats.losing_trades}",
            f"Win Rate:            {stats.win_rate:.1%}",
            "",
            f"Total P&L:           ${stats.total_pnl:,.2f}",
            f"Average P&L:         ${stats.avg_pnl:,.2f}",
            f"Profit Factor:       {stats.profit_factor:.2f}",
            f"Expectancy:          ${stats.expectancy:,.2f}",
            "",
            f"Largest Win:         ${stats.largest_win:,.2f}",
            f"Largest Loss:        ${stats.largest_loss:,.2f}",
            f"Avg Winner:          ${stats.avg_winner:,.2f}",
            f"Avg Loser:           ${stats.avg_loser:,.2f}",
            "",
            f"Avg Holding Time:    {stats.avg_holding_time_hours:.1f} hours",
            f"Max Consecutive Wins:   {stats.max_consecutive_wins}",
            f"Max Consecutive Losses: {stats.max_consecutive_losses}",
        ]

        if stats.avg_r_multiple is not None:
            lines.append(f"Avg R-Multiple:      {stats.avg_r_multiple:.2f}R")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)
