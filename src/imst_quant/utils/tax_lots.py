"""Tax lot accounting for realized profit and loss.

Average-cost accounting is enough to mark a portfolio to market, but it cannot
answer the questions that decide after-tax return: which shares were sold, how
long they were held, and how much of the gain is short term. This module tracks
individual lots and matches sells against them under the usual disposal
methods.

Supported matching methods:

- ``fifo``: oldest lot first. The default, and the IRS default for US equities.
- ``lifo``: newest lot first.
- ``hifo``: highest cost basis first, which defers gains.
- ``lofo``: lowest cost basis first, which harvests gains early.

Example:
    >>> from imst_quant.utils.tax_lots import TaxLotTracker
    >>> tracker = TaxLotTracker(method="fifo")
    >>> tracker.buy("AAPL", 100, 150.0, "2024-01-10")
    >>> tracker.buy("AAPL", 100, 170.0, "2024-06-10")
    >>> sales = tracker.sell("AAPL", 150, 180.0, "2024-08-10")
    >>> round(sum(s.realized_pnl for s in sales), 2)
    3500.0

References:
    - IRS Publication 550: Investment Income and Expenses
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Union

import polars as pl

# Holding period at which a US long-term capital gain begins.
LONG_TERM_DAYS = 365

MATCHING_METHODS = ("fifo", "lifo", "hifo", "lofo")

DateLike = Union[str, date, datetime]


def _to_date(value: DateLike) -> date:
    """Coerce a date-like value to a ``date``.

    Args:
        value: ISO-8601 string, ``date``, or ``datetime``.

    Returns:
        The corresponding ``date``.

    Raises:
        TypeError: If the value is not a supported type.
        ValueError: If a string is not valid ISO-8601.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        # fromisoformat handles both "2024-01-10" and full timestamps.
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date value: {value!r}")


@dataclass
class TaxLot:
    """A single purchase lot still open, or partially open.

    Attributes:
        symbol: Ticker symbol.
        quantity: Shares remaining in the lot.
        cost_per_share: Purchase price per share, including any commission
            that was amortized into it at acquisition.
        acquired: Date the lot was acquired.
        lot_id: Monotonic identifier, unique per tracker.
    """

    symbol: str
    quantity: float
    cost_per_share: float
    acquired: date
    lot_id: int

    @property
    def cost_basis(self) -> float:
        """Total remaining cost basis of the lot."""
        return self.quantity * self.cost_per_share


@dataclass
class LotSale:
    """One lot consumed by a sell order.

    A single sell can span several lots, producing one ``LotSale`` per lot.

    Attributes:
        symbol: Ticker symbol.
        quantity: Shares disposed of from this lot.
        cost_per_share: Cost basis per share of the consumed lot.
        proceeds_per_share: Sale price per share.
        acquired: Date the consumed lot was acquired.
        disposed: Date of the sale.
        holding_days: Days held, disposal date minus acquisition date.
        lot_id: Identifier of the consumed lot.
    """

    symbol: str
    quantity: float
    cost_per_share: float
    proceeds_per_share: float
    acquired: date
    disposed: date
    holding_days: int
    lot_id: int

    @property
    def cost_basis(self) -> float:
        """Cost basis of the shares sold."""
        return self.quantity * self.cost_per_share

    @property
    def proceeds(self) -> float:
        """Gross proceeds from the shares sold."""
        return self.quantity * self.proceeds_per_share

    @property
    def realized_pnl(self) -> float:
        """Realized gain or loss on the shares sold."""
        return self.proceeds - self.cost_basis

    @property
    def is_long_term(self) -> bool:
        """Whether the holding period qualifies as long term."""
        return self.holding_days >= LONG_TERM_DAYS


@dataclass
class RealizedSummary:
    """Aggregate realized results across a set of lot sales.

    Attributes:
        total_proceeds: Gross proceeds.
        total_cost_basis: Cost basis of all shares sold.
        realized_pnl: Total realized gain or loss.
        short_term_pnl: Realized gain or loss held under one year.
        long_term_pnl: Realized gain or loss held one year or more.
        realized_gains: Sum of the positive lot results.
        realized_losses: Sum of the negative lot results, as a negative number.
        sale_count: Number of lot disposals.
        quantity_sold: Total shares disposed of.
        avg_holding_days: Quantity-weighted average holding period.
    """

    total_proceeds: float = 0.0
    total_cost_basis: float = 0.0
    realized_pnl: float = 0.0
    short_term_pnl: float = 0.0
    long_term_pnl: float = 0.0
    realized_gains: float = 0.0
    realized_losses: float = 0.0
    sale_count: int = 0
    quantity_sold: float = 0.0
    avg_holding_days: float = 0.0


class InsufficientLotsError(ValueError):
    """Raised when a sell exceeds the shares on hand."""


class TaxLotTracker:
    """Tracks open lots per symbol and matches sells against them.

    The tracker is long only. Short positions have no acquisition lot to match
    against, so they are rejected rather than silently mis-basised.

    Attributes:
        method: Lot matching method, one of :data:`MATCHING_METHODS`.
    """

    def __init__(self, method: str = "fifo") -> None:
        """Initialize the tracker.

        Args:
            method: Lot matching method, one of :data:`MATCHING_METHODS`.

        Raises:
            ValueError: If the method is not recognized.
        """
        method = method.lower()
        if method not in MATCHING_METHODS:
            raise ValueError(
                f"Unknown matching method {method!r}; "
                f"expected one of {', '.join(MATCHING_METHODS)}"
            )
        self.method = method
        self._lots: Dict[str, List[TaxLot]] = {}
        self._sales: List[LotSale] = []
        self._next_lot_id = 1

    def buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        trade_date: DateLike,
        commission: float = 0.0,
    ) -> TaxLot:
        """Open a lot.

        Args:
            symbol: Ticker symbol.
            quantity: Shares purchased, must be positive.
            price: Price per share, must be positive.
            trade_date: Acquisition date.
            commission: Commission for the purchase, amortized into the basis.

        Returns:
            The newly opened lot.

        Raises:
            ValueError: If quantity or price is not positive, or commission is
                negative.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if commission < 0:
            raise ValueError(f"Commission must be non-negative, got {commission}")

        lot = TaxLot(
            symbol=symbol,
            quantity=quantity,
            cost_per_share=price + commission / quantity,
            acquired=_to_date(trade_date),
            lot_id=self._next_lot_id,
        )
        self._next_lot_id += 1
        self._lots.setdefault(symbol, []).append(lot)
        return lot

    def sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        trade_date: DateLike,
        commission: float = 0.0,
    ) -> List[LotSale]:
        """Close shares against open lots using the configured method.

        Args:
            symbol: Ticker symbol.
            quantity: Shares sold, must be positive.
            price: Price per share, must be positive.
            trade_date: Disposal date.
            commission: Commission for the sale, deducted from proceeds.

        Returns:
            One :class:`LotSale` per lot consumed, in consumption order.

        Raises:
            ValueError: If quantity or price is not positive, or commission is
                negative.
            InsufficientLotsError: If the sell exceeds the shares on hand.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if commission < 0:
            raise ValueError(f"Commission must be non-negative, got {commission}")

        held = self.shares_held(symbol)
        if quantity > held:
            raise InsufficientLotsError(
                f"Cannot sell {quantity} shares of {symbol}; only {held} held. "
                "TaxLotTracker is long only."
            )

        disposed = _to_date(trade_date)
        net_price = price - commission / quantity
        lots = self._ordered_lots(symbol)

        sales: List[LotSale] = []
        remaining = quantity
        for lot in lots:
            if remaining <= 0:
                break
            matched = min(remaining, lot.quantity)
            sales.append(
                LotSale(
                    symbol=symbol,
                    quantity=matched,
                    cost_per_share=lot.cost_per_share,
                    proceeds_per_share=net_price,
                    acquired=lot.acquired,
                    disposed=disposed,
                    holding_days=(disposed - lot.acquired).days,
                    lot_id=lot.lot_id,
                )
            )
            lot.quantity -= matched
            remaining -= matched

        # Drop fully consumed lots, preserving the original purchase order of
        # the rest so FIFO stays stable across repeated sells.
        self._lots[symbol] = [lot for lot in self._lots[symbol] if lot.quantity > 0]
        self._sales.extend(sales)
        return sales

    def _ordered_lots(self, symbol: str) -> List[TaxLot]:
        """Return open lots in the order the matching method consumes them."""
        lots = self._lots.get(symbol, [])
        if self.method == "fifo":
            return sorted(lots, key=lambda lot: (lot.acquired, lot.lot_id))
        if self.method == "lifo":
            return sorted(lots, key=lambda lot: (lot.acquired, lot.lot_id), reverse=True)
        if self.method == "hifo":
            return sorted(lots, key=lambda lot: (-lot.cost_per_share, lot.lot_id))
        return sorted(lots, key=lambda lot: (lot.cost_per_share, lot.lot_id))

    def shares_held(self, symbol: str) -> float:
        """Total open shares for a symbol."""
        return sum(lot.quantity for lot in self._lots.get(symbol, []))

    def cost_basis(self, symbol: str) -> float:
        """Total open cost basis for a symbol."""
        return sum(lot.cost_basis for lot in self._lots.get(symbol, []))

    def average_cost(self, symbol: str) -> float:
        """Average cost per open share, or 0.0 when flat."""
        held = self.shares_held(symbol)
        return self.cost_basis(symbol) / held if held else 0.0

    def open_lots(self, symbol: Optional[str] = None) -> List[TaxLot]:
        """Open lots for one symbol, or for every symbol when none is given."""
        if symbol is not None:
            return list(self._lots.get(symbol, []))
        return [lot for lots in self._lots.values() for lot in lots]

    def sales(self, symbol: Optional[str] = None) -> List[LotSale]:
        """Recorded lot disposals, optionally filtered to one symbol."""
        if symbol is None:
            return list(self._sales)
        return [sale for sale in self._sales if sale.symbol == symbol]

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Unrealized gain or loss on open lots.

        Args:
            prices: Current price per symbol. Symbols without a price are
                skipped rather than marked at zero.

        Returns:
            Total unrealized gain or loss.
        """
        total = 0.0
        for symbol, lots in self._lots.items():
            price = prices.get(symbol)
            if price is None:
                continue
            total += sum((price - lot.cost_per_share) * lot.quantity for lot in lots)
        return total

    def summarize(self, symbol: Optional[str] = None) -> RealizedSummary:
        """Aggregate realized results.

        Args:
            symbol: Restrict to one symbol, or summarize everything when None.

        Returns:
            A :class:`RealizedSummary` over the matching lot sales.
        """
        return summarize_sales(self.sales(symbol))

    def to_polars(self, symbol: Optional[str] = None) -> pl.DataFrame:
        """Return recorded lot disposals as a DataFrame."""
        return sales_to_polars(self.sales(symbol))


def summarize_sales(sales: Sequence[LotSale]) -> RealizedSummary:
    """Aggregate a sequence of lot sales.

    Args:
        sales: Lot disposals to aggregate.

    Returns:
        A :class:`RealizedSummary`. An empty sequence yields an all-zero
        summary rather than raising.
    """
    summary = RealizedSummary()
    if not sales:
        return summary

    weighted_days = 0.0
    for sale in sales:
        pnl = sale.realized_pnl
        summary.total_proceeds += sale.proceeds
        summary.total_cost_basis += sale.cost_basis
        summary.realized_pnl += pnl
        summary.quantity_sold += sale.quantity
        weighted_days += sale.holding_days * sale.quantity

        if sale.is_long_term:
            summary.long_term_pnl += pnl
        else:
            summary.short_term_pnl += pnl

        if pnl >= 0:
            summary.realized_gains += pnl
        else:
            summary.realized_losses += pnl

    summary.sale_count = len(sales)
    if summary.quantity_sold:
        summary.avg_holding_days = weighted_days / summary.quantity_sold
    return summary


def sales_to_polars(sales: Sequence[LotSale]) -> pl.DataFrame:
    """Convert lot sales to a Polars DataFrame.

    Args:
        sales: Lot disposals to convert.

    Returns:
        A DataFrame with one row per disposal. Empty input yields an empty
        DataFrame with the same schema so downstream code can chain safely.
    """
    schema = {
        "symbol": pl.Utf8,
        "lot_id": pl.Int64,
        "quantity": pl.Float64,
        "acquired": pl.Date,
        "disposed": pl.Date,
        "holding_days": pl.Int64,
        "cost_per_share": pl.Float64,
        "proceeds_per_share": pl.Float64,
        "cost_basis": pl.Float64,
        "proceeds": pl.Float64,
        "realized_pnl": pl.Float64,
        "is_long_term": pl.Boolean,
    }
    rows = [
        {
            "symbol": sale.symbol,
            "lot_id": sale.lot_id,
            "quantity": sale.quantity,
            "acquired": sale.acquired,
            "disposed": sale.disposed,
            "holding_days": sale.holding_days,
            "cost_per_share": sale.cost_per_share,
            "proceeds_per_share": sale.proceeds_per_share,
            "cost_basis": sale.cost_basis,
            "proceeds": sale.proceeds,
            "realized_pnl": sale.realized_pnl,
            "is_long_term": sale.is_long_term,
        }
        for sale in sales
    ]
    return pl.DataFrame(rows, schema=schema)


def compare_methods(
    trades: Sequence[Dict],
    methods: Sequence[str] = MATCHING_METHODS,
) -> Dict[str, RealizedSummary]:
    """Replay the same trades under each matching method.

    Useful for quantifying how much realized gain a disposal method defers.

    Args:
        trades: Ordered trades, each a mapping with ``symbol``, ``quantity``,
            ``price``, ``date``, and ``side`` (``"buy"`` or ``"sell"``), plus
            an optional ``commission``.
        methods: Methods to compare.

    Returns:
        Mapping of method name to its :class:`RealizedSummary`.

    Raises:
        ValueError: If a trade has an unrecognized side.
    """
    results: Dict[str, RealizedSummary] = {}
    for method in methods:
        tracker = TaxLotTracker(method=method)
        for trade in trades:
            side = str(trade["side"]).lower()
            args = (
                trade["symbol"],
                trade["quantity"],
                trade["price"],
                trade["date"],
                trade.get("commission", 0.0),
            )
            if side == "buy":
                tracker.buy(*args)
            elif side == "sell":
                tracker.sell(*args)
            else:
                raise ValueError(f"Unknown trade side {side!r}; expected buy or sell")
        results[method] = tracker.summarize()
    return results


def find_harvestable_lots(
    tracker: TaxLotTracker,
    prices: Dict[str, float],
    min_loss: float = 0.0,
) -> List[TaxLot]:
    """Find open lots sitting at an unrealized loss.

    Args:
        tracker: Tracker holding the open lots.
        prices: Current price per symbol. Symbols without a price are skipped.
        min_loss: Minimum loss magnitude to report, as a positive number.

    Returns:
        Lots whose unrealized loss meets the threshold, largest loss first.

    Raises:
        ValueError: If ``min_loss`` is negative.
    """
    if min_loss < 0:
        raise ValueError(f"min_loss must be non-negative, got {min_loss}")

    harvestable = []
    for lot in tracker.open_lots():
        price = prices.get(lot.symbol)
        if price is None:
            continue
        loss = (price - lot.cost_per_share) * lot.quantity
        if loss < 0 and abs(loss) >= min_loss:
            harvestable.append((loss, lot))

    harvestable.sort(key=lambda pair: pair[0])
    return [lot for _, lot in harvestable]
