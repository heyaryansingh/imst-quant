"""Paper trading simulator for live strategy testing without real capital.

This module provides a complete paper trading simulation system that mimics
real broker behavior including order execution, fill simulation, position
tracking, and P&L calculation. Designed for testing strategies before
deploying to live markets.

Features:
    - Simulated market orders with configurable slippage
    - Position tracking with average cost basis
    - Realized and unrealized P&L calculation
    - Trade history logging with timestamps
    - Cash management and margin tracking

Example:
    >>> from imst_quant.paper_trading.simulator import PaperTradingSimulator
    >>> sim = PaperTradingSimulator(initial_cash=100000.0)
    >>> sim.submit_order("AAPL", 100, "buy", current_price=150.0)
    >>> print(sim.get_position("AAPL"))
    {'quantity': 100, 'avg_cost': 150.15, 'current_value': 15015.0}

Note:
    This simulator uses simplified execution logic. Real market conditions
    include additional factors like partial fills, order queuing, and
    market impact that this simulator does not fully model.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a paper trading order.

    Attributes:
        symbol: Ticker symbol for the order.
        quantity: Number of shares to trade.
        side: Buy or sell direction.
        price: Execution price after slippage.
        timestamp: UTC time when order was created.
        status: Current order status.
        fill_time: UTC time when order was filled (if applicable).
        order_id: Unique identifier for the order.
    """

    symbol: str
    quantity: int
    side: OrderSide
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    fill_time: Optional[datetime] = None
    order_id: str = ""

    def __post_init__(self) -> None:
        """Generate order ID if not provided."""
        if not self.order_id:
            self.order_id = f"{self.symbol}-{self.timestamp.timestamp():.0f}"


@dataclass
class Position:
    """Tracks an open position in a single security.

    Attributes:
        symbol: Ticker symbol.
        quantity: Number of shares held (negative for short positions).
        avg_cost: Average cost basis per share.
        realized_pnl: Cumulative realized profit/loss from closed trades.
    """

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class SimulatorConfig:
    """Configuration for the paper trading simulator.

    Attributes:
        slippage_bps: Slippage in basis points (1bp = 0.0001).
        commission_per_share: Commission charged per share traded.
        min_commission: Minimum commission per order.
        allow_short_selling: Whether to allow short positions.
        max_position_value: Maximum value for a single position (optional).
    """

    slippage_bps: float = 10.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    allow_short_selling: bool = True
    max_position_value: Optional[float] = None


class PaperTradingSimulator:
    """Simulates paper trading with realistic order execution.

    This simulator provides a complete paper trading environment for testing
    trading strategies without risking real capital. It simulates order
    execution with slippage, tracks positions, and calculates P&L.

    Attributes:
        config: Simulator configuration settings.
        initial_cash: Starting cash balance.
        cash: Current available cash.
        positions: Dictionary of open positions by symbol.
        orders: List of all orders submitted.

    Example:
        >>> sim = PaperTradingSimulator(initial_cash=50000.0)
        >>> sim.submit_order("TSLA", 10, "buy", current_price=200.0)
        >>> sim.submit_order("TSLA", 5, "sell", current_price=210.0)
        >>> summary = sim.get_account_summary({"TSLA": 215.0})
        >>> print(f"Total equity: ${summary['total_equity']:,.2f}")
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        config: Optional[SimulatorConfig] = None,
    ) -> None:
        """Initialize the paper trading simulator.

        Args:
            initial_cash: Starting cash balance in USD.
            config: Optional simulator configuration. Uses defaults if not provided.
        """
        self.config = config or SimulatorConfig()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []

        logger.info(
            "paper_trading_simulator_initialized",
            initial_cash=initial_cash,
            slippage_bps=self.config.slippage_bps,
        )

    def _calculate_fill_price(
        self, current_price: float, side: OrderSide
    ) -> float:
        """Calculate execution price with slippage.

        Args:
            current_price: Current market price.
            side: Order direction.

        Returns:
            Fill price adjusted for slippage.
        """
        slippage_multiplier = self.config.slippage_bps / 10000.0
        if side == OrderSide.BUY:
            return current_price * (1 + slippage_multiplier)
        return current_price * (1 - slippage_multiplier)

    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a trade.

        Args:
            quantity: Number of shares traded.

        Returns:
            Commission amount in USD.
        """
        commission = abs(quantity) * self.config.commission_per_share
        return max(commission, self.config.min_commission)

    def submit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        current_price: float,
    ) -> Order:
        """Submit a paper trading order.

        Args:
            symbol: Ticker symbol to trade.
            quantity: Number of shares (positive integer).
            side: "buy" or "sell".
            current_price: Current market price for fill simulation.

        Returns:
            Order object with fill status.

        Raises:
            ValueError: If quantity is not positive or side is invalid.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        order_side = OrderSide(side.lower())
        fill_price = self._calculate_fill_price(current_price, order_side)
        commission = self._calculate_commission(quantity)
        order_value = fill_price * quantity

        now = datetime.now(timezone.utc)
        order = Order(
            symbol=symbol,
            quantity=quantity,
            side=order_side,
            price=fill_price,
            timestamp=now,
        )

        # Check if we have enough cash for buy orders
        if order_side == OrderSide.BUY:
            total_cost = order_value + commission
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    "order_rejected_insufficient_cash",
                    symbol=symbol,
                    required=total_cost,
                    available=self.cash,
                )
                self.orders.append(order)
                return order

        # Check short selling rules
        if order_side == OrderSide.SELL:
            current_qty = self.positions.get(symbol, Position(symbol)).quantity
            if current_qty < quantity and not self.config.allow_short_selling:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    "order_rejected_no_short_selling",
                    symbol=symbol,
                    requested=quantity,
                    held=current_qty,
                )
                self.orders.append(order)
                return order

        # Execute the order
        self._execute_order(order, commission)
        self.orders.append(order)

        logger.info(
            "order_executed",
            order_id=order.order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
        )

        return order

    def _execute_order(self, order: Order, commission: float) -> None:
        """Execute an order and update positions.

        Args:
            order: The order to execute.
            commission: Commission charged for the trade.
        """
        order.status = OrderStatus.FILLED
        order.fill_time = datetime.now(timezone.utc)

        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(symbol=order.symbol)

        position = self.positions[order.symbol]
        order_value = order.price * order.quantity

        if order.side == OrderSide.BUY:
            # Update cash
            self.cash -= order_value + commission

            # Update position with new average cost
            new_quantity = position.quantity + order.quantity
            if position.quantity > 0:
                total_cost = position.avg_cost * position.quantity + order_value
                position.avg_cost = total_cost / new_quantity
            else:
                position.avg_cost = order.price
            position.quantity = new_quantity

        else:  # SELL
            # Calculate realized P&L
            if position.quantity > 0:
                pnl = (order.price - position.avg_cost) * min(
                    order.quantity, position.quantity
                )
                position.realized_pnl += pnl

            # Update position
            position.quantity -= order.quantity

            # Update cash
            self.cash += order_value - commission

    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            Dictionary with position details or empty position if none held.
        """
        position = self.positions.get(symbol, Position(symbol=symbol))
        return {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "avg_cost": position.avg_cost,
            "realized_pnl": position.realized_pnl,
        }

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions.

        Returns:
            List of position dictionaries for all non-zero positions.
        """
        return [
            self.get_position(symbol)
            for symbol, pos in self.positions.items()
            if pos.quantity != 0
        ]

    def get_account_summary(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Calculate account summary with current valuations.

        Args:
            current_prices: Dictionary mapping symbols to current prices.
                Required to calculate unrealized P&L and market value.

        Returns:
            Dictionary containing:
                - cash: Available cash
                - positions_value: Total market value of positions
                - total_equity: Cash + positions value
                - total_realized_pnl: Sum of all realized P&L
                - total_unrealized_pnl: Sum of all unrealized P&L
                - total_pnl: Realized + unrealized P&L
                - return_pct: Percentage return on initial capital
        """
        current_prices = current_prices or {}

        positions_value = 0.0
        total_realized_pnl = 0.0
        total_unrealized_pnl = 0.0

        for symbol, position in self.positions.items():
            total_realized_pnl += position.realized_pnl

            if position.quantity != 0 and symbol in current_prices:
                market_value = position.quantity * current_prices[symbol]
                positions_value += market_value

                cost_basis = position.quantity * position.avg_cost
                unrealized = market_value - cost_basis
                total_unrealized_pnl += unrealized

        total_equity = self.cash + positions_value
        total_pnl = total_realized_pnl + total_unrealized_pnl
        return_pct = ((total_equity - self.initial_cash) / self.initial_cash) * 100

        return {
            "cash": self.cash,
            "positions_value": positions_value,
            "total_equity": total_equity,
            "total_realized_pnl": total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "order_count": len(self.orders),
            "filled_orders": len(
                [o for o in self.orders if o.status == OrderStatus.FILLED]
            ),
        }

    def get_trade_history(self) -> List[Dict]:
        """Get history of all trades.

        Returns:
            List of order dictionaries sorted by timestamp.
        """
        return [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": o.quantity,
                "price": o.price,
                "status": o.status.value,
                "timestamp": o.timestamp.isoformat(),
                "fill_time": o.fill_time.isoformat() if o.fill_time else None,
            }
            for o in sorted(self.orders, key=lambda x: x.timestamp)
        ]

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        logger.info("paper_trading_simulator_reset")
