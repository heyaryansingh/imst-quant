"""Execution simulator for realistic order execution modeling.

This module simulates realistic order execution including market impact,
slippage, partial fills, and latency effects.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import polars as pl
import numpy as np


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0


@dataclass
class Fill:
    """Execution fill representation."""
    order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float


class ExecutionSimulator:
    """Simulates realistic order execution."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_model: str = "fixed",
        slippage_bps: float = 5.0,
        market_impact_factor: float = 0.1,
        latency_ms: int = 100,
        partial_fill_prob: float = 0.1,
        min_fill_ratio: float = 0.3,
    ):
        """Initialize execution simulator.

        Args:
            commission_rate: Commission as fraction of trade value
            slippage_model: Slippage model ('fixed', 'proportional', 'volume_based')
            slippage_bps: Base slippage in basis points
            market_impact_factor: Market impact scaling factor
            latency_ms: Execution latency in milliseconds
            partial_fill_prob: Probability of partial fill
            min_fill_ratio: Minimum fill ratio for partial fills
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.market_impact_factor = market_impact_factor
        self.latency_ms = latency_ms
        self.partial_fill_prob = partial_fill_prob
        self.min_fill_ratio = min_fill_ratio

        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0

    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        market_volume: float = 1000000.0,
    ) -> float:
        """Calculate slippage for order execution.

        Args:
            order: Order to execute
            market_price: Current market price
            market_volume: Current market volume

        Returns:
            Slippage amount as fraction of price
        """
        base_slippage = self.slippage_bps / 10000.0

        if self.slippage_model == "fixed":
            return base_slippage

        elif self.slippage_model == "proportional":
            # Slippage proportional to order size
            order_value = order.quantity * market_price
            impact = self.market_impact_factor * (order_value / market_volume)
            return base_slippage + impact

        elif self.slippage_model == "volume_based":
            # Slippage based on volume ratio
            volume_ratio = (order.quantity * market_price) / market_volume
            if volume_ratio < 0.01:
                return base_slippage
            elif volume_ratio < 0.05:
                return base_slippage * 2.0
            else:
                return base_slippage * 5.0

        return base_slippage

    def calculate_market_impact(
        self,
        order: Order,
        market_price: float,
        market_volume: float = 1000000.0,
    ) -> float:
        """Calculate market impact of order.

        Args:
            order: Order being executed
            market_price: Current market price
            market_volume: Current market volume

        Returns:
            Price impact as fraction
        """
        # Square root impact model
        order_value = order.quantity * market_price
        participation_rate = order_value / market_volume

        impact = self.market_impact_factor * np.sqrt(participation_rate)

        # Direction matters
        if order.side == OrderSide.BUY:
            return impact
        else:
            return -impact

    def should_partial_fill(self) -> bool:
        """Determine if order should be partially filled.

        Returns:
            True if order should be partially filled
        """
        return np.random.random() < self.partial_fill_prob

    def get_fill_ratio(self) -> float:
        """Get random fill ratio for partial fills.

        Returns:
            Fill ratio between min_fill_ratio and 1.0
        """
        return np.random.uniform(self.min_fill_ratio, 1.0)

    def execute_market_order(
        self,
        order: Order,
        market_price: float,
        market_volume: float = 1000000.0,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Fill]:
        """Execute market order.

        Args:
            order: Market order to execute
            market_price: Current market price
            market_volume: Current market volume
            timestamp: Execution timestamp

        Returns:
            Fill object if execution successful, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Add latency
        execution_time = timestamp + timedelta(milliseconds=self.latency_ms)

        # Calculate slippage and impact
        slippage = self.calculate_slippage(order, market_price, market_volume)
        impact = self.calculate_market_impact(order, market_price, market_volume)

        # Determine fill quantity
        if self.should_partial_fill() and order.status == OrderStatus.PENDING:
            fill_ratio = self.get_fill_ratio()
            fill_quantity = order.quantity * fill_ratio
            order.status = OrderStatus.PARTIAL
        else:
            fill_quantity = order.quantity - order.filled_quantity
            order.status = OrderStatus.FILLED

        # Calculate execution price with slippage and impact
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + slippage + impact)
        else:
            execution_price = market_price * (1 - slippage + impact)

        # Calculate commission
        trade_value = fill_quantity * execution_price
        commission = trade_value * self.commission_rate

        # Update order
        total_filled = order.filled_quantity + fill_quantity
        order.avg_fill_price = (
            (order.avg_fill_price * order.filled_quantity + execution_price * fill_quantity)
            / total_filled
        )
        order.filled_quantity = total_filled
        order.commission += commission

        # Create fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=f"FILL_{self.order_counter}_{len(self.fills)}",
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=execution_time,
            commission=commission,
            slippage=slippage,
        )

        self.fills.append(fill)
        return fill

    def execute_limit_order(
        self,
        order: Order,
        market_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Fill]:
        """Execute limit order if price condition met.

        Args:
            order: Limit order to execute
            market_price: Current market price
            timestamp: Execution timestamp

        Returns:
            Fill object if order executed, None if not filled
        """
        if order.price is None:
            raise ValueError("Limit order must have price")

        # Check if limit price reached
        can_execute = False
        if order.side == OrderSide.BUY and market_price <= order.price:
            can_execute = True
        elif order.side == OrderSide.SELL and market_price >= order.price:
            can_execute = True

        if not can_execute:
            return None

        # Execute at limit price (no slippage for limit orders)
        if timestamp is None:
            timestamp = datetime.now()

        execution_time = timestamp + timedelta(milliseconds=self.latency_ms)

        # Determine fill quantity
        fill_quantity = order.quantity - order.filled_quantity

        # Calculate commission
        trade_value = fill_quantity * order.price
        commission = trade_value * self.commission_rate

        # Update order
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.price
        order.commission += commission
        order.status = OrderStatus.FILLED

        # Create fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=f"FILL_{order.order_id}_{len(self.fills)}",
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=order.price,
            timestamp=execution_time,
            commission=commission,
            slippage=0.0,
        )

        self.fills.append(fill)
        return fill

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit new order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Created order
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=datetime.now(),
        )

        self.orders[order_id] = order
        return order

    def process_tick(
        self,
        symbol: str,
        market_price: float,
        market_volume: float = 1000000.0,
        timestamp: Optional[datetime] = None,
    ) -> List[Fill]:
        """Process market tick and execute pending orders.

        Args:
            symbol: Trading symbol
            market_price: Current market price
            market_volume: Current market volume
            timestamp: Tick timestamp

        Returns:
            List of fills executed on this tick
        """
        tick_fills = []

        for order in self.orders.values():
            if order.symbol != symbol:
                continue

            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                continue

            fill = None

            if order.order_type == OrderType.MARKET:
                fill = self.execute_market_order(order, market_price, market_volume, timestamp)

            elif order.order_type == OrderType.LIMIT:
                fill = self.execute_limit_order(order, market_price, timestamp)

            if fill:
                tick_fills.append(fill)

        return tick_fills

    def get_fills_df(self) -> pl.DataFrame:
        """Get fills as DataFrame.

        Returns:
            DataFrame with all fills
        """
        if not self.fills:
            return pl.DataFrame()

        fills_data = [
            {
                "order_id": f.order_id,
                "fill_id": f.fill_id,
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "timestamp": f.timestamp,
                "commission": f.commission,
                "slippage": f.slippage,
            }
            for f in self.fills
        ]

        return pl.DataFrame(fills_data)

    def get_order_stats(self) -> Dict:
        """Get order execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        if not self.fills:
            return {}

        total_fills = len(self.fills)
        total_commission = sum(f.commission for f in self.fills)
        avg_slippage = np.mean([f.slippage for f in self.fills])

        filled_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
        partial_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.PARTIAL)

        return {
            "total_orders": len(self.orders),
            "filled_orders": filled_orders,
            "partial_orders": partial_orders,
            "total_fills": total_fills,
            "total_commission": total_commission,
            "avg_slippage_bps": avg_slippage * 10000,
            "avg_commission_per_fill": total_commission / total_fills if total_fills > 0 else 0.0,
        }


def simulate_execution_scenario(
    signals_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    volume_col: str = "volume",
    timestamp_col: str = "timestamp",
    initial_capital: float = 10000.0,
    position_size: float = 0.1,
    **simulator_kwargs,
) -> Tuple[pl.DataFrame, Dict]:
    """Simulate execution scenario from signals.

    Args:
        signals_df: DataFrame with trading signals
        prices_df: DataFrame with market prices
        signal_col: Column containing signals (-1, 0, 1)
        price_col: Column containing prices
        volume_col: Column containing volume
        timestamp_col: Column containing timestamps
        initial_capital: Starting capital
        position_size: Position size as fraction of capital
        **simulator_kwargs: Additional arguments for ExecutionSimulator

    Returns:
        Tuple of (fills DataFrame, statistics dictionary)
    """
    simulator = ExecutionSimulator(**simulator_kwargs)

    # Merge signals and prices
    df = signals_df.join(prices_df, on=timestamp_col, how="inner")

    current_position = 0.0
    fills_list = []

    for row in df.iter_rows(named=True):
        signal = row[signal_col]
        price = row[price_col]
        volume = row.get(volume_col, 1000000.0)
        timestamp = row[timestamp_col]

        # Determine target position
        target_position = signal * position_size * initial_capital / price

        # Calculate required trade
        trade_quantity = target_position - current_position

        if abs(trade_quantity) < 0.001:  # Skip tiny trades
            continue

        # Submit order
        side = OrderSide.BUY if trade_quantity > 0 else OrderSide.SELL
        order = simulator.submit_order(
            symbol="ASSET",
            side=side,
            quantity=abs(trade_quantity),
            order_type=OrderType.MARKET,
        )

        # Execute order
        fills = simulator.process_tick("ASSET", price, volume, timestamp)
        fills_list.extend(fills)

        # Update position
        if order.status == OrderStatus.FILLED:
            current_position = target_position

    # Get results
    fills_df = simulator.get_fills_df()
    stats = simulator.get_order_stats()

    return fills_df, stats
