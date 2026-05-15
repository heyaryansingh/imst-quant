"""Real-time data streaming module for live market data.

This module provides WebSocket-based streaming for real-time market data,
enabling live trading signals, monitoring, and alerts.

Features:
- WebSocket connection management with auto-reconnection
- Real-time price updates from multiple sources
- Order book depth streaming
- Trade flow monitoring
- Signal generation on live data
- Event-driven architecture for callbacks

Example:
    >>> from imst_quant.streaming.realtime import RealtimeStream
    >>> def on_price_update(data):
    ...     print(f"Price: {data['price']}")
    >>> stream = RealtimeStream(symbols=["AAPL", "TSLA"])
    >>> stream.subscribe_trades(on_price_update)
    >>> stream.start()
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class StreamStatus(Enum):
    """Status of the data stream."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    price: float
    size: float
    timestamp: datetime
    side: str  # "buy" or "sell"
    trade_id: str


@dataclass
class Quote:
    """Represents a bid/ask quote."""
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime


@dataclass
class OrderBookLevel:
    """Represents a single order book level."""
    price: float
    size: float
    num_orders: int


@dataclass
class OrderBook:
    """Represents the full order book."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime


class StreamBuffer:
    """Circular buffer for streaming data with fixed size."""

    def __init__(self, maxlen: int = 1000):
        """Initialize buffer.

        Args:
            maxlen: Maximum number of items to store (default: 1000).
        """
        self.buffer: Deque[Any] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def append(self, item: Any) -> None:
        """Add item to buffer."""
        self.buffer.append(item)

    def get_recent(self, n: int = 100) -> List[Any]:
        """Get the N most recent items.

        Args:
            n: Number of items to retrieve.

        Returns:
            List of recent items (up to N).
        """
        return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Get buffer length."""
        return len(self.buffer)


class RealtimeStream:
    """Real-time data streaming manager.

    Handles WebSocket connections, data buffering, and event callbacks
    for live market data streaming.

    Example:
        >>> stream = RealtimeStream(
        ...     symbols=["AAPL", "MSFT"],
        ...     buffer_size=1000
        ... )
        >>> stream.subscribe_trades(lambda trade: print(f"Trade: {trade.price}"))
        >>> stream.subscribe_quotes(lambda quote: print(f"Spread: {quote.ask - quote.bid}"))
        >>> stream.start()
    """

    def __init__(
        self,
        symbols: List[str],
        buffer_size: int = 1000,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10,
    ):
        """Initialize realtime stream.

        Args:
            symbols: List of symbols to stream (e.g., ["AAPL", "TSLA"]).
            buffer_size: Size of circular buffer for each data type (default: 1000).
            reconnect_delay: Delay between reconnection attempts in seconds (default: 5).
            max_reconnect_attempts: Maximum number of reconnection attempts (default: 10).
        """
        self.symbols = symbols
        self.status = StreamStatus.DISCONNECTED
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_count = 0

        # Data buffers
        self.trade_buffer: Dict[str, StreamBuffer] = defaultdict(
            lambda: StreamBuffer(buffer_size)
        )
        self.quote_buffer: Dict[str, StreamBuffer] = defaultdict(
            lambda: StreamBuffer(buffer_size)
        )
        self.orderbook_buffer: Dict[str, OrderBook] = {}

        # Callbacks
        self.trade_callbacks: List[Callable[[Trade], None]] = []
        self.quote_callbacks: List[Callable[[Quote], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderBook], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []

        # Statistics
        self.stats = {
            "trades_received": 0,
            "quotes_received": 0,
            "orderbooks_received": 0,
            "connection_time": None,
            "last_message_time": None,
        }

        logger.info("realtime_stream_initialized", symbols=symbols)

    def subscribe_trades(self, callback: Callable[[Trade], None]) -> None:
        """Subscribe to trade updates.

        Args:
            callback: Function to call when a trade is received.
                Must accept a Trade object.

        Example:
            >>> def on_trade(trade: Trade):
            ...     print(f"{trade.symbol}: ${trade.price}")
            >>> stream.subscribe_trades(on_trade)
        """
        self.trade_callbacks.append(callback)
        logger.info("subscribed_to_trades", callback=callback.__name__)

    def subscribe_quotes(self, callback: Callable[[Quote], None]) -> None:
        """Subscribe to quote (bid/ask) updates.

        Args:
            callback: Function to call when a quote is received.
                Must accept a Quote object.

        Example:
            >>> def on_quote(quote: Quote):
            ...     spread = quote.ask - quote.bid
            ...     print(f"Spread: ${spread:.2f}")
            >>> stream.subscribe_quotes(on_quote)
        """
        self.quote_callbacks.append(callback)
        logger.info("subscribed_to_quotes", callback=callback.__name__)

    def subscribe_orderbook(self, callback: Callable[[OrderBook], None]) -> None:
        """Subscribe to order book updates.

        Args:
            callback: Function to call when order book is updated.
                Must accept an OrderBook object.

        Example:
            >>> def on_orderbook(book: OrderBook):
            ...     print(f"Best bid: ${book.bids[0].price}")
            ...     print(f"Best ask: ${book.asks[0].price}")
            >>> stream.subscribe_orderbook(on_orderbook)
        """
        self.orderbook_callbacks.append(callback)
        logger.info("subscribed_to_orderbook", callback=callback.__name__)

    def subscribe_errors(self, callback: Callable[[Exception], None]) -> None:
        """Subscribe to error events.

        Args:
            callback: Function to call when an error occurs.
                Must accept an Exception object.

        Example:
            >>> def on_error(error: Exception):
            ...     logger.error("stream_error", error=str(error))
            >>> stream.subscribe_errors(on_error)
        """
        self.error_callbacks.append(callback)
        logger.info("subscribed_to_errors", callback=callback.__name__)

    def _process_trade(self, data: Dict[str, Any]) -> None:
        """Process incoming trade data.

        Args:
            data: Raw trade data from WebSocket.
        """
        try:
            trade = Trade(
                symbol=data["symbol"],
                price=float(data["price"]),
                size=float(data["size"]),
                timestamp=datetime.fromtimestamp(data["timestamp"], tz=timezone.utc),
                side=data.get("side", "unknown"),
                trade_id=data.get("trade_id", ""),
            )

            # Update buffer
            self.trade_buffer[trade.symbol].append(trade)

            # Update stats
            self.stats["trades_received"] += 1
            self.stats["last_message_time"] = datetime.now(tz=timezone.utc)

            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    logger.error("trade_callback_error", callback=callback.__name__, error=str(e))
                    for error_cb in self.error_callbacks:
                        error_cb(e)

        except Exception as e:
            logger.error("trade_processing_error", data=data, error=str(e))
            for error_cb in self.error_callbacks:
                error_cb(e)

    def _process_quote(self, data: Dict[str, Any]) -> None:
        """Process incoming quote data.

        Args:
            data: Raw quote data from WebSocket.
        """
        try:
            quote = Quote(
                symbol=data["symbol"],
                bid=float(data["bid"]),
                ask=float(data["ask"]),
                bid_size=float(data.get("bid_size", 0)),
                ask_size=float(data.get("ask_size", 0)),
                timestamp=datetime.fromtimestamp(data["timestamp"], tz=timezone.utc),
            )

            # Update buffer
            self.quote_buffer[quote.symbol].append(quote)

            # Update stats
            self.stats["quotes_received"] += 1
            self.stats["last_message_time"] = datetime.now(tz=timezone.utc)

            # Trigger callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    logger.error("quote_callback_error", callback=callback.__name__, error=str(e))
                    for error_cb in self.error_callbacks:
                        error_cb(e)

        except Exception as e:
            logger.error("quote_processing_error", data=data, error=str(e))
            for error_cb in self.error_callbacks:
                error_cb(e)

    def get_recent_trades(self, symbol: str, n: int = 100) -> List[Trade]:
        """Get recent trades for a symbol.

        Args:
            symbol: Symbol to query (e.g., "AAPL").
            n: Number of recent trades to retrieve (default: 100).

        Returns:
            List of recent Trade objects.

        Example:
            >>> recent = stream.get_recent_trades("AAPL", n=50)
            >>> avg_price = sum(t.price for t in recent) / len(recent)
            >>> print(f"Average price: ${avg_price:.2f}")
        """
        return self.trade_buffer[symbol].get_recent(n)

    def get_recent_quotes(self, symbol: str, n: int = 100) -> List[Quote]:
        """Get recent quotes for a symbol.

        Args:
            symbol: Symbol to query (e.g., "AAPL").
            n: Number of recent quotes to retrieve (default: 100).

        Returns:
            List of recent Quote objects.

        Example:
            >>> recent = stream.get_recent_quotes("AAPL", n=20)
            >>> spreads = [q.ask - q.bid for q in recent]
            >>> avg_spread = sum(spreads) / len(spreads)
            >>> print(f"Average spread: ${avg_spread:.4f}")
        """
        return self.quote_buffer[symbol].get_recent(n)

    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for a symbol.

        Args:
            symbol: Symbol to query (e.g., "AAPL").

        Returns:
            Current OrderBook or None if not available.

        Example:
            >>> book = stream.get_orderbook("AAPL")
            >>> if book:
            ...     print(f"Bid depth: {sum(b.size for b in book.bids)}")
            ...     print(f"Ask depth: {sum(a.size for a in book.asks)}")
        """
        return self.orderbook_buffer.get(symbol)

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics.

        Returns:
            Dictionary with connection and data statistics.

        Example:
            >>> stats = stream.get_statistics()
            >>> print(f"Trades received: {stats['trades_received']}")
            >>> print(f"Uptime: {stats['connection_time']}")
        """
        uptime = None
        if self.stats["connection_time"]:
            uptime = (datetime.now(tz=timezone.utc) - self.stats["connection_time"]).total_seconds()

        return {
            **self.stats,
            "status": self.status.value,
            "symbols": self.symbols,
            "uptime_seconds": uptime,
            "buffer_sizes": {
                "trades": {sym: len(buf) for sym, buf in self.trade_buffer.items()},
                "quotes": {sym: len(buf) for sym, buf in self.quote_buffer.items()},
            },
        }

    def start(self) -> None:
        """Start the real-time stream.

        Note:
            This is a placeholder. In production, this would:
            1. Establish WebSocket connection
            2. Subscribe to market data feeds
            3. Start async event loop
            4. Handle reconnection logic

        Example:
            >>> stream = RealtimeStream(symbols=["AAPL"])
            >>> stream.subscribe_trades(on_trade_handler)
            >>> stream.start()  # Blocks until stream is stopped
        """
        logger.info("realtime_stream_starting", symbols=self.symbols)
        self.status = StreamStatus.CONNECTING
        self.stats["connection_time"] = datetime.now(tz=timezone.utc)

        # TODO: Implement WebSocket connection
        # For now, this is a framework ready for integration with:
        # - Alpaca API (websocket-client)
        # - Interactive Brokers (ib_insync)
        # - Binance (python-binance)
        # - Coinbase (cbpro)

        logger.warning(
            "realtime_stream_placeholder",
            message="WebSocket implementation pending. Framework is ready for integration."
        )

    def stop(self) -> None:
        """Stop the real-time stream.

        Example:
            >>> stream.stop()
            >>> print("Stream stopped successfully")
        """
        logger.info("realtime_stream_stopping")
        self.status = StreamStatus.DISCONNECTED
        # TODO: Close WebSocket connections
        # TODO: Cleanup resources


def create_momentum_scanner(
    stream: RealtimeStream,
    threshold: float = 0.02,
    window_seconds: int = 60,
) -> None:
    """Create a momentum scanner that alerts on price moves.

    Args:
        stream: RealtimeStream instance to attach to.
        threshold: Price move threshold as decimal (default: 0.02 = 2%).
        window_seconds: Time window for momentum calculation (default: 60s).

    Example:
        >>> stream = RealtimeStream(symbols=["AAPL", "TSLA", "MSFT"])
        >>> create_momentum_scanner(stream, threshold=0.03, window_seconds=120)
        >>> stream.start()
        # Will print alerts when stocks move >3% in 2 minutes
    """
    price_history: Dict[str, List[tuple[datetime, float]]] = defaultdict(list)

    def on_trade(trade: Trade) -> None:
        """Handle incoming trade for momentum detection."""
        # Add to history
        price_history[trade.symbol].append((trade.timestamp, trade.price))

        # Keep only recent data
        cutoff = trade.timestamp.timestamp() - window_seconds
        price_history[trade.symbol] = [
            (ts, price) for ts, price in price_history[trade.symbol]
            if ts.timestamp() >= cutoff
        ]

        # Calculate momentum
        if len(price_history[trade.symbol]) >= 2:
            oldest_price = price_history[trade.symbol][0][1]
            current_price = trade.price
            pct_change = (current_price - oldest_price) / oldest_price

            if abs(pct_change) >= threshold:
                direction = "UP" if pct_change > 0 else "DOWN"
                logger.warning(
                    "momentum_alert",
                    symbol=trade.symbol,
                    direction=direction,
                    pct_change=f"{pct_change*100:.2f}%",
                    price=current_price,
                    window_seconds=window_seconds,
                )

    stream.subscribe_trades(on_trade)
    logger.info(
        "momentum_scanner_created",
        threshold=f"{threshold*100:.1f}%",
        window_seconds=window_seconds,
    )
