"""Paper trading simulation system for strategy testing without real capital.

This module provides a complete paper trading environment that simulates
real broker behavior for testing trading strategies before live deployment.

Features:
    - Simulated market order execution with configurable slippage
    - Position tracking with average cost basis calculation
    - Realized and unrealized P&L computation
    - Trade history logging with timestamps
    - Cash and margin management

Example:
    >>> from imst_quant.paper_trading import PaperTradingSimulator
    >>> sim = PaperTradingSimulator(initial_cash=100000.0)
    >>> sim.submit_order("AAPL", 100, "buy", current_price=150.0)
    >>> print(sim.get_account_summary({"AAPL": 155.0}))
"""

from imst_quant.paper_trading.simulator import (
    Order,
    OrderSide,
    OrderStatus,
    PaperTradingSimulator,
    Position,
    SimulatorConfig,
)

__all__ = [
    "PaperTradingSimulator",
    "SimulatorConfig",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Position",
]


def run_paper_trade():
    """Placeholder for Alpaca paper trading integration.

    This function will be implemented to connect to Alpaca's
    paper trading API for live simulation with market data.
    """
    pass
