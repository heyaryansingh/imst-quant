"""Portfolio rebalancing optimization utilities.

This module provides advanced portfolio rebalancing strategies including
threshold-based, periodic, volatility-adjusted, and tactical rebalancing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import polars as pl
import numpy as np


class RebalanceStrategy(Enum):
    """Rebalancing strategy types."""
    THRESHOLD = "threshold"
    PERIODIC = "periodic"
    VOLATILITY = "volatility"
    TACTICAL = "tactical"
    BAND = "band"


@dataclass
class RebalanceTrade:
    """Represents a rebalancing trade."""
    symbol: str
    current_weight: float
    target_weight: float
    current_value: float
    target_value: float
    trade_value: float
    trade_shares: float
    timestamp: datetime


class PortfolioRebalancer:
    """Portfolio rebalancing optimizer."""

    def __init__(
        self,
        target_weights: Dict[str, float],
        rebalance_threshold: float = 0.05,
        rebalance_frequency_days: int = 30,
        min_trade_value: float = 100.0,
        transaction_cost: float = 0.001,
    ):
        """Initialize portfolio rebalancer.

        Args:
            target_weights: Target weight for each asset
            rebalance_threshold: Drift threshold to trigger rebalance
            rebalance_frequency_days: Days between periodic rebalances
            min_trade_value: Minimum trade value to execute
            transaction_cost: Transaction cost as fraction
        """
        # Validate weights sum to 1
        total_weight = sum(target_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight}")

        self.target_weights = target_weights
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency_days = rebalance_frequency_days
        self.min_trade_value = min_trade_value
        self.transaction_cost = transaction_cost

        self.last_rebalance: Optional[datetime] = None
        self.rebalance_history: List[Dict] = []

    def calculate_current_weights(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            positions: Current position sizes (shares)
            prices: Current prices for each asset

        Returns:
            Dictionary of current weights
        """
        values = {symbol: positions.get(symbol, 0.0) * prices.get(symbol, 0.0)
                  for symbol in self.target_weights.keys()}

        total_value = sum(values.values())

        if total_value == 0:
            return {symbol: 0.0 for symbol in self.target_weights.keys()}

        return {symbol: value / total_value for symbol, value in values.items()}

    def calculate_drift(
        self,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate weight drift from target.

        Args:
            current_weights: Current portfolio weights

        Returns:
            Dictionary of drift amounts
        """
        return {
            symbol: current_weights.get(symbol, 0.0) - self.target_weights[symbol]
            for symbol in self.target_weights.keys()
        }

    def needs_rebalance_threshold(
        self,
        current_weights: Dict[str, float],
    ) -> bool:
        """Check if rebalance needed based on threshold.

        Args:
            current_weights: Current portfolio weights

        Returns:
            True if any asset exceeds drift threshold
        """
        drift = self.calculate_drift(current_weights)
        max_drift = max(abs(d) for d in drift.values())
        return max_drift > self.rebalance_threshold

    def needs_rebalance_periodic(
        self,
        current_time: datetime,
    ) -> bool:
        """Check if rebalance needed based on time.

        Args:
            current_time: Current timestamp

        Returns:
            True if enough time has passed since last rebalance
        """
        if self.last_rebalance is None:
            return True

        days_since_rebalance = (current_time - self.last_rebalance).days
        return days_since_rebalance >= self.rebalance_frequency_days

    def calculate_rebalance_trades(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        total_value: float,
        timestamp: datetime,
    ) -> List[RebalanceTrade]:
        """Calculate trades needed for rebalancing.

        Args:
            positions: Current position sizes
            prices: Current prices
            total_value: Total portfolio value
            timestamp: Rebalance timestamp

        Returns:
            List of required trades
        """
        current_weights = self.calculate_current_weights(positions, prices)
        trades = []

        for symbol in self.target_weights.keys():
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = self.target_weights[symbol]

            current_value = positions.get(symbol, 0.0) * prices.get(symbol, 0.0)
            target_value = total_value * target_weight

            trade_value = target_value - current_value

            # Skip small trades
            if abs(trade_value) < self.min_trade_value:
                continue

            price = prices.get(symbol, 0.0)
            if price == 0:
                continue

            trade_shares = trade_value / price

            trade = RebalanceTrade(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                current_value=current_value,
                target_value=target_value,
                trade_value=trade_value,
                trade_shares=trade_shares,
                timestamp=timestamp,
            )

            trades.append(trade)

        return trades

    def calculate_rebalance_cost(
        self,
        trades: List[RebalanceTrade],
    ) -> float:
        """Calculate total cost of rebalancing.

        Args:
            trades: List of trades to execute

        Returns:
            Total transaction cost
        """
        total_trade_value = sum(abs(t.trade_value) for t in trades)
        return total_trade_value * self.transaction_cost

    def execute_rebalance(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        timestamp: datetime,
        strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
    ) -> Optional[Tuple[List[RebalanceTrade], float]]:
        """Execute rebalancing if needed.

        Args:
            positions: Current position sizes
            prices: Current prices
            timestamp: Current timestamp
            strategy: Rebalancing strategy to use

        Returns:
            Tuple of (trades, cost) if rebalance executed, None otherwise
        """
        current_weights = self.calculate_current_weights(positions, prices)

        # Check if rebalance needed based on strategy
        should_rebalance = False

        if strategy == RebalanceStrategy.THRESHOLD:
            should_rebalance = self.needs_rebalance_threshold(current_weights)
        elif strategy == RebalanceStrategy.PERIODIC:
            should_rebalance = self.needs_rebalance_periodic(timestamp)
        elif strategy in [RebalanceStrategy.VOLATILITY, RebalanceStrategy.TACTICAL, RebalanceStrategy.BAND]:
            # These strategies always attempt rebalance and decide internally
            should_rebalance = True

        if not should_rebalance:
            return None

        # Calculate total portfolio value
        total_value = sum(positions.get(s, 0.0) * prices.get(s, 0.0) for s in self.target_weights.keys())

        # Calculate required trades
        trades = self.calculate_rebalance_trades(positions, prices, total_value, timestamp)

        if not trades:
            return None

        # Calculate cost
        cost = self.calculate_rebalance_cost(trades)

        # Record rebalance
        self.last_rebalance = timestamp
        self.rebalance_history.append({
            "timestamp": timestamp,
            "strategy": strategy.value,
            "trades": len(trades),
            "cost": cost,
            "total_value": total_value,
        })

        return trades, cost

    def calculate_turnover(
        self,
        trades: List[RebalanceTrade],
        total_value: float,
    ) -> float:
        """Calculate portfolio turnover from trades.

        Args:
            trades: List of trades
            total_value: Total portfolio value

        Returns:
            Turnover as fraction of portfolio
        """
        if total_value == 0:
            return 0.0

        total_trade_value = sum(abs(t.trade_value) for t in trades)
        return total_trade_value / total_value


def optimize_rebalance_bands(
    returns_df: pl.DataFrame,
    target_weights: Dict[str, float],
    symbol_col: str = "symbol",
    returns_col: str = "returns",
    min_band: float = 0.01,
    max_band: float = 0.20,
    step: float = 0.01,
    transaction_cost: float = 0.001,
) -> Dict[str, float]:
    """Optimize rebalancing bands for each asset.

    Args:
        returns_df: DataFrame with historical returns
        target_weights: Target portfolio weights
        symbol_col: Column containing symbols
        returns_col: Column containing returns
        min_band: Minimum band width
        max_band: Maximum band width
        step: Band width step size
        transaction_cost: Transaction cost fraction

    Returns:
        Dictionary of optimal bands for each symbol
    """
    optimal_bands = {}

    for symbol in target_weights.keys():
        symbol_returns = returns_df.filter(pl.col(symbol_col) == symbol)[returns_col]

        if symbol_returns.is_empty():
            optimal_bands[symbol] = 0.05  # Default
            continue

        # Calculate volatility
        volatility = float(symbol_returns.std())

        # Test different band widths
        best_band = 0.05
        best_sharpe = -np.inf

        for band in np.arange(min_band, max_band + step, step):
            # Simulate rebalancing with this band
            # Simple heuristic: balance transaction costs with drift
            avg_rebalances = volatility / (band * np.sqrt(252))  # Approximate rebalances per year
            avg_turnover = 2 * band  # Average turnover per rebalance
            annual_cost = avg_rebalances * avg_turnover * transaction_cost

            # Lower cost is better
            # Higher band = fewer rebalances = lower cost but more drift
            score = -annual_cost + volatility * 0.1  # Penalize drift

            if score > best_sharpe:
                best_sharpe = score
                best_band = float(band)

        optimal_bands[symbol] = best_band

    return optimal_bands


def calculate_rebalance_efficiency(
    rebalance_history: List[Dict],
    returns_df: pl.DataFrame,
    returns_col: str = "returns",
) -> Dict:
    """Calculate efficiency metrics for rebalancing strategy.

    Args:
        rebalance_history: List of historical rebalance records
        returns_df: DataFrame with returns
        returns_col: Column containing returns

    Returns:
        Dictionary with efficiency metrics
    """
    if not rebalance_history:
        return {
            "total_rebalances": 0,
            "avg_cost": 0.0,
            "total_cost": 0.0,
            "avg_interval_days": 0.0,
        }

    total_rebalances = len(rebalance_history)
    total_cost = sum(r["cost"] for r in rebalance_history)
    avg_cost = total_cost / total_rebalances if total_rebalances > 0 else 0.0

    # Calculate average interval
    if total_rebalances > 1:
        intervals = [
            (rebalance_history[i]["timestamp"] - rebalance_history[i-1]["timestamp"]).days
            for i in range(1, total_rebalances)
        ]
        avg_interval_days = float(np.mean(intervals))
    else:
        avg_interval_days = 0.0

    # Calculate cost as percentage of total returns
    total_return = float((1 + returns_df[returns_col]).prod() - 1)
    cost_drag = total_cost / abs(total_return) if total_return != 0 else 0.0

    return {
        "total_rebalances": total_rebalances,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "avg_interval_days": avg_interval_days,
        "cost_drag": cost_drag,
    }


def simulate_rebalancing_strategy(
    prices_df: pl.DataFrame,
    target_weights: Dict[str, float],
    strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
    initial_value: float = 10000.0,
    **rebalancer_kwargs,
) -> Tuple[pl.DataFrame, Dict]:
    """Simulate rebalancing strategy over time.

    Args:
        prices_df: DataFrame with historical prices
        target_weights: Target portfolio weights
        strategy: Rebalancing strategy to use
        initial_value: Initial portfolio value
        **rebalancer_kwargs: Additional rebalancer parameters

    Returns:
        Tuple of (portfolio history DataFrame, statistics)
    """
    rebalancer = PortfolioRebalancer(target_weights, **rebalancer_kwargs)

    # Initialize positions
    positions = {}
    for symbol, weight in target_weights.items():
        if symbol in prices_df.columns:
            initial_price = float(prices_df[symbol][0])
            if initial_price > 0:
                positions[symbol] = (initial_value * weight) / initial_price

    history = []
    total_cost = 0.0

    for row in prices_df.iter_rows(named=True):
        timestamp = row.get("timestamp", datetime.now())
        prices = {symbol: row.get(symbol, 0.0) for symbol in target_weights.keys()}

        # Calculate current value
        current_value = sum(positions.get(s, 0.0) * prices.get(s, 0.0) for s in target_weights.keys())

        # Attempt rebalance
        result = rebalancer.execute_rebalance(positions, prices, timestamp, strategy)

        if result:
            trades, cost = result
            total_cost += cost

            # Apply trades
            for trade in trades:
                positions[trade.symbol] = positions.get(trade.symbol, 0.0) + trade.trade_shares

        # Record state
        weights = rebalancer.calculate_current_weights(positions, prices)
        history.append({
            "timestamp": timestamp,
            "value": current_value,
            **{f"{symbol}_weight": weights.get(symbol, 0.0) for symbol in target_weights.keys()},
        })

    history_df = pl.DataFrame(history)

    stats = {
        "final_value": current_value,
        "total_return": (current_value / initial_value) - 1,
        "total_cost": total_cost,
        "cost_as_pct_return": total_cost / abs(current_value - initial_value) if current_value != initial_value else 0.0,
        "num_rebalances": len(rebalancer.rebalance_history),
    }

    return history_df, stats
