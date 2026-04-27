"""Kelly Criterion Position Sizing: Optimal bet sizing for trading strategies.

Implements various Kelly criterion variants for position sizing:
- Full Kelly: Maximizes log wealth growth
- Fractional Kelly: Reduces risk by betting fraction of Kelly
- Half Kelly: Common conservative approach (0.5x Kelly)
- Kelly with drawdown constraints
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class KellySizer:
    """Kelly criterion position sizing calculator."""

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position: float = 0.25,
        min_position: float = 0.01,
    ):
        """Initialize Kelly position sizer.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
            max_position: Maximum position size as fraction of capital
            min_position: Minimum position size (below this, don't trade)
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")

        if not 0 < max_position <= 1.0:
            raise ValueError("max_position must be in (0, 1]")

        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate full Kelly criterion position size.

        Formula: f* = (p * b - q) / b
        where:
        - p = win rate
        - q = loss rate (1 - p)
        - b = avg_win / avg_loss (win/loss ratio)

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade size (positive)
            avg_loss: Average losing trade size (positive, will be treated as loss)

        Returns:
            Full Kelly position size (0 to 1)
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning("invalid_win_rate", win_rate=win_rate)
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning("invalid_win_loss", avg_win=avg_win, avg_loss=avg_loss)
            return 0.0

        loss_rate = 1.0 - win_rate
        win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Kelly can be negative if strategy has negative expectancy
        if kelly < 0:
            logger.warning(
                "negative_kelly",
                kelly=kelly,
                win_rate=win_rate,
                win_loss_ratio=win_loss_ratio,
            )
            return 0.0

        return kelly

    def size_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> Dict[str, float]:
        """Calculate position size with constraints applied.

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size

        Returns:
            Dictionary with:
            - full_kelly: Unconstrained Kelly size
            - fractional_kelly: Kelly * kelly_fraction
            - recommended_size: Final size after all constraints
            - expectancy: Expected value per trade
        """
        full_kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        fractional_kelly = full_kelly * self.kelly_fraction

        # Apply position constraints
        recommended_size = np.clip(
            fractional_kelly,
            self.min_position,
            self.max_position,
        )

        # If below minimum, set to 0 (don't trade)
        if recommended_size < self.min_position:
            recommended_size = 0.0

        # Calculate expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "full_kelly": round(full_kelly, 4),
            "fractional_kelly": round(fractional_kelly, 4),
            "recommended_size": round(recommended_size, 4),
            "expectancy": round(expectancy, 6),
        }

    def size_from_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate position size from historical trades DataFrame.

        Args:
            trades: DataFrame with 'pnl' column (profit/loss per trade)

        Returns:
            Dictionary with position sizing recommendations
        """
        if "pnl" not in trades.columns:
            raise ValueError("trades DataFrame must have 'pnl' column")

        if len(trades) == 0:
            logger.warning("no_trades_provided")
            return {
                "full_kelly": 0.0,
                "fractional_kelly": 0.0,
                "recommended_size": 0.0,
                "expectancy": 0.0,
            }

        winning_trades = trades[trades["pnl"] > 0]
        losing_trades = trades[trades["pnl"] < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        total_trades = len(trades)

        if total_trades == 0 or num_wins == 0 or num_losses == 0:
            logger.warning(
                "insufficient_trade_data",
                total=total_trades,
                wins=num_wins,
                losses=num_losses,
            )
            return {
                "full_kelly": 0.0,
                "fractional_kelly": 0.0,
                "recommended_size": 0.0,
                "expectancy": 0.0,
            }

        win_rate = num_wins / total_trades
        avg_win = winning_trades["pnl"].mean()
        avg_loss = abs(losing_trades["pnl"].mean())

        return self.size_position(win_rate, avg_win, avg_loss)


def calculate_optimal_f(trades: pd.DataFrame) -> float:
    """Calculate Optimal F (Ralph Vince) position sizing.

    Optimal F maximizes geometric growth by finding the fraction
    that maximizes the terminal wealth function.

    Args:
        trades: DataFrame with 'pnl' column

    Returns:
        Optimal F value (0 to 1)
    """
    if "pnl" not in trades.columns:
        raise ValueError("trades DataFrame must have 'pnl' column")

    if len(trades) == 0:
        return 0.0

    # Largest loss (used as normalization)
    largest_loss = abs(trades["pnl"].min())

    if largest_loss == 0:
        logger.warning("no_losses_in_trades")
        return 0.0

    # Normalized returns
    normalized_pnl = trades["pnl"] / largest_loss

    # Search for optimal f in [0, 1]
    f_values = np.linspace(0.01, 1.0, 100)
    terminal_wealth = []

    for f in f_values:
        # Calculate terminal wealth function (TWR)
        hpr = 1 + (f * normalized_pnl)  # Holding period returns
        twr = hpr.prod()  # Terminal wealth ratio
        terminal_wealth.append(twr)

    # Find f that maximizes TWR
    optimal_idx = np.argmax(terminal_wealth)
    optimal_f = f_values[optimal_idx]

    return float(optimal_f)


def variance_adjusted_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    win_variance: float,
    loss_variance: float,
) -> float:
    """Calculate variance-adjusted Kelly criterion.

    Accounts for variance in wins and losses, not just means.

    Args:
        win_rate: Win rate (0 to 1)
        avg_win: Average win
        avg_loss: Average loss
        win_variance: Variance of winning trades
        loss_variance: Variance of losing trades

    Returns:
        Variance-adjusted Kelly fraction
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0

    loss_rate = 1.0 - win_rate

    # Expected return
    mu = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Variance of strategy
    var = (win_rate * (avg_win**2 + win_variance)) + (
        loss_rate * (avg_loss**2 + loss_variance)
    ) - (mu**2)

    if var <= 0:
        logger.warning("non_positive_variance", var=var)
        return 0.0

    # Variance-adjusted Kelly
    kelly = mu / var

    return max(0.0, kelly)
