"""Portfolio policy with risk limits (TRAD-U01).

This module implements portfolio allocation with risk management including
drawdown limits, daily loss limits, position size constraints, and
automatic risk scaling based on current portfolio state.

Example:
    >>> policy = PortfolioPolicy(max_drawdown=0.10, max_position=0.20)
    >>> signals = {"BTC": 1, "ETH": -1, "SOL": 1}
    >>> prices = {"BTC": 50000, "ETH": 3000, "SOL": 100}
    >>> weights = policy.allocate(signals, prices)
    >>> policy.update_equity(0.015)  # 1.5% gain
"""

from typing import Dict, Optional


class PortfolioPolicy:
    """Portfolio policy with drawdown/exposure limits and risk scaling.

    Manages position sizing with configurable risk limits:
    - Maximum drawdown triggers trading halt
    - Daily loss limits reduce position sizes
    - Per-position limits prevent concentration
    - Automatic deleveraging when limits approached

    Attributes:
        max_drawdown: Maximum allowed peak-to-trough decline.
        max_daily_loss: Maximum allowed loss per day.
        max_position: Maximum weight per single position.
        equity: Current portfolio equity (starts at 1.0).
        peak: Highest equity reached.
        daily_pnl: Accumulated profit/loss for current day.
        is_halted: Whether trading is halted due to limits.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        max_daily_loss: float = 0.02,
        max_position: float = 0.25,
        leverage_limit: float = 1.0,
    ):
        """Initialize portfolio policy with risk parameters.

        Args:
            max_drawdown: Maximum allowed drawdown before halt (default: 10%).
            max_daily_loss: Maximum daily loss before reducing size (default: 2%).
            max_position: Maximum weight per position (default: 25%).
            leverage_limit: Maximum total exposure (default: 1.0 = no leverage).
        """
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_position = max_position
        self.leverage_limit = leverage_limit

        self.peak = 1.0
        self.equity = 1.0
        self.daily_pnl = 0.0
        self.is_halted = False
        self._position_history: list[Dict[str, float]] = []

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak == 0:
            return 0.0
        return (self.peak - self.equity) / self.peak

    @property
    def risk_scalar(self) -> float:
        """Calculate risk scaling factor based on current state.

        Returns value between 0 and 1 to scale down positions when
        approaching risk limits.
        """
        if self.is_halted:
            return 0.0

        # Scale down if approaching drawdown limit
        dd_utilization = self.current_drawdown / self.max_drawdown
        dd_scalar = max(0.0, 1.0 - (dd_utilization * 0.5))

        # Scale down if daily loss is significant
        daily_utilization = abs(min(0, self.daily_pnl)) / self.max_daily_loss
        daily_scalar = max(0.0, 1.0 - (daily_utilization * 0.5))

        return min(dd_scalar, daily_scalar)

    def allocate(
        self,
        signals: Dict[str, int],
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Allocate portfolio weights based on signals with risk limits.

        Uses equal weight allocation scaled by risk state. Positions are
        capped at max_position and total exposure at leverage_limit.

        Args:
            signals: Dict mapping asset to signal (-1, 0, or 1).
            prices: Optional dict of current prices (unused, for API compat).

        Returns:
            Dict mapping asset to portfolio weight (can be negative for shorts).
        """
        if self.is_halted:
            return {a: 0.0 for a in signals}

        active_signals = {a: s for a, s in signals.items() if s != 0}
        n = len(active_signals)

        if n == 0:
            return {a: 0.0 for a in signals}

        # Base equal weight
        base_weight = 1.0 / n

        # Apply risk scaling
        scaled_weight = base_weight * self.risk_scalar

        # Cap at max position
        capped_weight = min(scaled_weight, self.max_position)

        # Build allocation
        allocation = {}
        total_exposure = 0.0

        for asset, signal in signals.items():
            if signal == 0:
                allocation[asset] = 0.0
            else:
                weight = capped_weight * signal
                # Enforce leverage limit
                if total_exposure + abs(weight) > self.leverage_limit:
                    remaining = self.leverage_limit - total_exposure
                    weight = remaining * (1 if signal > 0 else -1)
                allocation[asset] = weight
                total_exposure += abs(weight)

        self._position_history.append(allocation.copy())
        return allocation

    def update_equity(self, daily_return: float) -> None:
        """Update equity based on daily portfolio return.

        Updates equity, peak, daily PnL, and checks halt conditions.

        Args:
            daily_return: Portfolio return for the day (e.g., 0.01 = 1% gain).
        """
        self.daily_pnl = daily_return
        self.equity *= (1 + daily_return)

        # Update peak
        if self.equity > self.peak:
            self.peak = self.equity

        # Check halt conditions
        if self.current_drawdown >= self.max_drawdown:
            self.is_halted = True

    def reset_daily(self) -> None:
        """Reset daily tracking for new trading day."""
        self.daily_pnl = 0.0

    def reset_halt(self) -> None:
        """Manually reset halt status (e.g., after review)."""
        self.is_halted = False

    def get_status(self) -> Dict[str, float]:
        """Get current portfolio risk status.

        Returns:
            Dict with equity, drawdown, daily_pnl, risk_scalar, and halt status.
        """
        return {
            "equity": self.equity,
            "peak": self.peak,
            "drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "risk_scalar": self.risk_scalar,
            "is_halted": float(self.is_halted),
        }
