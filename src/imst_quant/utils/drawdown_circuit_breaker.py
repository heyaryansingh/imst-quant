"""Drawdown circuit breaker for automated risk management.

Monitors portfolio drawdown in real-time and triggers protective actions
when configurable thresholds are breached. Supports tiered responses
(reduce, hedge, halt) and cooldown periods before resuming.

Example:
    >>> breaker = DrawdownCircuitBreaker(
    ...     warning_threshold=0.05,
    ...     reduce_threshold=0.10,
    ...     halt_threshold=0.20,
    ... )
    >>> action = breaker.check(current_drawdown=0.12)
    >>> print(action)  # CircuitAction.REDUCE
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class CircuitAction(Enum):
    """Actions the circuit breaker can recommend."""

    NORMAL = "normal"
    WARNING = "warning"
    REDUCE = "reduce"
    HEDGE = "hedge"
    HALT = "halt"


@dataclass
class CircuitEvent:
    """Record of a circuit breaker trigger.

    Attributes:
        timestamp: When the event occurred.
        action: The action triggered.
        drawdown: Drawdown level that caused the trigger.
        message: Human-readable description.
    """

    timestamp: datetime
    action: CircuitAction
    drawdown: float
    message: str


@dataclass
class CircuitState:
    """Current state of the circuit breaker.

    Attributes:
        current_action: The current recommended action.
        drawdown: Current drawdown level.
        peak_drawdown: Worst drawdown seen in this episode.
        is_halted: Whether trading is fully halted.
        cooldown_remaining: Periods remaining before resuming.
        events: History of trigger events.
        position_scale: Recommended position scale factor (0.0 to 1.0).
    """

    current_action: CircuitAction
    drawdown: float
    peak_drawdown: float
    is_halted: bool
    cooldown_remaining: int
    events: List[CircuitEvent]
    position_scale: float


class DrawdownCircuitBreaker:
    """Multi-tier drawdown circuit breaker.

    Monitors drawdown and recommends escalating protective actions:
    1. WARNING: Alert only, no position change.
    2. REDUCE: Scale down positions by reduce_factor.
    3. HEDGE: Signal to add hedges, further scale down.
    4. HALT: Stop all new trades, wait for cooldown.

    Args:
        warning_threshold: Drawdown level for warning (default 5%).
        reduce_threshold: Drawdown level to start reducing (default 10%).
        hedge_threshold: Drawdown level to add hedges (default 15%).
        halt_threshold: Drawdown level to halt trading (default 20%).
        reduce_factor: Position scale when reducing (default 0.5).
        hedge_factor: Position scale when hedging (default 0.25).
        cooldown_periods: Periods to wait after halt before resuming.
        recovery_threshold: Drawdown must recover below this to resume.

    Example:
        >>> breaker = DrawdownCircuitBreaker(
        ...     warning_threshold=0.05,
        ...     reduce_threshold=0.10,
        ...     halt_threshold=0.20,
        ... )
        >>> state = breaker.update(0.03)
        >>> assert state.current_action == CircuitAction.NORMAL
        >>> state = breaker.update(0.12)
        >>> assert state.current_action == CircuitAction.REDUCE
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,
        reduce_threshold: float = 0.10,
        hedge_threshold: float = 0.15,
        halt_threshold: float = 0.20,
        reduce_factor: float = 0.50,
        hedge_factor: float = 0.25,
        cooldown_periods: int = 10,
        recovery_threshold: float = 0.05,
    ) -> None:
        if not (0 < warning_threshold <= reduce_threshold <= halt_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: 0 < warning <= reduce <= halt <= 1.0"
            )
        if hedge_threshold < reduce_threshold or hedge_threshold > halt_threshold:
            raise ValueError(
                "hedge_threshold must be between reduce_threshold and halt_threshold"
            )

        self.warning_threshold = warning_threshold
        self.reduce_threshold = reduce_threshold
        self.hedge_threshold = hedge_threshold
        self.halt_threshold = halt_threshold
        self.reduce_factor = reduce_factor
        self.hedge_factor = hedge_factor
        self.cooldown_periods = cooldown_periods
        self.recovery_threshold = recovery_threshold

        self._events: List[CircuitEvent] = []
        self._is_halted = False
        self._cooldown_remaining = 0
        self._peak_drawdown = 0.0
        self._last_action = CircuitAction.NORMAL

    def check(self, current_drawdown: float) -> CircuitAction:
        """Determine the recommended action for a given drawdown level.

        This is a stateless check; use ``update()`` for stateful tracking.

        Args:
            current_drawdown: Current drawdown as a positive fraction (e.g., 0.10 = 10%).

        Returns:
            The recommended CircuitAction.
        """
        dd = abs(current_drawdown)

        if dd >= self.halt_threshold:
            return CircuitAction.HALT
        if dd >= self.hedge_threshold:
            return CircuitAction.HEDGE
        if dd >= self.reduce_threshold:
            return CircuitAction.REDUCE
        if dd >= self.warning_threshold:
            return CircuitAction.WARNING
        return CircuitAction.NORMAL

    def update(self, current_drawdown: float) -> CircuitState:
        """Update the circuit breaker with the current drawdown.

        Tracks state transitions, enforces cooldown after halt,
        and records trigger events.

        Args:
            current_drawdown: Current drawdown as a positive fraction.

        Returns:
            CircuitState with the current recommendation.
        """
        dd = abs(current_drawdown)
        self._peak_drawdown = max(self._peak_drawdown, dd)

        # Handle cooldown after halt
        if self._is_halted:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return self._build_state(CircuitAction.HALT, dd, position_scale=0.0)

            # Cooldown expired: check if recovered enough
            if dd <= self.recovery_threshold:
                self._is_halted = False
                self._peak_drawdown = dd
                self._record_event(
                    CircuitAction.NORMAL,
                    dd,
                    "Circuit breaker reset: drawdown recovered",
                )
            else:
                return self._build_state(CircuitAction.HALT, dd, position_scale=0.0)

        action = self.check(dd)

        # Record transitions
        if action != self._last_action:
            messages = {
                CircuitAction.WARNING: f"Warning: drawdown at {dd:.1%}",
                CircuitAction.REDUCE: f"Reducing positions: drawdown at {dd:.1%}",
                CircuitAction.HEDGE: f"Hedging triggered: drawdown at {dd:.1%}",
                CircuitAction.HALT: f"Trading halted: drawdown at {dd:.1%}",
                CircuitAction.NORMAL: f"Returned to normal: drawdown at {dd:.1%}",
            }
            self._record_event(action, dd, messages[action])
            self._last_action = action

        # Enter halt state
        if action == CircuitAction.HALT:
            self._is_halted = True
            self._cooldown_remaining = self.cooldown_periods

        # Determine position scale
        scale_map = {
            CircuitAction.NORMAL: 1.0,
            CircuitAction.WARNING: 1.0,
            CircuitAction.REDUCE: self.reduce_factor,
            CircuitAction.HEDGE: self.hedge_factor,
            CircuitAction.HALT: 0.0,
        }

        return self._build_state(action, dd, position_scale=scale_map[action])

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._events.clear()
        self._is_halted = False
        self._cooldown_remaining = 0
        self._peak_drawdown = 0.0
        self._last_action = CircuitAction.NORMAL

    @property
    def events(self) -> List[CircuitEvent]:
        """All recorded circuit breaker events."""
        return list(self._events)

    @property
    def trigger_count(self) -> dict:
        """Count of events by action type."""
        counts: dict = {}
        for event in self._events:
            key = event.action.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _record_event(
        self, action: CircuitAction, drawdown: float, message: str
    ) -> None:
        self._events.append(
            CircuitEvent(
                timestamp=datetime.now(timezone.utc),
                action=action,
                drawdown=drawdown,
                message=message,
            )
        )

    def _build_state(
        self, action: CircuitAction, drawdown: float, position_scale: float
    ) -> CircuitState:
        return CircuitState(
            current_action=action,
            drawdown=drawdown,
            peak_drawdown=self._peak_drawdown,
            is_halted=self._is_halted,
            cooldown_remaining=self._cooldown_remaining,
            events=list(self._events),
            position_scale=position_scale,
        )


def simulate_circuit_breaker(
    drawdown_series: list,
    warning_threshold: float = 0.05,
    reduce_threshold: float = 0.10,
    hedge_threshold: float = 0.15,
    halt_threshold: float = 0.20,
    cooldown_periods: int = 10,
) -> List[CircuitState]:
    """Run a circuit breaker simulation over a series of drawdown values.

    Args:
        drawdown_series: List of drawdown values (positive fractions).
        warning_threshold: Warning level.
        reduce_threshold: Position reduction level.
        hedge_threshold: Hedging level.
        halt_threshold: Trading halt level.
        cooldown_periods: Periods to wait after halt.

    Returns:
        List of CircuitState for each period.

    Example:
        >>> drawdowns = [0.01, 0.03, 0.06, 0.12, 0.18, 0.22, 0.15, 0.08, 0.03]
        >>> states = simulate_circuit_breaker(drawdowns)
        >>> actions = [s.current_action.value for s in states]
        >>> assert "halt" in actions
    """
    breaker = DrawdownCircuitBreaker(
        warning_threshold=warning_threshold,
        reduce_threshold=reduce_threshold,
        hedge_threshold=hedge_threshold,
        halt_threshold=halt_threshold,
        cooldown_periods=cooldown_periods,
    )

    states = []
    for dd in drawdown_series:
        states.append(breaker.update(dd))

    return states
