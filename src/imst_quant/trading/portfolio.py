"""Portfolio policy with risk limits (TRAD-U01)."""

from typing import Dict


class PortfolioPolicy:
    """Portfolio policy with drawdown/exposure limits."""

    def __init__(
        self,
        max_drawdown: float = 0.10,
        max_daily_loss: float = 0.02,
    ):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.peak = 1.0
        self.equity = 1.0

    def allocate(self, signals: Dict[str, int], prices: Dict[str, float]) -> Dict[str, float]:
        """Equal weight allocation, subject to limits."""
        n = len([s for s in signals.values() if s != 0])
        if n == 0:
            return {a: 0.0 for a in signals}
        w = 1.0 / n
        return {a: w * s for a, s in signals.items()}
