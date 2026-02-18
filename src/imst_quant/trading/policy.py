"""Trading policies (TRAD-01, TRAD-02)."""

from typing import List


class FixedThresholdPolicy:
    """
    Fixed threshold - LEAKAGE DEMO.
    Uses future data to optimize threshold. For demo only.
    """

    def __init__(self, thresholds: List[float] | None = None):
        self.thresholds = thresholds or [0.5, 0.6, 0.7]
        self.optimal_threshold = 0.5

    def fit(self, predictions: List[float], returns: List[float]) -> float:
        """Find best threshold (LOOKAHEAD - uses future returns)."""
        best, best_sharpe = 0.5, -1e9
        for th in self.thresholds:
            pnl = 0.0
            for pred, ret in zip(predictions, returns):
                if pred > th:
                    pnl += ret
                elif pred < (1 - th):
                    pnl -= ret
            sharpe = pnl / (1e-8 + abs(pnl))
            if pnl > best_sharpe:
                best_sharpe = pnl
                best = th
        self.optimal_threshold = best
        return best

    def signal(self, prediction: float) -> int:
        """Return 1 (long), -1 (short), 0 (neutral)."""
        if prediction > self.optimal_threshold:
            return 1
        if prediction < (1 - self.optimal_threshold):
            return -1
        return 0


class DynamicThresholdPolicy:
    """Dynamic threshold with incremental updates (TRAD-02)."""

    def __init__(self, initial: float = 0.5, lr: float = 0.1):
        self.threshold = initial
        self.lr = lr
        self.errors = []

    def update(self, prediction: float, actual_return: float) -> None:
        """Update threshold based on prediction error."""
        if prediction > 0.5 and actual_return > 0:
            pass
        elif prediction < 0.5 and actual_return < 0:
            pass
        else:
            self.errors.append(abs(prediction - 0.5))
        if len(self.errors) >= 5:
            adj = sum(self.errors[-5:]) / 5
            self.threshold = self.lr * (0.5 + adj) + (1 - self.lr) * self.threshold
            self.threshold = max(0.2, min(0.8, self.threshold))
            self.errors = []

    def signal(self, prediction: float) -> int:
        if prediction > self.threshold:
            return 1
        if prediction < (1 - self.threshold):
            return -1
        return 0
