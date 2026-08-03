"""Downside deviation must be RMS shortfall, not the std of losing periods.

The pandas-based Sortino calculations in advanced_risk and performance_tracker
share the definition tested in tests/unit/test_sortino_ratio.py. These tests
pin the same property for those two call sites.
"""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.advanced_risk import tail_risk_metrics
from imst_quant.utils.performance_tracker import PerformanceTracker


class TestAdvancedRiskDownsideDeviation:
    def test_steady_loss_is_not_risk_free(self):
        # Identical losses have zero dispersion but plenty of downside; the
        # old std-of-losers denominator scored this as sortino 0.0.
        result = tail_risk_metrics(pd.Series([-0.05] * 20))
        assert result["downside_deviation"] == pytest.approx(0.05)
        assert result["sortino_ratio"] < 0

    def test_deviation_equals_rms_shortfall(self):
        returns = [0.02, -0.01, 0.03, -0.02]
        expected = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        result = tail_risk_metrics(pd.Series(returns))
        assert result["downside_deviation"] == pytest.approx(expected)

    def test_no_losses_gives_zero_deviation(self):
        result = tail_risk_metrics(pd.Series([0.01] * 10))
        assert result["downside_deviation"] == pytest.approx(0.0)
        assert result["sortino_ratio"] == 0.0

    def test_deeper_losses_raise_the_deviation(self):
        shallow = tail_risk_metrics(pd.Series([0.02, -0.01] * 10))
        deep = tail_risk_metrics(pd.Series([0.02, -0.05] * 10))
        assert deep["downside_deviation"] > shallow["downside_deviation"]


class TestPerformanceTrackerSortino:
    def _tracker_with_returns(self, returns):
        tracker = PerformanceTracker(initial_capital=100_000.0)
        tracker.daily_returns = list(returns)
        # The metrics path needs an equity curve to get past its guard.
        equity = 100_000.0
        for r in returns:
            equity *= 1 + r
            tracker.equity_curve.append(equity)
        tracker.current_capital = equity
        return tracker

    def test_fresh_tracker_exposes_every_metric_key(self):
        """The empty-history path must return the same keys as the full one."""
        fresh = PerformanceTracker(initial_capital=100_000.0).get_current_metrics()
        populated = self._tracker_with_returns([0.01, -0.02, 0.03]).get_current_metrics()
        assert set(fresh) == set(populated)

    def test_steady_loss_reports_negative_sortino(self):
        tracker = self._tracker_with_returns([-0.05] * 20)
        assert tracker.get_current_metrics()["sortino_ratio"] < 0

    def test_sortino_matches_rms_shortfall_definition(self):
        returns = [0.02, -0.01, 0.03, -0.02] * 5
        tracker = self._tracker_with_returns(returns)
        deviation = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        expected = (np.mean(returns) / deviation) * np.sqrt(252)
        assert tracker.get_current_metrics()["sortino_ratio"] == pytest.approx(expected)

    def test_no_returns_gives_zero_sortino(self):
        tracker = self._tracker_with_returns([])
        assert tracker.get_current_metrics()["sortino_ratio"] == 0.0

    def test_no_losing_periods_gives_zero_sortino(self):
        tracker = self._tracker_with_returns([0.01] * 10)
        assert tracker.get_current_metrics()["sortino_ratio"] == 0.0
