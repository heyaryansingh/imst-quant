"""Tests for signal quality metrics.

Focus on the forward-return construction and the hit rate, both of which
silently misreport signal quality if they are wrong.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.signal_quality_metrics import (
    _forward_return_expr,
    analyze_signal_decay,
    calculate_signal_metrics,
)


def _panel(signals, returns, assets=None, start=dt.date(2024, 1, 1)):
    n = len(signals)
    data = {
        "date": [start + dt.timedelta(days=i) for i in range(n)],
        "signal": list(signals),
        "return_1d": list(returns),
    }
    if assets is not None:
        data["asset_id"] = list(assets)
    return pl.DataFrame(data)


class TestForwardReturnExpr:
    def test_accumulates_the_whole_window(self):
        df = pl.DataFrame({"r": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = df.with_columns(_forward_return_expr("r", 2))

        # Row 0 must see r1 + r2, not r2 alone.
        assert out["forward_return"].to_list()[:3] == [5.0, 7.0, 9.0]

    def test_uses_no_contemporaneous_information(self):
        df = pl.DataFrame({"r": [10.0, 1.0, 1.0, 1.0]})
        out = df.with_columns(_forward_return_expr("r", 1))

        # The big move at row 0 must not appear in row 0's forward return.
        assert out["forward_return"].to_list()[0] == 1.0

    def test_tail_rows_are_null(self):
        df = pl.DataFrame({"r": [1.0, 2.0, 3.0]})
        out = df.with_columns(_forward_return_expr("r", 2))

        assert out["forward_return"].to_list()[-2:] == [None, None]

    def test_windows_do_not_straddle_assets(self):
        df = pl.DataFrame(
            {"asset_id": ["A", "A", "A", "B", "B", "B"], "r": [1.0, 2, 3, 100, 200, 300]}
        )
        out = df.with_columns(_forward_return_expr("r", 2, group_by="asset_id"))

        values = out["forward_return"].to_list()
        assert values[0] == 5.0  # 2 + 3, never reaching into B
        assert values[1] is None
        assert values[3] == 500.0  # 200 + 300


class TestHitRate:
    def test_flat_signals_are_not_counted_as_hits(self):
        # One directional call, and it is wrong. The two flat bars in between
        # used to count as hits (sign 0 == sign 0), lifting this to 2/3.
        signals = [1.0, 0.0, 0.0, 0.0]
        returns = [0.0, -0.02, 0.0, 0.0]
        metrics = calculate_signal_metrics(_panel(signals, returns), forward_periods=1)

        assert metrics["hit_rate"] == 0.0

    def test_sparse_signal_is_scored_on_its_calls_only(self):
        # Two calls, one right and one wrong, surrounded by flat bars.
        signals = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        returns = [0.0, 0.03, 0.0, 0.0, 0.04, 0.0]
        metrics = calculate_signal_metrics(_panel(signals, returns), forward_periods=1)

        assert metrics["hit_rate"] == pytest.approx(0.5)

    def test_wrong_directional_calls_lower_the_rate(self):
        signals = [1.0, 1.0, 1.0, 1.0, 1.0]
        returns = [0.01, -0.01, 0.01, -0.01, 0.01]
        metrics = calculate_signal_metrics(_panel(signals, returns), forward_periods=1)

        # Forward returns for the first four rows: -1%, +1%, -1%, +1%.
        assert metrics["hit_rate"] == pytest.approx(0.5)

    def test_no_directional_calls_returns_zero(self):
        metrics = calculate_signal_metrics(
            _panel([0.0] * 5, [0.0] * 5), forward_periods=1
        )

        assert metrics["hit_rate"] == 0.0


class TestSignalMetrics:
    def test_perfect_signal_has_positive_ic(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 100)
        # A signal that knows the next 3-period return exactly.
        forward = np.array(
            [returns[i + 1 : i + 4].sum() for i in range(97)] + [0.0, 0.0, 0.0]
        )
        metrics = calculate_signal_metrics(
            _panel(forward, returns), forward_periods=3
        )

        assert metrics["ic"] > 0.99

    def test_rejects_non_positive_forward_periods(self):
        with pytest.raises(ValueError, match="forward_periods"):
            calculate_signal_metrics(_panel([1.0, 2.0], [0.01, 0.02]), forward_periods=0)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="not found"):
            calculate_signal_metrics(_panel([1.0], [0.01]), signal_col="absent")

    def test_short_panel_returns_finite_metrics(self):
        metrics = calculate_signal_metrics(_panel([1.0, 2.0], [0.01, 0.02]))

        assert all(np.isfinite(v) for v in metrics.values())

    def test_single_usable_row_is_finite(self):
        metrics = calculate_signal_metrics(
            _panel([1.0, 2.0], [0.01, 0.02]), forward_periods=1
        )

        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["signal_std"] == 0.0
        assert metrics["sharpe"] == 0.0


class TestSignalDecay:
    def test_decay_curve_covers_every_horizon(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, 60)
        decay = analyze_signal_decay(
            _panel(rng.normal(0, 1, 60), returns), max_horizon=5
        )

        assert sorted(decay) == [1, 2, 3, 4, 5]
        assert all(-1.0 <= ic <= 1.0 for ic in decay.values())

    def test_signal_that_predicts_the_cumulative_move_scores_highest_at_its_horizon(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, 200)
        cumulative_4d = np.array(
            [returns[i + 1 : i + 5].sum() for i in range(196)] + [0.0] * 4
        )
        decay = analyze_signal_decay(_panel(cumulative_4d, returns), max_horizon=6)

        assert decay[4] == max(decay.values())
        assert decay[4] > decay[1]
