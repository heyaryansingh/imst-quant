"""Tests for NaN handling in streak_analysis."""

import polars as pl

from imst_quant.utils.streak_analysis import analyze_streaks, identify_streaks


def test_nan_returns_are_dropped_not_counted_as_losses():
    s = pl.Series([0.01, float("nan"), 0.02, -0.01])
    stats = analyze_streaks(s)
    assert stats.total_periods == 3
    assert stats.win_count == 2
    assert stats.max_win_streak == 2


def test_nan_does_not_poison_streak_returns():
    wins, _ = identify_streaks(pl.Series([0.01, float("nan"), 0.02]))
    assert wins[0].length == 2
    assert abs(wins[0].total_return - 0.03) < 1e-12
