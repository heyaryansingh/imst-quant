"""Calibration tests for signal_quality_score."""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.signal_performance import analyze_signal_performance, signal_quality_score


def _frame(signal, returns):
    return pl.DataFrame({"signal": signal, "returns": returns})


def test_coin_flip_signal_scores_low():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 5000)
    signal = np.sign(rng.normal(size=5000))
    assert signal_quality_score(_frame(signal, returns)) < 30


def test_skilled_signal_scores_high():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 5000)
    # Right direction 70% of the time
    signal = np.where(rng.random(5000) < 0.7, np.sign(returns), -np.sign(returns))
    assert signal_quality_score(_frame(signal, returns)) >= 70


def test_perfect_signal_scores_full_marks():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    assert signal_quality_score(_frame(np.sign(returns), returns)) >= 75


def test_nan_rows_are_dropped_not_counted_as_wins():
    df = pl.DataFrame({
        "signal": [1.0, 1.0, float("nan"), -1.0, 1.0],
        "returns": [0.02, float("nan"), 0.05, 0.01, None],
    })
    m = analyze_signal_performance(df)
    assert m["total_trades"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_win"] == pytest.approx(0.02)
    assert m["total_pnl"] == pytest.approx(0.01)
    assert all(v == v for v in m.values())  # no NaN anywhere
