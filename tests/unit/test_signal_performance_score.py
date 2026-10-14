"""Calibration tests for signal_quality_score."""

import numpy as np
import polars as pl

from imst_quant.utils.signal_performance import signal_quality_score


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
