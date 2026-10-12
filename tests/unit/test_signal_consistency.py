"""Tests for signal consistency degradation detection."""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.signal_consistency import detect_signal_degradation


def test_degradation_flags_most_recent_window():
    rng = np.random.default_rng(0)
    signals = rng.choice([-1.0, 1.0], 250)
    returns = signals * 0.01 + rng.normal(0, 0.002, 250)
    returns[200:] = rng.normal(0, 0.01, 50)  # edge vanishes in the final window
    events = detect_signal_degradation(
        pl.Series(signals), returns=pl.Series(returns), window_size=50
    )
    assert 200 in [e["period_start"] for e in events]


def test_always_flipping_lookback_does_not_divide_by_zero():
    assert detect_signal_degradation(pl.Series([1, -1] * 100), window_size=20) == []


def test_constant_window_counts_as_zero_quality():
    signals = [1.0, -1.0] * 100 + [1.0] * 50
    returns = [0.01, -0.01] * 100 + [0.0] * 50
    events = detect_signal_degradation(
        pl.Series(signals), returns=pl.Series(returns), window_size=50
    )
    assert events and events[-1]["quality_current"] == 0.0


def test_rejects_tiny_window():
    with pytest.raises(ValueError):
        detect_signal_degradation(pl.Series([1, -1, 1]), window_size=1)
