"""Tests for signal_decay alignment and hit-rate pairing."""

import numpy as np
import polars as pl

from imst_quant.utils.signal_decay import detect_signal_staleness, measure_signal_decay


def test_null_signal_does_not_shift_returns():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 300)
    sig = np.empty(300)
    sig[:-1] = ret[1:]  # perfect next-bar predictor
    sig[-1] = 0.0
    sig_list = sig.tolist()
    sig_list[50] = None  # one gap must not misalign everything after it
    curve = measure_signal_decay(pl.Series("s", sig_list), pl.Series("r", ret), horizons=[1])
    assert curve.ic_values[0] > 0.99


def test_hit_rate_uses_next_bar_return():
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, 400)
    sig = np.append(np.sign(ret[1:]), 1.0)  # always right about the next bar
    result = detect_signal_staleness(sig, ret, recent_window=60)
    assert result.recent_hit_rate == 1.0
    assert not result.is_stale
