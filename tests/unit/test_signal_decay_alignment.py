"""Tests that signal decay measurements keep signals aligned to their returns.

Nulls used to be dropped from the signal series and the return series
independently. One missing signal shifted every later signal against the wrong
return, so the information coefficient was computed on scrambled pairs while
still looking like a clean result.
"""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.signal_decay import (
    _align_pair,
    detect_signal_staleness,
    measure_signal_decay,
    rolling_signal_ic,
)


def _with_hole(values, index):
    """The same series as a Python list, with one entry blanked out to null."""
    holed = [float(v) for v in values]
    holed[index] = None
    return pl.Series(holed, dtype=pl.Float64)


def _perfect_signal(n, seed=7):
    """A signal that is exactly the sign of the next period's return."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.02, n)
    signals = np.concatenate([np.sign(returns[1:]), [0.0]])
    return signals, returns


def test_align_pair_drops_the_same_rows_from_both_series():
    signals = pl.Series("signal", [1.0, None, 3.0, 4.0])
    returns = pl.Series("return", [0.1, 0.2, None, 0.4])

    sig, ret = _align_pair(signals, returns)

    assert sig.tolist() == [1.0, 4.0]
    assert ret.tolist() == pytest.approx([0.1, 0.4])


def test_align_pair_truncates_to_the_shorter_series():
    sig, ret = _align_pair(pl.Series([1.0, 2.0, 3.0]), pl.Series([0.1, 0.2]))

    assert len(sig) == len(ret) == 2


def test_align_pair_accepts_numpy_input():
    sig, ret = _align_pair(np.array([1.0, np.nan, 3.0]), np.array([0.1, 0.2, 0.3]))

    assert sig.tolist() == [1.0, 3.0]
    assert ret.tolist() == pytest.approx([0.1, 0.3])


def test_align_pair_handles_integer_signals_with_nulls():
    signals = pl.Series("signal", [1, None, -1], dtype=pl.Int64)
    returns = pl.Series("return", [0.1, 0.2, 0.3])

    sig, ret = _align_pair(signals, returns)

    assert sig.tolist() == [1.0, -1.0]
    assert ret.tolist() == pytest.approx([0.1, 0.3])


def test_a_null_signal_does_not_scramble_the_remaining_pairs():
    signals, returns = _perfect_signal(240)

    clean = measure_signal_decay(
        pl.Series(signals), pl.Series(returns), horizons=[1]
    )

    # Blank out one signal early in the series. Everything after it used to be
    # paired with the previous period's return.
    holed_curve = measure_signal_decay(
        _with_hole(signals, 5), pl.Series(returns), horizons=[1]
    )

    # A sign-only signal caps out below 1.0 against continuous returns.
    assert clean.ic_values[0] > 0.8
    # One dropped observation out of 240 should barely move the IC.
    assert holed_curve.ic_values[0] == pytest.approx(clean.ic_values[0], abs=0.05)


def test_rolling_ic_survives_a_null_signal():
    signals, returns = _perfect_signal(300)
    ic_df = rolling_signal_ic(_with_hole(signals, 10), pl.Series(returns), window=60)

    assert ic_df.height > 0
    assert ic_df["ic"].mean() > 0.8


def test_staleness_detection_survives_a_null_signal():
    signals, returns = _perfect_signal(400)
    result = detect_signal_staleness(
        _with_hole(signals, 3),
        pl.Series(returns),
        recent_window=60,
        baseline_window=252,
    )

    # A signal that still predicts perfectly is not stale.
    assert result.is_stale is False
    assert result.current_ic > 0.8


def test_null_returns_are_dropped_with_their_signals():
    signals, returns = _perfect_signal(240)
    curve = measure_signal_decay(
        pl.Series(signals), _with_hole(returns, 7), horizons=[1]
    )

    assert curve.ic_values[0] > 0.8
