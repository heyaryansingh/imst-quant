"""Tests for the trailing stop-loss module.

Tests cover:
- ATR-based trailing stop ratcheting (long and short)
- Percentage-based trailing stop math
- Stop trigger detection
"""

import polars as pl
import pytest

from imst_quant.trading.trailing_stop import (
    atr_trailing_stop,
    check_stop_triggered,
    percent_trailing_stop,
)


def test_atr_trailing_stop_long_ratchets_up_only():
    df = pl.DataFrame({
        "close": [100.0, 105.0, 103.0, 110.0, 106.0],
        "atr": [2.0, 2.0, 2.0, 2.0, 2.0],
    })
    result = atr_trailing_stop(df, multiplier=2.0, direction="long")
    stops = result["trailing_stop"].to_list()

    # raw stops would be [96, 101, 99, 106, 102] -> ratcheted long never decreases
    assert stops == [96.0, 101.0, 101.0, 106.0, 106.0]
    for i in range(1, len(stops)):
        assert stops[i] >= stops[i - 1]


def test_atr_trailing_stop_short_ratchets_down_only():
    df = pl.DataFrame({
        "close": [100.0, 95.0, 97.0, 90.0, 94.0],
        "atr": [2.0, 2.0, 2.0, 2.0, 2.0],
    })
    result = atr_trailing_stop(df, multiplier=2.0, direction="short")
    stops = result["trailing_stop"].to_list()

    # raw stops would be [104, 99, 101, 94, 98] -> ratcheted short never increases
    assert stops == [104.0, 99.0, 99.0, 94.0, 94.0]
    for i in range(1, len(stops)):
        assert stops[i] <= stops[i - 1]


def test_percent_trailing_stop_math():
    df = pl.DataFrame({"close": [100.0, 200.0]})
    result = percent_trailing_stop(df, pct=0.05, direction="long")
    stops = result["trailing_stop"].to_list()

    assert stops[0] == pytest.approx(95.0)
    assert stops[1] == pytest.approx(190.0)


def test_percent_trailing_stop_invalid_pct_raises():
    df = pl.DataFrame({"close": [100.0]})
    with pytest.raises(ValueError):
        percent_trailing_stop(df, pct=1.5)


def test_check_stop_triggered_long():
    df = pl.DataFrame({
        "close": [100.0, 95.0, 89.0],
        "trailing_stop": [90.0, 90.0, 90.0],
    })
    result = check_stop_triggered(df, direction="long")
    assert result["stop_triggered"].to_list() == [False, False, True]


def test_check_stop_triggered_short():
    df = pl.DataFrame({
        "close": [100.0, 105.0, 111.0],
        "trailing_stop": [110.0, 110.0, 110.0],
    })
    result = check_stop_triggered(df, direction="short")
    assert result["stop_triggered"].to_list() == [False, False, True]


def test_invalid_direction_raises():
    df = pl.DataFrame({"close": [100.0], "atr": [1.0]})
    with pytest.raises(ValueError):
        atr_trailing_stop(df, direction="sideways")
