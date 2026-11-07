"""OBV and VWAP skip null/NaN bars instead of being poisoned by them."""

import math

import polars as pl
import pytest

from imst_quant.utils.technical_indicators import obv, vwap


def test_obv_skips_nan_volume_and_nan_close():
    df = pl.DataFrame({
        "close": [10.0, 11.0, 12.0, math.nan, 11.0, 13.0],
        "volume": [100.0, 200.0, math.nan, 50.0, 300.0, 400.0],
    })
    # bar2 (NaN vol) and bar3 (NaN close) add nothing; bar4 compares to 11.0
    # (last good close, bar1) -> flat; bar5 up.
    assert obv(df)["obv"].to_list() == [0.0, 200.0, 200.0, 200.0, 200.0, 600.0]


def test_obv_null_volume():
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, None, 5.0]})
    assert obv(df)["obv"].to_list() == [0.0, 0.0, 5.0]


def test_vwap_skips_bad_bars_and_waits_for_volume():
    df = pl.DataFrame({
        "high": [10.0, 10.0, 20.0, 30.0],
        "low": [10.0, 10.0, 20.0, 30.0],
        "close": [10.0, 10.0, math.nan, 30.0],
        "volume": [0.0, 1.0, 5.0, 1.0],
    })
    out = vwap(df)["vwap"].to_list()
    assert out[0] is None
    assert out[1:] == pytest.approx([10.0, 10.0, 20.0])
