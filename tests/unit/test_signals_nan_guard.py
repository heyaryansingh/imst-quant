"""NaN/null/inf inputs must produce hold (0), never a long signal."""

import math

import polars as pl

from imst_quant.trading.signals import (
    composite_signal,
    momentum_signal,
    prediction_to_signal,
    roc_signal,
)


def test_nan_probability_is_hold():
    df = pl.DataFrame({"prob_up": [0.8, 0.2, math.nan, None]})
    assert prediction_to_signal(df, threshold=0.6)["signal"].to_list() == [1, -1, 0, 0]


def test_bad_prices_do_not_go_long():
    df = pl.DataFrame({"close": [100.0, 0.0, 110.0, math.nan, 115.0]})
    # 110/0 -> inf momentum, NaN price -> NaN momentum: both hold
    assert momentum_signal(df, lookback=1)["momentum_signal"].to_list() == [0, -1, 0, 0, 0]
    assert roc_signal(df, lookback=1, threshold=1.0)["roc_signal"].to_list() == [0, -1, 0, 0, 0]


def test_nan_component_composite_is_hold():
    df = pl.DataFrame({"a": [1.0, math.nan], "b": [1.0, 1.0]})
    out = composite_signal(df, signal_weights={"a": 0.5, "b": 0.5}, threshold=0.3)
    assert out["composite_signal"].to_list() == [1, 0]
