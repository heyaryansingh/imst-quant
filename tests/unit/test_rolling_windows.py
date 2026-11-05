"""RSI helpers must run on polars 1.x (Series.clip_min was removed)."""

import polars as pl
import pytest

from imst_quant.utils.multi_timeframe import calculate_timeframe_signal
from imst_quant.utils.rolling_windows import add_rsi


def test_add_rsi_bounds_and_edges():
    rsi = add_rsi(pl.DataFrame({"close": [100.0, 102, 101, 105, 108, 106, 110, 109]}), period=6)["rsi"]
    vals = rsi.drop_nulls().to_list()
    assert all(0 <= v <= 100 for v in vals)
    # Wilder smoothing: after +2 then -1, avg_gain=2*(5/6), avg_loss=1/6 -> RS=10
    assert vals[1] == pytest.approx(100 - 100 / 11)


def test_add_rsi_flat_is_50_and_rising_is_100():
    assert add_rsi(pl.DataFrame({"close": [5.0, 5, 5]}), period=3)["rsi"].to_list()[1:] == [50.0, 50.0]
    assert add_rsi(pl.DataFrame({"close": [1.0, 2, 3]}), period=3)["rsi"].to_list()[1:] == [100.0, 100.0]


def test_momentum_rsi_no_losses_is_100():
    df = pl.DataFrame({"close": [100.0 + i for i in range(30)]})
    out = calculate_timeframe_signal(df, "momentum", lookback=5)
    assert out["rsi"].to_list()[-1] == 100.0


def test_rolling_zscore_flat_window_is_zero():
    from imst_quant.utils.rolling_windows import add_rolling_zscore

    out = add_rolling_zscore(pl.DataFrame({"p": [1.0, 1, 1, 2]}), "p", window=3)["p_zscore3"].to_list()
    assert out[2] == 0.0
    assert out[3] == pytest.approx((2 - 4 / 3) / pl.Series([1.0, 1, 2]).std())
