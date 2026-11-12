"""breakout_signal must hold (0), not go long, on NaN/inf prices."""

import polars as pl

from imst_quant.trading.signals import breakout_signal


def test_nan_close_and_nan_high_hold():
    nan = float("nan")
    df = pl.DataFrame({
        "close": [100.0, 102, 105, nan, 110, 98],
        "high": [101.0, 103, 106, 104, nan, 99],
        "low": [99.0, 101, 104, 102, 108, 97],
    })
    sig = breakout_signal(df, lookback=2)["breakout_signal"].to_list()
    assert sig[:3] == [0, 0, 1]  # warm-up holds, real breakout still fires
    assert sig[3] == 0  # NaN close
    assert sig[5] == 0  # NaN high inside the prior window
