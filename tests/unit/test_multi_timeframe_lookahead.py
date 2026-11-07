"""analyze_multi_timeframe must not use prices after each row's timestamp."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from imst_quant.utils.multi_timeframe import analyze_multi_timeframe


def _frame(n=600):
    # Sawtooth: rises for 11 bars then drops hard on the 12th, so the last
    # bar of each hour flips the hourly trend.
    close = 100 + np.array([(i % 12) - 30 * (i % 12 == 11) for i in range(n)], dtype=float) + 0.01 * np.arange(n)
    dates = [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame({
        "date": dates, "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close, "volume": [1000.0] * n,
    })


def test_signal_at_t_only_uses_data_up_to_t():
    df = _frame()
    full = analyze_multi_timeframe(df, timeframes=[15, 60], lookback=5)
    for cut in range(200, 600, 7):
        part = analyze_multi_timeframe(df.head(cut + 1), timeframes=[15, 60], lookback=5)
        ts = df["date"][cut]
        a = full.filter(pl.col("date") == ts).select("tf_15_signal", "tf_60_signal").row(0)
        b = part.filter(pl.col("date") == ts).select("tf_15_signal", "tf_60_signal").row(0)
        assert a == b, f"future data leaked into row {cut}: {a} vs {b}"
