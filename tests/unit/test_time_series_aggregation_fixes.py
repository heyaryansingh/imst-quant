import numpy as np
import pandas as pd
from imst_quant.utils.time_series_aggregation import (
    TimeSeriesAggregator, compute_time_weighted_average)


def test_multi_timeframe_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=4, freq="1min")
    agg = TimeSeriesAggregator(pd.DataFrame({"price": [1.0, 2, 3, 100]}, index=idx))
    out = agg.multi_timeframe_features(["2min"], ["max"])
    # first bin [1,2]: its max must not be visible at the first tick
    assert np.isnan(out["price_2min_max"].iloc[0])
    assert out["price_2min_max"].iloc[2] == 2.0


def test_twap_uses_held_price():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"])
    df = pd.DataFrame({"price": [10.0, 20.0, 30.0]}, index=idx)
    # 10 held 0-10, 20 held 10-20 => 15
    assert compute_time_weighted_average(df, freq="1D").iloc[0] == 15.0
