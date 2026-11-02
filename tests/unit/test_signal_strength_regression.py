import polars as pl

from imst_quant.utils.signal_strength import SignalStrengthAnalyzer


def test_trend_persistence_never_exceeds_100():
    prices = pl.Series([float(i) for i in range(30)])
    assert SignalStrengthAnalyzer().score_trend_persistence(prices) == 100.0
