import math

import polars as pl

from imst_quant.utils.equity_curve import equity_curve_statistics, rolling_cagr


def test_wipeout_reports_minus_100pct_cagr():
    stats = equity_curve_statistics(pl.Series([0.1, -1.0, 0.0]))
    assert stats.cagr == -1.0
    assert stats.calmar_ratio == -1.0


def test_rolling_cagr_wipeout_window():
    vals = rolling_cagr(pl.Series([0.1, -1.0, 0.0]), window=2).to_list()
    assert math.isnan(vals[0])
    assert vals[1:] == [-1.0, -1.0]


def test_missing_returns_are_skipped():
    clean = equity_curve_statistics(pl.Series([0.01, 0.02]))
    for dirty in ([0.01, None, 0.02], [0.01, float("nan"), 0.02], [0.01, float("inf"), 0.02]):
        stats = equity_curve_statistics(pl.Series(dirty))
        assert stats.num_periods == 2
        assert math.isclose(stats.total_return, clean.total_return)
        assert math.isfinite(stats.sharpe_ratio)
