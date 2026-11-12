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


def test_first_period_loss_counts_in_max_drawdown():
    import pytest
    import polars as pl
    from imst_quant.utils.equity_curve import build_equity_curve

    curve = build_equity_curve(pl.Series([-0.10, 0.05]), initial_capital=100.0)
    assert curve["drawdown_pct"].min() == pytest.approx(-0.10)


def test_build_equity_curve_bad_periods_and_wipeout():
    import polars as pl
    import pytest

    from imst_quant.utils.equity_curve import build_equity_curve, equity_curve_statistics

    curve = build_equity_curve(pl.Series([0.1, None, float("nan"), 0.1]), initial_capital=100.0)
    assert curve["equity"].to_list() == pytest.approx([110.0, 110.0, 110.0, 121.0])
    assert curve["returns"].len() == 4

    # -150% then -50%: equity must stay at 0, not flip to +25.
    wiped = build_equity_curve(pl.Series([-1.5, -0.5, 0.2]), initial_capital=100.0)
    assert wiped["equity"].to_list() == [0.0, 0.0, 0.0]
    stats = equity_curve_statistics(pl.Series([-1.5, -0.5]), initial_capital=100.0)
    assert stats.final_value == 0.0 and stats.cagr == -1.0 and stats.max_drawdown == -1.0
