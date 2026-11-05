import polars as pl

from imst_quant.utils.backtest_summary import calculate_trade_stats, generate_summary_report


def test_all_losing_trades_have_no_largest_win():
    stats = calculate_trade_stats(pl.DataFrame({"pnl": [-30.0, -50.0, -10.0]}))
    assert stats.largest_win == 0.0
    assert stats.largest_loss == 50.0


def test_all_winning_trades_have_no_largest_loss():
    stats = calculate_trade_stats(pl.DataFrame({"pnl": [30.0, 50.0]}))
    assert stats.largest_loss == 0.0
    assert stats.largest_win == 50.0


def test_cagr_is_real_minus_one_when_wiped_out():
    report = generate_summary_report(pl.Series([0.1, -1.5, 0.0, 0.0, 0.0]), periods_per_year=252)
    assert isinstance(report.cagr, float)
    assert report.cagr == -1.0
    report.to_string()  # formatting must not choke on a complex number
