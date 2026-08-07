"""Tests for trade performance metrics, focused on nulls and one-sided records."""

import math

import polars as pl
import pytest

from imst_quant.utils.trade_performance import (
    calculate_average_win_loss,
    calculate_consecutive_streaks,
    calculate_expectancy,
    calculate_profit_factor,
    calculate_r_multiples,
    calculate_trade_metrics,
    calculate_win_rate,
)

PNL = [100.0, -50.0, 150.0, -30.0, 200.0]


def test_win_rate_matches_documented_example():
    assert calculate_win_rate(pl.Series("pnl", PNL)) == pytest.approx(0.60)


def test_profit_factor_matches_documented_example():
    assert calculate_profit_factor(pl.Series("pnl", PNL)) == pytest.approx(450 / 80)


def test_expectancy_matches_documented_example():
    assert calculate_expectancy(pl.Series("pnl", PNL)) == pytest.approx(74.0)


def test_metrics_accept_a_dataframe_and_a_series_alike():
    frame = pl.DataFrame({"pnl": PNL})

    assert calculate_win_rate(frame) == calculate_win_rate(frame["pnl"])
    assert calculate_profit_factor(frame) == calculate_profit_factor(frame["pnl"])
    assert calculate_expectancy(frame) == calculate_expectancy(frame["pnl"])


def test_null_trades_do_not_dilute_the_win_rate():
    """A null is an unrecorded trade, not a loss."""
    with_nulls = pl.Series("pnl", [100.0, None, -50.0, 150.0, None, -30.0, 200.0])

    assert calculate_win_rate(with_nulls) == pytest.approx(0.60)


def test_streaks_survive_null_trades():
    """Comparing a null against zero used to raise TypeError mid-loop."""
    streaks = calculate_consecutive_streaks(pl.Series("pnl", [100.0, 50.0, None, -30.0, -20.0]))

    assert streaks == {"max_win_streak": 2, "max_loss_streak": 2}


def test_streaks_match_documented_example():
    streaks = calculate_consecutive_streaks(pl.Series("pnl", [100, 50, -30, -20, 150, 200, -10]))

    assert streaks["max_win_streak"] == 2
    assert streaks["max_loss_streak"] == 2


def test_breakeven_trades_end_a_streak():
    streaks = calculate_consecutive_streaks(pl.Series("pnl", [100.0, 0.0, 100.0]))

    assert streaks["max_win_streak"] == 1


def test_expectancy_is_zero_when_every_trade_is_null():
    all_null = pl.Series("pnl", [None, None], dtype=pl.Float64)

    assert calculate_expectancy(all_null) == 0.0
    assert calculate_win_rate(all_null) == 0.0
    assert calculate_profit_factor(all_null) == 0.0


def test_profit_factor_is_infinite_without_losses():
    assert math.isinf(calculate_profit_factor(pl.Series("pnl", [10.0, 20.0])))


def test_profit_factor_is_zero_without_profits():
    assert calculate_profit_factor(pl.Series("pnl", [-10.0, -20.0])) == 0.0


def test_average_win_loss_reports_positive_magnitudes():
    metrics = calculate_average_win_loss(pl.Series("pnl", PNL))

    assert metrics["avg_win"] == pytest.approx(150.0)
    assert metrics["avg_loss"] == pytest.approx(40.0)
    assert metrics["win_loss_ratio"] == pytest.approx(3.75)


def test_r_multiples_match_documented_example():
    r_mults = calculate_r_multiples(pl.Series("pnl", [100, -50, 150]), pl.Series("risk", [50, 50, 50]))

    assert r_mults.to_list() == [2.0, -1.0, 3.0]


def test_r_multiples_null_out_a_zero_risk_trade():
    r_mults = calculate_r_multiples(pl.Series("pnl", [100.0, 50.0]), pl.Series("risk", [50.0, 0.0]))

    assert r_mults.to_list() == [2.0, None]


def test_r_multiples_reject_a_length_mismatch():
    with pytest.raises(ValueError):
        calculate_r_multiples(pl.Series("pnl", [1.0, 2.0]), pl.Series("risk", [1.0]))


def test_trade_metrics_cover_the_documented_example():
    trades = pl.DataFrame({"pnl": [100.0, -50.0, 150.0, -30.0, 200.0, -80.0], "risk": [50.0] * 6})

    metrics = calculate_trade_metrics(trades)

    assert metrics["total_trades"] == 6
    assert metrics["winning_trades"] == 3
    assert metrics["losing_trades"] == 3
    assert metrics["total_pnl"] == pytest.approx(290.0)
    assert metrics["max_win"] == pytest.approx(200.0)
    assert metrics["max_loss"] == pytest.approx(80.0)
    assert metrics["avg_r_multiple"] == pytest.approx(290.0 / 6 / 50.0)


def test_trade_metrics_report_no_loss_on_an_all_winning_record():
    """pnl.min() here is the smallest *profit*, which is not a loss at all."""
    metrics = calculate_trade_metrics(pl.DataFrame({"pnl": [100.0, 50.0, 200.0]}))

    assert metrics["max_loss"] == 0.0
    assert metrics["max_win"] == pytest.approx(200.0)
    assert metrics["losing_trades"] == 0


def test_trade_metrics_report_no_win_on_an_all_losing_record():
    metrics = calculate_trade_metrics(pl.DataFrame({"pnl": [-100.0, -50.0, -200.0]}))

    assert metrics["max_win"] == 0.0
    assert metrics["max_loss"] == pytest.approx(200.0)
    assert metrics["win_rate"] == 0.0


def test_trade_metrics_ignore_null_trades():
    metrics = calculate_trade_metrics(pl.DataFrame({"pnl": [100.0, None, -50.0, None]}))

    assert metrics["total_trades"] == 2
    assert metrics["total_pnl"] == pytest.approx(50.0)


def test_trade_metrics_keep_pnl_and_risk_aligned_across_nulls():
    """Dropping nulls per-column would pair the wrong risk with each trade."""
    trades = pl.DataFrame({"pnl": [100.0, None, 150.0], "risk": [50.0, 50.0, 50.0]})

    assert calculate_trade_metrics(trades)["avg_r_multiple"] == pytest.approx(2.5)


def test_trade_metrics_handle_an_empty_frame():
    metrics = calculate_trade_metrics(pl.DataFrame({"pnl": []}, schema={"pnl": pl.Float64}))

    assert metrics["total_trades"] == 0
    assert metrics["profit_factor"] == 0.0


def test_trade_metrics_handle_an_all_null_frame():
    trades = pl.DataFrame({"pnl": [None, None]}, schema={"pnl": pl.Float64})

    assert calculate_trade_metrics(trades)["total_trades"] == 0
