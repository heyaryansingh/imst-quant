"""Tests for Kelly Criterion sizing, focused on degenerate trade records."""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.kelly_criterion import (
    calculate_kelly_metrics,
    fractional_kelly,
    kelly_formula,
    kelly_from_sharpe,
    kelly_from_trades,
    kelly_portfolio,
    kelly_win_rate,
    optimal_f,
)

WINNERS_AND_LOSERS = [100.0, -50.0, 150.0, -30.0, 200.0, -80.0]


def test_kelly_formula_agrees_with_the_win_rate_form():
    """(pW - qL) / (WL) reduces to W - (1-W)/R when the loss is 1 unit."""
    assert kelly_formula(probability=0.6, win_amount=2.0, loss_amount=1.0) == pytest.approx(0.40)
    assert kelly_formula(0.6, 2.0, 1.0) == pytest.approx(kelly_win_rate(0.6, 2.0))


def test_kelly_win_rate_matches_documented_example():
    assert kelly_win_rate(win_rate=0.55, win_loss_ratio=2.0) == pytest.approx(0.325)


def test_fractional_kelly_matches_module_docstring():
    assert fractional_kelly(0.55, 2.0, fraction=0.25) == pytest.approx(0.08125)


def test_fractional_kelly_floors_a_negative_edge_at_zero():
    assert fractional_kelly(win_rate=0.2, win_loss_ratio=1.0) == 0.0


def test_kelly_from_sharpe_is_zero_for_a_losing_strategy():
    assert kelly_from_sharpe(sharpe_ratio=-1.0) == 0.0


def test_optimal_f_is_between_zero_and_one():
    assert 0.0 < optimal_f(WINNERS_AND_LOSERS) <= 1.0


def test_optimal_f_is_zero_for_a_losing_record():
    """No f beats sitting out when the edge is negative, so risk nothing."""
    assert optimal_f([10.0, -50.0, 20.0, -80.0, 5.0, -40.0]) == 0.0


def test_optimal_f_risks_everything_when_nothing_ever_lost():
    """Scaling by np.min() would divide by the smallest *profit* here."""
    assert optimal_f([100.0, 50.0, 20.0]) == 1.0


def test_optimal_f_is_zero_without_trades():
    assert optimal_f([]) == 0.0
    assert optimal_f(pl.Series("pnl", [], dtype=pl.Float64)) == 0.0


def test_optimal_f_is_zero_when_nothing_moved():
    assert optimal_f([0.0, 0.0, 0.0]) == 0.0


def test_optimal_f_accepts_series_lists_and_arrays_alike():
    expected = optimal_f(WINNERS_AND_LOSERS)

    assert optimal_f(np.array(WINNERS_AND_LOSERS)) == expected
    assert optimal_f(pl.Series("pnl", WINNERS_AND_LOSERS)) == expected


def test_kelly_from_trades_handles_an_all_losing_record():
    """A win/loss ratio of 0 used to raise ValueError out of kelly_win_rate."""
    trades = pl.DataFrame({"pnl": [-10.0, -20.0, -5.0]})

    assert kelly_from_trades(trades) == 0.0


def test_kelly_from_trades_handles_an_all_winning_record():
    trades = pl.DataFrame({"pnl": [10.0, 20.0, 5.0]})

    assert kelly_from_trades(trades, fraction=0.5) == pytest.approx(0.5)


def test_kelly_from_trades_is_zero_without_trades():
    assert kelly_from_trades(pl.DataFrame({"pnl": []}, schema={"pnl": pl.Float64})) == 0.0


def test_kelly_from_trades_ignores_nulls():
    trades = pl.DataFrame({"pnl": [100.0, None, -50.0, 150.0, None, -30.0, 200.0, -80.0]})

    assert kelly_from_trades(trades) == pytest.approx(
        kelly_from_trades(pl.DataFrame({"pnl": WINNERS_AND_LOSERS}))
    )


def test_kelly_metrics_scale_linearly_with_the_fraction():
    metrics = calculate_kelly_metrics(pl.DataFrame({"pnl": WINNERS_AND_LOSERS}))

    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["kelly_0.25"] == pytest.approx(metrics["full_kelly"] * 0.25)
    assert metrics["kelly_1.0"] == pytest.approx(metrics["full_kelly"])


def test_kelly_metrics_handle_an_all_losing_record():
    metrics = calculate_kelly_metrics(pl.DataFrame({"pnl": [-10.0, -20.0, -5.0]}))

    assert metrics["win_rate"] == 0.0
    assert metrics["win_loss_ratio"] == 0.0
    assert metrics["full_kelly"] == 0.0
    assert metrics["kelly_0.25"] == 0.0
    assert metrics["optimal_f"] == 0.0


def test_kelly_metrics_saturate_when_nothing_ever_lost():
    metrics = calculate_kelly_metrics(pl.DataFrame({"pnl": [10.0, 20.0, 5.0]}))

    assert metrics["full_kelly"] == pytest.approx(1.0)
    assert metrics["kelly_0.5"] == pytest.approx(0.5)


def test_kelly_metrics_handle_an_empty_frame():
    metrics = calculate_kelly_metrics(pl.DataFrame({"pnl": []}, schema={"pnl": pl.Float64}))

    assert metrics["full_kelly"] == 0.0
    assert metrics["kelly_1.0"] == 0.0


def test_kelly_portfolio_solves_the_covariance_system():
    expected_returns = np.array([0.10, 0.08])
    cov = np.array([[0.04, 0.01], [0.01, 0.03]])

    weights = kelly_portfolio(expected_returns, cov)

    assert cov @ weights == pytest.approx(expected_returns)


def test_kelly_portfolio_falls_back_to_equal_weights_when_singular():
    weights = kelly_portfolio(np.array([0.1, 0.1]), np.zeros((2, 2)))

    assert weights == pytest.approx(np.array([0.5, 0.5]))


def test_kelly_portfolio_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        kelly_portfolio(np.array([0.1, 0.1, 0.1]), np.eye(2))
