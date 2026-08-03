"""Tests for streak and drawdown-duration analytics."""

import datetime as dt

import polars as pl
import pytest

from imst_quant.utils.trade_analytics import (
    calculate_consecutive_metrics,
    calculate_drawdown_duration,
)


def _trades(pnl):
    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(pnl))]
    return pl.DataFrame({"entry_time": dates, "exit_time": dates, "pnl": pnl})


def test_streaks_over_mixed_run():
    """Used to raise AttributeError: 'Expr' object has no attribute 'cumsum'."""
    metrics = calculate_consecutive_metrics(_trades([10.0, 20.0, -5.0, -3.0, -1.0, 7.0, 8.0, 9.0]))

    assert metrics["max_win_streak"] == 3
    assert metrics["max_loss_streak"] == 3
    assert metrics["current_streak"] == 3


def test_leading_streak_is_not_split_off():
    """shift(1) leaves a null on row 0; it must open a streak, not its own group."""
    metrics = calculate_consecutive_metrics(_trades([5.0, 6.0, 7.0]))

    assert metrics["max_win_streak"] == 3
    assert metrics["current_streak"] == 3


def test_current_streak_is_negative_while_losing():
    metrics = calculate_consecutive_metrics(_trades([5.0, 6.0, -1.0, -2.0]))

    assert metrics["max_win_streak"] == 2
    assert metrics["current_streak"] == -2


def test_empty_trades_return_zero_streaks():
    empty = pl.DataFrame({"entry_time": [], "exit_time": [], "pnl": []})

    assert calculate_consecutive_metrics(empty) == {
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "current_streak": 0,
    }


def test_drawdown_durations():
    equity = pl.DataFrame({"equity": [100.0, 110.0, 105.0, 103.0, 112.0, 108.0]})

    dd = calculate_drawdown_duration(equity)

    assert dd["max_drawdown_duration_days"] == pytest.approx(2.0)
    assert dd["avg_drawdown_duration_days"] == pytest.approx(1.5)
    assert dd["current_drawdown_duration_days"] == pytest.approx(1.0)


def test_no_drawdown_when_monotonically_rising():
    equity = pl.DataFrame({"equity": [100.0, 101.0, 102.0]})

    dd = calculate_drawdown_duration(equity)

    assert dd["max_drawdown_duration_days"] == 0.0
    assert dd["current_drawdown_duration_days"] == 0.0


def test_drawdown_open_from_the_second_bar():
    """A drawdown starting on row 1 must not be merged with the leading peak."""
    equity = pl.DataFrame({"equity": [100.0, 90.0, 95.0]})

    dd = calculate_drawdown_duration(equity)

    assert dd["max_drawdown_duration_days"] == pytest.approx(2.0)
    assert dd["current_drawdown_duration_days"] == pytest.approx(2.0)
