"""Tests for `generate_summary_stats`, which used to raise on every input."""

import datetime as dt

import polars as pl
import pytest

from imst_quant.utils.backtest_viz import generate_summary_stats

MIXED = [0.01, -0.02, 0.03, -0.01, 0.02, -0.005]


def _frame(returns):
    return pl.DataFrame({
        "timestamp": [
            dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(returns))
        ],
        "returns": returns,
    })


def test_returns_stats_instead_of_raising():
    stats = generate_summary_stats(_frame(MIXED))

    assert stats["periods"] == len(MIXED)
    assert stats["total_trades"] == len(MIXED)


def test_total_return_compounds():
    stats = generate_summary_stats(_frame(MIXED))

    expected = 1.0
    for r in MIXED:
        expected *= 1 + r

    assert stats["total_return"] == pytest.approx(expected - 1)


def test_win_rate_counts_positive_periods():
    assert generate_summary_stats(_frame(MIXED))["win_rate"] == pytest.approx(0.5)


def test_sortino_uses_downside_only():
    stats = generate_summary_stats(_frame(MIXED))

    # Losses are smaller than gains here, so downside risk is below total
    # volatility and Sortino must clear Sharpe.
    assert stats["sortino_ratio"] > stats["sharpe_ratio"] > 0


def test_single_period_has_no_volatility():
    stats = generate_summary_stats(_frame([0.01]))

    assert stats["volatility"] == 0.0
    assert stats["sharpe_ratio"] == 0.0


def test_total_wipeout_reports_a_total_loss():
    # Compounding a zero equity to a fractional power is undefined.
    stats = generate_summary_stats(_frame([-1.0, 0.0]))

    assert stats["total_return"] == pytest.approx(-1.0)
    assert stats["annualized_return"] == -1.0


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        generate_summary_stats(_frame(MIXED), returns_col="pnl")
