"""Tests for the polars-based backtest report utilities."""

import datetime as dt

import polars as pl
import pytest

from imst_quant.utils.backtest_report import (
    calculate_performance_stats,
    compare_strategies,
    generate_backtest_report,
    generate_trade_log,
)


@pytest.fixture
def equity_curve() -> pl.DataFrame:
    returns = [0.01, -0.02, 0.03, -0.01, 0.0, 0.02, -0.03, 0.01, 0.01, -0.005]
    equity = [100_000.0]
    for r in returns[1:]:
        equity.append(equity[-1] * (1 + r))
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(returns))]
    return pl.DataFrame({"date": dates, "equity": equity, "returns": returns})


@pytest.fixture
def trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exit_time": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(4)],
            "pnl": [100.0, -50.0, 200.0, -10.0],
        }
    )


def test_report_sections_are_populated(equity_curve, trades):
    """Every section used to raise TypeError from Series.filter(pl.col(...))."""
    report = generate_backtest_report(equity_curve, trades)

    for section in ("summary", "returns_stats", "risk_metrics", "drawdown_stats", "trade_stats"):
        assert report[section], f"{section} came back empty"


def test_returns_stats_counts_signs(equity_curve):
    stats = generate_backtest_report(equity_curve)["returns_stats"]

    assert stats["positive_days"] == 5
    assert stats["negative_days"] == 4  # the 0.0 day counts as neither
    assert stats["positive_pct"] == pytest.approx(0.5)


def test_drawdown_uses_running_max(equity_curve):
    dd = generate_backtest_report(equity_curve)["drawdown_stats"]

    assert dd["max_drawdown"] == pytest.approx(-0.03, abs=1e-9)
    assert dd["max_drawdown_date"] == dt.date(2024, 1, 7)


def test_sortino_uses_rms_shortfall_over_all_periods():
    """A steady loss has zero dispersion but is not a zero-Sortino strategy."""
    returns = [-0.05] * 10
    df = pl.DataFrame(
        {
            "date": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)],
            "equity": [100_000.0 * (0.95 ** i) for i in range(10)],
            "returns": returns,
        }
    )

    risk = generate_backtest_report(df)["risk_metrics"]

    assert risk["downside_deviation"] == pytest.approx(0.05 * (252 ** 0.5))
    assert risk["sortino_ratio"] == pytest.approx(-1.0 * (252 ** 0.5))


def test_sortino_is_zero_without_downside():
    df = pl.DataFrame(
        {
            "date": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)],
            "equity": [100_000.0 * (1.01 ** i) for i in range(5)],
            "returns": [0.01] * 5,
        }
    )

    assert generate_backtest_report(df)["risk_metrics"]["sortino_ratio"] == 0.0


def test_performance_stats_reports_downside_volatility(equity_curve):
    stats = calculate_performance_stats(equity_curve["returns"])

    assert stats["sharpe_ratio"] != 0.0
    assert stats["downside_volatility"] > 0.0


def test_performance_stats_single_loss_has_no_dispersion():
    """One losing period cannot have a sample standard deviation."""
    stats = calculate_performance_stats(pl.Series("returns", [0.01, 0.02, -0.01]))

    assert stats["downside_volatility"] == 0.0


def test_trade_log_accumulates(trades):
    log = generate_trade_log(trades)

    assert log["cumulative_pnl"].to_list() == [100.0, 50.0, 250.0, 240.0]
    assert log["total_wins"].to_list() == [1, 1, 2, 2]
    assert log["running_win_rate"][-1] == pytest.approx(0.5)


def test_compare_strategies_ranks_by_metric(equity_curve):
    flat = equity_curve.with_columns(
        pl.lit(0.0).alias("returns"),
        pl.lit(100_000.0).alias("equity"),
    )

    comparison = compare_strategies({"flat": flat, "live": equity_curve})

    assert comparison["strategy"][0] == "live"
    assert comparison.height == 2
