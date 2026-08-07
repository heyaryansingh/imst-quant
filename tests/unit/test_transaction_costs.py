"""Tests for transaction cost estimation, focused on bad market-data inputs."""

import polars as pl
import pytest

from imst_quant.utils.transaction_costs import (
    analyze_turnover_costs,
    batch_cost_analysis,
    estimate_commission,
    estimate_market_impact,
    estimate_slippage,
    estimate_total_cost,
    optimal_execution_schedule,
)


def test_commission_applies_the_rate():
    assert estimate_commission(10000.0, commission_rate=0.0005) == pytest.approx(5.0)


def test_commission_respects_the_floor_and_the_cap():
    assert estimate_commission(100.0, commission_rate=0.0005, min_commission=1.0) == 1.0
    assert estimate_commission(1_000_000.0, max_commission=50.0) == 50.0


def test_slippage_is_half_the_spread_plus_impact():
    slippage = estimate_slippage(10_000, 1_000_000, spread_bps=5.0, volatility=0.02)

    assert slippage > 2.5


def test_slippage_grows_with_order_size():
    small = estimate_slippage(10_000, 1_000_000)
    large = estimate_slippage(500_000, 1_000_000)

    assert large > small


def test_market_impact_scales_with_participation():
    assert estimate_market_impact(100_000, 1_000_000) == pytest.approx(
        2 * estimate_market_impact(50_000, 1_000_000)
    )


def test_total_cost_sums_its_three_parts():
    total = estimate_total_cost(25_000, 1_500_000)
    parts = (
        (estimate_commission(25_000) / 25_000) * 10_000.0
        + estimate_slippage(25_000, 1_500_000)
        + estimate_market_impact(25_000, 1_500_000)
    )

    assert total == pytest.approx(parts)


@pytest.mark.parametrize("bad_volume", [0, 0.0, -1000.0, None, float("nan")])
def test_zero_volume_is_rejected_by_name(bad_volume):
    """A halted symbol reports zero volume; that used to be ZeroDivisionError."""
    with pytest.raises(ValueError, match="avg_daily_volume"):
        estimate_slippage(10_000, bad_volume)

    with pytest.raises(ValueError, match="avg_daily_volume"):
        estimate_market_impact(10_000, bad_volume)


@pytest.mark.parametrize("bad_size", [0, -5000.0, None])
def test_zero_order_size_is_rejected_by_name(bad_size):
    with pytest.raises(ValueError, match="order_size"):
        estimate_total_cost(bad_size, 1_000_000)


def test_turnover_costs_scale_with_turnover():
    costs = analyze_turnover_costs(1_000_000, 2.0, avg_cost_bps=15.0)

    assert costs["annual_cost_usd"] == pytest.approx(3000.0)
    assert costs["annual_cost_pct"] == pytest.approx(0.003)
    assert costs["daily_avg_cost_usd"] == pytest.approx(3000.0 / 252)
    assert costs["breakeven_alpha"] == pytest.approx(costs["annual_cost_pct"])


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"portfolio_value": 0.0}, "portfolio_value"),
        ({"periods_per_year": 0}, "periods_per_year"),
    ],
)
def test_turnover_costs_reject_zero_divisors(kwargs, expected):
    args = {"portfolio_value": 1_000_000.0, "turnover_rate": 1.0, **kwargs}

    with pytest.raises(ValueError, match=expected):
        analyze_turnover_costs(**args)


def test_execution_schedule_fills_a_small_order_in_one_day():
    schedule = optimal_execution_schedule(50_000, 2_000_000, max_participation_rate=0.05)

    assert len(schedule) == 1
    assert schedule[0]["cumulative_filled"] == pytest.approx(50_000)


def test_execution_schedule_spreads_a_large_order_and_fills_it_exactly():
    schedule = optimal_execution_schedule(500_000, 2_000_000, max_participation_rate=0.05)

    assert len(schedule) == 5
    assert schedule[-1]["cumulative_filled"] == pytest.approx(500_000)
    assert [day["day"] for day in schedule] == [1, 2, 3, 4, 5]


def test_execution_schedule_caps_at_the_time_horizon():
    schedule = optimal_execution_schedule(
        10_000_000, 1_000_000, max_participation_rate=0.05, time_horizon_days=3
    )

    assert len(schedule) == 3
    assert schedule[-1]["cumulative_filled"] == pytest.approx(10_000_000)


def test_execution_schedule_rejects_zero_volume():
    with pytest.raises(ValueError, match="avg_daily_volume"):
        optimal_execution_schedule(500_000, 0)


def test_batch_cost_analysis_appends_cost_columns():
    trades = pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "order_size": [50_000.0, 75_000.0],
            "avg_daily_volume": [2_000_000.0, 3_000_000.0],
            "spread_bps": [3.0, 4.0],
            "volatility": [0.015, 0.018],
        }
    )

    result = batch_cost_analysis(trades)

    assert result.height == 2
    assert result["total_cost_bps"].to_list() == pytest.approx(
        [
            result["commission_bps"][i] + result["slippage_bps"][i] + result["impact_bps"][i]
            for i in range(2)
        ]
    )


def test_batch_cost_analysis_defaults_a_missing_volatility():
    trades = pl.DataFrame(
        {
            "order_size": [50_000.0],
            "avg_daily_volume": [2_000_000.0],
            "spread_bps": [3.0],
            "volatility": [None],
        }
    )

    assert batch_cost_analysis(trades)["total_cost_bps"][0] == pytest.approx(
        estimate_total_cost(50_000.0, 2_000_000.0, spread_bps=3.0, volatility=0.02)
    )


def test_batch_cost_analysis_keeps_its_schema_on_an_empty_frame():
    """A horizontal concat with a column-less frame used to drop the costs."""
    trades = pl.DataFrame(
        schema={
            "order_size": pl.Float64,
            "avg_daily_volume": pl.Float64,
            "spread_bps": pl.Float64,
        }
    )

    result = batch_cost_analysis(trades)

    assert result.height == 0
    assert "total_cost_bps" in result.columns
    assert "commission_bps" in result.columns


def test_batch_cost_analysis_names_the_bad_column_on_a_zero_volume_row():
    trades = pl.DataFrame(
        {
            "order_size": [50_000.0],
            "avg_daily_volume": [0.0],
            "spread_bps": [3.0],
        }
    )

    with pytest.raises(ValueError, match="avg_daily_volume"):
        batch_cost_analysis(trades)
