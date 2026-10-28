"""Tests for the market impact modeling module.

Tests cover:
- SquareRootModel impact scaling with participation rate
- NonlinearImpactModel power-law scaling
- AdaptiveImpactModel model selection by order size
- VolumeProfileAnalyzer VWAP schedule proportionality
- estimate_slippage sign convention for buy/sell orders
"""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.market_impact import (
    AdaptiveImpactModel,
    AlmgrenChrissModel,
    ImpactParameters,
    NonlinearImpactModel,
    SquareRootModel,
    VolumeProfileAnalyzer,
    estimate_slippage,
)


def test_square_root_model_larger_trade_has_larger_impact():
    params = ImpactParameters(volatility=0.02, daily_volume=1_000_000)
    model = SquareRootModel(params)

    small = model.estimate_impact(trade_size=10_000, current_price=100.0)
    large = model.estimate_impact(trade_size=90_000, current_price=100.0)

    assert large["impact_bps"] > small["impact_bps"]
    assert small["participation_rate"] == pytest.approx(0.01)


def test_square_root_model_zero_trade_has_zero_impact():
    params = ImpactParameters(daily_volume=1_000_000)
    model = SquareRootModel(params)

    result = model.estimate_impact(trade_size=0, current_price=100.0)
    assert result["impact_fraction"] == 0.0
    # zero notional must not divide-by-zero into NaN
    assert result["total_cost_bps"] == 0.0


def test_nonlinear_model_matches_power_law_formula():
    params = ImpactParameters(daily_volume=1_000_000, power_law_exponent=0.6)
    model = NonlinearImpactModel(params)

    result = model.estimate_impact(trade_size=100_000, current_price=50.0, alpha=0.1)
    expected_fraction = 0.1 * (0.1 ** 0.6)

    assert result["impact_fraction"] == pytest.approx(expected_fraction)
    assert result["power_exponent"] == 0.6


def test_adaptive_model_picks_square_root_for_small_orders():
    params = ImpactParameters(daily_volume=1_000_000)
    model = AdaptiveImpactModel(params)

    result = model.estimate_impact(trade_size=10_000, current_price=100.0, adv=1_000_000)
    assert result["recommended_model"] == "square_root"
    assert "square_root" in result


def test_adaptive_model_picks_nonlinear_for_medium_orders():
    params = ImpactParameters(daily_volume=1_000_000)
    model = AdaptiveImpactModel(params)

    result = model.estimate_impact(trade_size=100_000, current_price=100.0, adv=1_000_000)
    assert result["recommended_model"] == "nonlinear"


def test_adaptive_model_picks_almgren_chriss_for_large_orders():
    params = ImpactParameters(daily_volume=1_000_000)
    model = AdaptiveImpactModel(params)

    result = model.estimate_impact(trade_size=300_000, current_price=100.0, adv=1_000_000)
    assert result["recommended_model"] == "almgren_chriss"
    assert result["time_horizon_days"] > 0


def test_vwap_optimal_schedule_sums_to_total_shares():
    volume_curve = np.array([0.1, 0.2, 0.3, 0.4])
    schedule = VolumeProfileAnalyzer.vwap_optimal_schedule(10_000, volume_curve)

    assert schedule.sum() == pytest.approx(10_000)
    assert schedule[-1] > schedule[0]  # proportional to volume curve


def test_estimate_slippage_buy_positive_when_execution_above_arrival():
    orders = pd.DataFrame({"side": ["buy"], "size": [100]})
    execution = pd.Series([101.0])
    arrival = pd.Series([100.0])

    result = estimate_slippage(orders, execution, arrival)

    assert result["price_diff"][0] == pytest.approx(1.0)
    assert result["slippage_dollars"][0] == pytest.approx(100.0)


def test_estimate_slippage_sell_positive_when_execution_below_arrival():
    orders = pd.DataFrame({"side": ["sell"], "size": [100]})
    execution = pd.Series([99.0])
    arrival = pd.Series([100.0])

    result = estimate_slippage(orders, execution, arrival)

    # favorable sell fill: arrival - execution = +1.0
    assert result["price_diff"][0] == pytest.approx(1.0)


def test_sell_orders_cost_the_same_as_buys():
    params = ImpactParameters(volatility=0.02, daily_volume=1_000_000)
    for model in (SquareRootModel(params), NonlinearImpactModel(params)):
        buy = model.estimate_impact(trade_size=50_000, current_price=100.0)
        sell = model.estimate_impact(trade_size=-50_000, current_price=100.0)
        assert np.isfinite(sell["impact_bps"])
        assert sell["impact_bps"] == pytest.approx(buy["impact_bps"])
        assert sell["total_cost"] == pytest.approx(buy["total_cost"])
        assert sell["total_cost"] > 0


def test_zero_or_negative_adv_is_rejected_not_defaulted():
    model = SquareRootModel(ImpactParameters(daily_volume=1_000_000))
    for adv in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            model.estimate_impact(trade_size=1_000, current_price=10.0, adv=adv)


@pytest.mark.parametrize("risk_aversion", [0.0, 1.0, 1e4])
def test_almgren_chriss_trajectory_fully_liquidates(risk_aversion):
    model = AlmgrenChrissModel(ImpactParameters())
    sizes = model.optimal_trajectory(300_000, 2.0, 1_000_000, risk_aversion)
    assert sizes.sum() == pytest.approx(300_000)
    assert (sizes > 0).all()
    # more risk aversion front-loads execution
    if risk_aversion > 1:
        assert sizes[0] > sizes[-1]


def test_almgren_chriss_sub_hour_horizon_does_not_divide_by_zero():
    model = AlmgrenChrissModel(ImpactParameters())
    cost = model.expected_cost(10_000, 0.01, 1_000_000, 50.0)
    assert np.isfinite(cost["cost_bps"]) and cost["optimal_periods"] == 1


def test_adaptive_model_uses_current_market_volatility():
    model = AdaptiveImpactModel(ImpactParameters(volatility=0.02))
    calm = model.estimate_impact(10_000, 100.0, 1_000_000)
    stressed = model.estimate_impact(10_000, 100.0, 1_000_000, market_volatility=0.04)

    assert stressed["square_root"]["impact_bps"] == pytest.approx(
        2 * calm["square_root"]["impact_bps"]
    )
    assert model.params.volatility == 0.02  # override does not leak
    with pytest.raises(ValueError):
        model.estimate_impact(10_000, 100.0, 1_000_000, market_volatility=float("nan"))
