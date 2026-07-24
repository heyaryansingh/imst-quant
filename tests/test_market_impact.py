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
