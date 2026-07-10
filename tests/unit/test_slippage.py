"""Tests for slippage estimation utilities."""

import polars as pl
import pytest

from imst_quant.utils.slippage import (
    SlippageEstimate,
    SlippageEstimator,
    quick_slippage_estimate,
)


class TestSlippageEstimatorEstimate:
    def test_returns_slippage_estimate_dataclass(self):
        estimator = SlippageEstimator()
        result = estimator.estimate(
            trade_value=10_000, avg_daily_volume=5_000_000, volatility=0.2
        )
        assert isinstance(result, SlippageEstimate)

    def test_total_slippage_is_sum_of_components(self):
        estimator = SlippageEstimator()
        result = estimator.estimate(
            trade_value=10_000,
            avg_daily_volume=5_000_000,
            volatility=0.2,
            bid_ask_spread_bps=10.0,
        )
        fixed_cost = result.fixed_cost_bps
        components_sum = (
            result.price_impact_bps + result.timing_cost_bps + fixed_cost
        )
        # total = price_impact + timing + fixed (+ liquidity penalty, 0 here
        # since volume is well above the default 1e6 threshold)
        assert result.estimated_slippage_bps == pytest.approx(components_sum)

    def test_market_order_adds_extra_fixed_cost(self):
        estimator = SlippageEstimator()
        market = estimator.estimate(
            trade_value=10_000, avg_daily_volume=5_000_000, volatility=0.2,
            bid_ask_spread_bps=10.0, is_market_order=True,
        )
        limit = estimator.estimate(
            trade_value=10_000, avg_daily_volume=5_000_000, volatility=0.2,
            bid_ask_spread_bps=10.0, is_market_order=False,
        )
        assert market.fixed_cost_bps == pytest.approx(limit.fixed_cost_bps + 1.0)

    def test_low_volume_adds_liquidity_penalty(self):
        estimator = SlippageEstimator(liquidity_threshold=1_000_000)
        low_volume = estimator.estimate(
            trade_value=1_000, avg_daily_volume=100_000, volatility=0.2
        )
        high_volume = estimator.estimate(
            trade_value=1_000, avg_daily_volume=10_000_000, volatility=0.2
        )
        assert low_volume.estimated_slippage_bps > high_volume.estimated_slippage_bps

    def test_zero_avg_daily_volume_does_not_raise(self):
        estimator = SlippageEstimator()
        result = estimator.estimate(
            trade_value=1_000, avg_daily_volume=0, volatility=0.2
        )
        assert result.estimated_slippage_bps >= 0.0

    @pytest.mark.parametrize(
        "avg_daily_volume,volatility,expected",
        [
            (20_000_000, 0.1, "high"),
            (2_000_000, 0.4, "medium"),
            (100_000, 0.8, "low"),
        ],
    )
    def test_confidence_level_thresholds(self, avg_daily_volume, volatility, expected):
        estimator = SlippageEstimator()
        result = estimator.estimate(
            trade_value=10_000, avg_daily_volume=avg_daily_volume, volatility=volatility
        )
        assert result.confidence_level == expected


class TestPrivateHelpers:
    def test_price_impact_saturates_at_full_volume_fraction(self):
        estimator = SlippageEstimator()
        impact_at_cap = estimator._compute_price_impact(1.0)
        impact_over_cap = estimator._compute_price_impact(2.0)
        assert impact_at_cap == pytest.approx(impact_over_cap)

    def test_price_impact_zero_at_zero_volume_fraction(self):
        estimator = SlippageEstimator()
        assert estimator._compute_price_impact(0.0) == 0.0

    def test_timing_cost_scales_with_volatility(self):
        estimator = SlippageEstimator()
        low_vol_cost = estimator._compute_timing_cost(0.1)
        high_vol_cost = estimator._compute_timing_cost(0.5)
        assert high_vol_cost > low_vol_cost

    def test_estimated_spread_is_clamped(self):
        estimator = SlippageEstimator()
        spread = estimator._estimate_spread(volatility=100.0, avg_daily_volume=1.0)
        assert 1.0 <= spread <= 50.0


class TestEstimateBatch:
    def test_adds_slippage_columns(self):
        estimator = SlippageEstimator()
        trades_df = pl.DataFrame(
            {
                "trade_value": [10_000, 20_000],
                "avg_daily_volume": [5_000_000, 2_000_000],
                "volatility": [0.2, 0.3],
            }
        )
        result = estimator.estimate_batch(trades_df)
        assert "estimated_slippage_bps" in result.columns
        assert "confidence" in result.columns
        assert result.height == trades_df.height

    def test_original_columns_preserved(self):
        estimator = SlippageEstimator()
        trades_df = pl.DataFrame(
            {
                "trade_value": [10_000],
                "avg_daily_volume": [5_000_000],
                "volatility": [0.2],
            }
        )
        result = estimator.estimate_batch(trades_df)
        for col in trades_df.columns:
            assert col in result.columns


class TestQuickSlippageEstimate:
    def test_returns_float(self):
        result = quick_slippage_estimate(10_000, 5_000_000)
        assert isinstance(result, float)

    def test_uses_default_volatility(self):
        explicit = quick_slippage_estimate(10_000, 5_000_000, volatility=0.2)
        default = quick_slippage_estimate(10_000, 5_000_000)
        assert explicit == pytest.approx(default)

    def test_higher_volatility_increases_estimate(self):
        low = quick_slippage_estimate(10_000, 5_000_000, volatility=0.1)
        high = quick_slippage_estimate(10_000, 5_000_000, volatility=0.9)
        assert high > low
