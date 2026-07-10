"""Tests for position sizing utilities."""

import polars as pl
import pytest

from imst_quant.utils.position_sizing import (
    atr_based_size,
    calculate_optimal_units,
    calculate_position_sizes,
    fixed_fractional_size,
    get_recommended_sizing,
    kelly_criterion_size,
    volatility_based_size,
)


class TestFixedFractionalSize:
    def test_no_stop_returns_risk_amount(self):
        size = fixed_fractional_size(100000, risk_per_trade=0.02)
        assert size == pytest.approx(2000.0)

    def test_with_entry_and_stop_returns_dollar_value(self):
        # risk_amount=2000, price_risk_per_unit=2 -> units=1000 -> dollar=1000*50=50000
        size = fixed_fractional_size(
            100000, risk_per_trade=0.02, entry_price=50.0, stop_loss=48.0
        )
        assert size == pytest.approx(50000.0)

    def test_zero_price_risk_falls_back_to_risk_amount(self):
        size = fixed_fractional_size(
            100000, risk_per_trade=0.02, entry_price=50.0, stop_loss=50.0
        )
        assert size == pytest.approx(2000.0)


class TestKellyCriterionSize:
    def test_known_formula_result(self):
        # p=0.6, b=avg_win/avg_loss=2, q=0.4 -> kelly_pct=(0.6*2-0.4)/2=0.4
        # quarter kelly (default fraction 0.25) -> 0.1 * capital
        size = kelly_criterion_size(
            100000, win_rate=0.6, avg_win=2.0, avg_loss=1.0, kelly_fraction=0.25
        )
        assert size == pytest.approx(10000.0)

    @pytest.mark.parametrize("win_rate", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_win_rate_defaults_to_half(self, win_rate):
        # should not raise, falls back to win_rate=0.5 internally
        size = kelly_criterion_size(100000, win_rate=win_rate, avg_win=0.02, avg_loss=0.01)
        assert size >= 0

    def test_negative_avg_loss_uses_abs(self):
        size_pos = kelly_criterion_size(100000, win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        size_neg = kelly_criterion_size(100000, win_rate=0.55, avg_win=0.02, avg_loss=-0.01)
        assert size_pos == pytest.approx(size_neg)

    def test_negative_edge_clips_to_zero(self):
        size = kelly_criterion_size(100000, win_rate=0.1, avg_win=0.01, avg_loss=0.05)
        assert size == pytest.approx(0.0)


class TestVolatilityBasedSize:
    def test_scales_inversely_with_asset_volatility(self):
        size = volatility_based_size(100000, target_volatility=0.15, asset_volatility=0.30)
        assert size == pytest.approx(50000.0)

    def test_caps_at_full_capital(self):
        size = volatility_based_size(100000, target_volatility=0.30, asset_volatility=0.10)
        assert size == pytest.approx(100000.0)

    def test_zero_asset_volatility_uses_fallback(self):
        size = volatility_based_size(100000, target_volatility=0.15, asset_volatility=0.0)
        assert size > 0

    def test_entry_price_returns_units(self):
        units = volatility_based_size(
            100000, target_volatility=0.15, asset_volatility=0.30, entry_price=50.0
        )
        assert units == pytest.approx(1000.0)


class TestAtrBasedSize:
    def test_known_result(self):
        # risk_amount=2000, stop_distance=2.5*2=5 -> units=400, dollar=400*50=20000 < capital
        units = atr_based_size(100000, atr=2.5, entry_price=50.0, atr_multiplier=2.0, risk_per_trade=0.02)
        assert units == pytest.approx(400.0)

    def test_caps_units_at_capital(self):
        units = atr_based_size(1000, atr=0.01, entry_price=50.0, atr_multiplier=1.0, risk_per_trade=0.5)
        assert units * 50.0 <= 1000.0 + 1e-6

    def test_zero_stop_distance_returns_zero(self):
        units = atr_based_size(100000, atr=0.0, entry_price=50.0, atr_multiplier=2.0)
        assert units == 0


class TestCalculatePositionSizes:
    def test_fixed_fractional_method(self):
        df = pl.DataFrame({
            "close": [100.0, 102.0, 98.0],
            "signal_direction": [1, 0, -1],
        })
        result = calculate_position_sizes(df, capital=100000, method="fixed_fractional")
        assert "position_size" in result.columns
        assert "position_units" in result.columns
        # zero-signal row must have zero position
        assert result["position_size"][1] == 0

    def test_atr_method_requires_atr_col(self):
        df = pl.DataFrame({
            "close": [100.0],
            "signal_direction": [1],
        })
        with pytest.raises(ValueError):
            calculate_position_sizes(df, capital=100000, method="atr")

    def test_volatility_method_requires_volatility_col(self):
        df = pl.DataFrame({
            "close": [100.0],
            "signal_direction": [1],
        })
        with pytest.raises(ValueError):
            calculate_position_sizes(df, capital=100000, method="volatility")

    def test_unknown_method_raises(self):
        df = pl.DataFrame({
            "close": [100.0],
            "signal_direction": [1],
        })
        with pytest.raises(ValueError):
            calculate_position_sizes(df, capital=100000, method="bogus")

    def test_atr_method_computes_units(self):
        df = pl.DataFrame({
            "close": [50.0, 50.0],
            "signal_direction": [1, 1],
            "atr": [2.5, 2.5],
        })
        result = calculate_position_sizes(
            df, capital=100000, method="atr", atr_col="atr", risk_per_trade=0.02
        )
        assert result["position_units"][0] == pytest.approx(400.0)


class TestGetRecommendedSizing:
    @pytest.mark.parametrize("tolerance", ["conservative", "medium", "aggressive"])
    def test_known_profiles_have_expected_keys(self, tolerance):
        params = get_recommended_sizing(tolerance)
        assert set(["method", "risk_per_trade", "kelly_fraction", "atr_multiplier", "target_volatility", "max_positions"]).issubset(params.keys())

    def test_unknown_tolerance_falls_back_to_medium(self):
        params = get_recommended_sizing("bogus")
        assert params["method"] == "atr"

    def test_account_size_adds_max_position_size(self):
        params = get_recommended_sizing("medium", account_size=100000)
        assert "max_position_size" in params
        assert params["account_size"] == 100000


class TestCalculateOptimalUnits:
    def test_known_result(self):
        result = calculate_optimal_units(entry_price=50.0, stop_loss=48.0, capital=100000, risk_pct=0.02)
        # risk_amount=2000, price_risk=2 -> units=1000, dollar_size=50000, within 25% cap? 25000 cap -> capped
        assert result["dollar_size"] <= 100000 * 0.25 + 1e-6

    def test_zero_price_risk_raises(self):
        with pytest.raises(ValueError):
            calculate_optimal_units(entry_price=50.0, stop_loss=50.0, capital=100000)

    def test_uncapped_result_matches_risk_amount(self):
        result = calculate_optimal_units(
            entry_price=50.0, stop_loss=49.0, capital=1000000, risk_pct=0.02, max_capital_pct=1.0
        )
        assert result["risk_amount"] == pytest.approx(1000000 * 0.02, rel=1e-3)
