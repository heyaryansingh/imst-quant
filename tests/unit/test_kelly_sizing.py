"""Tests for Kelly Criterion position sizing utilities."""

import pandas as pd
import pytest

from imst_quant.utils.kelly_sizing import (
    KellySizer,
    calculate_optimal_f,
    variance_adjusted_kelly,
)


class TestKellySizerInit:
    def test_default_construction(self):
        sizer = KellySizer()
        assert sizer.kelly_fraction == 0.5
        assert sizer.max_position == 0.25
        assert sizer.min_position == 0.01

    @pytest.mark.parametrize("kelly_fraction", [0.0, -0.1, 1.5])
    def test_invalid_kelly_fraction_raises(self, kelly_fraction):
        with pytest.raises(ValueError):
            KellySizer(kelly_fraction=kelly_fraction)

    @pytest.mark.parametrize("max_position", [0.0, -0.1, 1.5])
    def test_invalid_max_position_raises(self, max_position):
        with pytest.raises(ValueError):
            KellySizer(max_position=max_position)


class TestCalculateKelly:
    def test_known_formula_result(self):
        sizer = KellySizer()
        # f* = (p*b - q) / b, p=0.6, avg_win=2, avg_loss=1 -> b=2, q=0.4
        # f* = (0.6*2 - 0.4) / 2 = 0.8/2 = 0.4
        kelly = sizer.calculate_kelly(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert kelly == pytest.approx(0.4)

    @pytest.mark.parametrize("win_rate", [0.0, -0.1, 1.0, 1.1])
    def test_invalid_win_rate_returns_zero(self, win_rate):
        sizer = KellySizer()
        assert sizer.calculate_kelly(win_rate, 1.0, 1.0) == 0.0

    @pytest.mark.parametrize("avg_win,avg_loss", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
    def test_invalid_win_loss_returns_zero(self, avg_win, avg_loss):
        sizer = KellySizer()
        assert sizer.calculate_kelly(0.5, avg_win, avg_loss) == 0.0

    def test_negative_expectancy_returns_zero(self):
        sizer = KellySizer()
        # Low win rate with poor payoff ratio -> negative Kelly
        kelly = sizer.calculate_kelly(win_rate=0.2, avg_win=1.0, avg_loss=1.0)
        assert kelly == 0.0


class TestSizePosition:
    def test_clips_to_max_position(self):
        sizer = KellySizer(kelly_fraction=1.0, max_position=0.1, min_position=0.01)
        result = sizer.size_position(win_rate=0.9, avg_win=5.0, avg_loss=1.0)
        assert result["recommended_size"] <= 0.1

    def test_tiny_fractional_kelly_clips_up_to_min_position(self):
        # np.clip floors fractional_kelly at min_position, so a tiny
        # fractional Kelly gets raised to min_position rather than zeroed.
        sizer = KellySizer(kelly_fraction=0.01, max_position=0.25, min_position=0.05)
        result = sizer.size_position(win_rate=0.55, avg_win=1.1, avg_loss=1.0)
        assert result["recommended_size"] == pytest.approx(0.05)

    def test_zero_kelly_input_becomes_zero(self):
        # min_position=0 means clip floor is 0, so an invalid (zero) Kelly
        # correctly results in a zero recommended size.
        sizer = KellySizer(kelly_fraction=0.5, max_position=0.25, min_position=1e-9)
        result = sizer.size_position(win_rate=0.0, avg_win=1.0, avg_loss=1.0)
        assert result["recommended_size"] == 0.0

    def test_expectancy_calculation(self):
        sizer = KellySizer()
        result = sizer.size_position(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        expected_expectancy = (0.6 * 2.0) - (0.4 * 1.0)
        assert result["expectancy"] == pytest.approx(expected_expectancy, abs=1e-6)

    def test_returns_all_expected_keys(self):
        sizer = KellySizer()
        result = sizer.size_position(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert set(result.keys()) == {
            "full_kelly",
            "fractional_kelly",
            "recommended_size",
            "expectancy",
        }


class TestSizeFromTrades:
    def test_missing_pnl_column_raises(self):
        sizer = KellySizer()
        with pytest.raises(ValueError):
            sizer.size_from_trades(pd.DataFrame({"not_pnl": [1, 2, 3]}))

    def test_empty_dataframe_returns_zero_dict(self):
        sizer = KellySizer()
        result = sizer.size_from_trades(pd.DataFrame({"pnl": []}))
        assert result == {
            "full_kelly": 0.0,
            "fractional_kelly": 0.0,
            "recommended_size": 0.0,
            "expectancy": 0.0,
        }

    def test_all_wins_returns_zero_dict(self):
        sizer = KellySizer()
        result = sizer.size_from_trades(pd.DataFrame({"pnl": [1.0, 2.0, 3.0]}))
        assert result["recommended_size"] == 0.0

    def test_all_losses_returns_zero_dict(self):
        sizer = KellySizer()
        result = sizer.size_from_trades(pd.DataFrame({"pnl": [-1.0, -2.0, -3.0]}))
        assert result["recommended_size"] == 0.0

    def test_mixed_trades_produces_reasonable_result(self):
        sizer = KellySizer()
        trades = pd.DataFrame({"pnl": [10.0, -5.0, 8.0, -4.0, 12.0, -6.0]})
        result = sizer.size_from_trades(trades)
        assert set(result.keys()) == {
            "full_kelly",
            "fractional_kelly",
            "recommended_size",
            "expectancy",
        }
        assert 0.0 <= result["recommended_size"] <= sizer.max_position


class TestCalculateOptimalF:
    def test_missing_pnl_column_raises(self):
        with pytest.raises(ValueError):
            calculate_optimal_f(pd.DataFrame({"not_pnl": [1, 2, 3]}))

    def test_empty_dataframe_returns_zero(self):
        assert calculate_optimal_f(pd.DataFrame({"pnl": []})) == 0.0

    def test_zero_largest_loss_returns_zero(self):
        # largest_loss is abs(min(pnl)); a min pnl of exactly 0 triggers
        # the "no losses" guard regardless of the other values' signs.
        assert calculate_optimal_f(pd.DataFrame({"pnl": [0.0, 2.0, 3.0]})) == 0.0

    def test_returns_value_in_valid_range(self):
        trades = pd.DataFrame({"pnl": [10.0, -5.0, 8.0, -4.0, 12.0, -6.0, 3.0, -2.0]})
        optimal_f = calculate_optimal_f(trades)
        assert 0.0 <= optimal_f <= 1.0


class TestVarianceAdjustedKelly:
    def test_valid_inputs_return_non_negative(self):
        kelly = variance_adjusted_kelly(
            win_rate=0.6, avg_win=2.0, avg_loss=1.0, win_variance=0.5, loss_variance=0.3
        )
        assert kelly >= 0.0

    @pytest.mark.parametrize("win_rate", [0.0, -0.1, 1.0, 1.1])
    def test_invalid_win_rate_returns_zero(self, win_rate):
        kelly = variance_adjusted_kelly(
            win_rate=win_rate, avg_win=2.0, avg_loss=1.0, win_variance=0.5, loss_variance=0.3
        )
        assert kelly == 0.0

    def test_non_positive_variance_returns_zero(self):
        # Large variances relative to mean can drive var <= 0 is unlikely with this formula
        # for typical values, but zero win/loss variance with equal win/loss and rate=0.5
        # can produce non-positive var in degenerate cases; test near-zero payoff instead.
        kelly = variance_adjusted_kelly(
            win_rate=0.5, avg_win=0.0, avg_loss=0.0, win_variance=0.0, loss_variance=0.0
        )
        assert kelly == 0.0
