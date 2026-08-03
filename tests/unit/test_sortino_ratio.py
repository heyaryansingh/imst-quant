"""Tests for the Sortino ratio's downside deviation denominator."""

import polars as pl
import pytest

from imst_quant.trading.risk_metrics import calculate_sortino_ratio


class TestSortinoEdgeCases:
    """Degenerate inputs must return a number rather than raising."""

    def test_empty_series_returns_zero(self):
        assert calculate_sortino_ratio(pl.Series([], dtype=pl.Float64)) == 0.0

    def test_single_losing_period_does_not_raise(self):
        # A one-element sample std is undefined in polars and returns None,
        # so the previous float(std) crashed with a TypeError here.
        assert calculate_sortino_ratio(pl.Series([-0.05])) == pytest.approx(-1.0)

    def test_no_losses_is_positive_infinity(self):
        assert calculate_sortino_ratio(pl.Series([0.01] * 10)) == float("inf")

    def test_all_zero_returns_is_zero(self):
        assert calculate_sortino_ratio(pl.Series([0.0] * 10)) == 0.0


class TestDownsideDeviation:
    """The denominator is RMS shortfall, not the std of the losing periods."""

    def test_constant_losses_are_not_risk_free(self):
        # Every period loses exactly 5%. The std of those losses is 0, which
        # previously short-circuited to 0.0 -- reporting a strategy that
        # bleeds 5% a day as having no downside risk.
        result = calculate_sortino_ratio(pl.Series([-0.05] * 10))
        assert result == pytest.approx(-1.0)

    def test_matches_rms_shortfall_definition(self):
        returns = [0.10, -0.05, 0.03, -0.02, 0.04]
        # mean = 0.02; shortfalls = [0, -0.05, 0, -0.02, 0]
        # downside deviation = sqrt((0.05^2 + 0.02^2) / 5) = 0.0240832
        expected = 0.02 / ((0.05**2 + 0.02**2) / 5) ** 0.5
        result = calculate_sortino_ratio(pl.Series(returns))
        assert result == pytest.approx(expected)

    def test_shortfalls_average_over_all_periods_not_just_losses(self):
        # Same single loss, but padded with flat periods. Averaging the
        # squared shortfall over all periods must shrink the denominator,
        # so adding non-losing periods cannot make the ratio look worse.
        few = calculate_sortino_ratio(pl.Series([0.01, -0.02]))
        many = calculate_sortino_ratio(pl.Series([0.01, -0.02] + [0.01] * 8))
        assert many > few

    def test_deeper_loss_lowers_the_ratio(self):
        shallow = calculate_sortino_ratio(pl.Series([0.05, -0.01, 0.05, -0.01]))
        deep = calculate_sortino_ratio(pl.Series([0.05, -0.10, 0.05, -0.10]))
        assert deep < shallow


class TestSortinoParameters:
    """risk_free_rate and target_return must both shift the calculation."""

    def test_risk_free_rate_lowers_excess_return(self):
        returns = pl.Series([0.05, -0.01, 0.04, -0.02])
        assert calculate_sortino_ratio(returns, risk_free_rate=0.01) < (
            calculate_sortino_ratio(returns)
        )

    def test_target_return_applies_to_numerator_and_denominator(self):
        returns = pl.Series([0.10, -0.05, 0.03, -0.02, 0.04])
        # With target 0.01: mean excess = 0.02 - 0.01 = 0.01 and the
        # shortfalls deepen to [0, -0.06, 0, -0.03, 0].
        expected = 0.01 / ((0.06**2 + 0.03**2) / 5) ** 0.5
        assert calculate_sortino_ratio(returns, target_return=0.01) == pytest.approx(
            expected
        )

    def test_raising_target_never_raises_the_ratio(self):
        returns = pl.Series([0.10, -0.05, 0.03, -0.02, 0.04])
        assert calculate_sortino_ratio(returns, target_return=0.02) < (
            calculate_sortino_ratio(returns, target_return=0.0)
        )
