"""Tests for the shared downside deviation helper and `utils.risk_metrics`.

The Sortino denominator is the root-mean-square shortfall below the target
over *every* period. The standard deviation of the losing periods measures
something else entirely — how much the losses differ from each other — so a
strategy losing an identical amount every period scored as risk-free.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from imst_quant.utils.risk_metrics import downside_deviation, sortino_ratio

ANNUAL = 252 ** 0.5


class TestDownsideDeviation:
    def test_no_losses_is_zero(self):
        assert downside_deviation([0.01, 0.02, 0.03]) == 0.0

    def test_gains_do_not_offset_losses(self):
        # Only the shortfall below the target counts; the +0.10 is ignored.
        assert downside_deviation([0.10, -0.02, 0.10, -0.02]) == pytest.approx(
            (2 * 0.02 ** 2 / 4) ** 0.5
        )

    def test_constant_losses_are_not_risk_free(self):
        assert downside_deviation([-0.03] * 8) == pytest.approx(0.03)

    def test_single_loss_is_defined(self):
        # A one-element sample std is undefined; the RMS shortfall is not.
        assert downside_deviation([0.01, -0.02]) == pytest.approx(
            (0.02 ** 2 / 2) ** 0.5
        )

    def test_target_shifts_the_threshold(self):
        # Against a 1% hurdle, a 0.5% gain is a 0.5% shortfall.
        assert downside_deviation([0.005], target=0.01) == pytest.approx(0.005)

    def test_empty_input_is_zero(self):
        assert downside_deviation([]) == 0.0

    def test_non_finite_values_are_ignored(self):
        assert downside_deviation([-0.02, float("nan")]) == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "wrap",
        [list, np.array, pl.Series, pd.Series],
        ids=["list", "numpy", "polars", "pandas"],
    )
    def test_accepts_any_sequence_type(self, wrap):
        assert downside_deviation(wrap([0.01, -0.02])) == pytest.approx(
            (0.02 ** 2 / 2) ** 0.5
        )


class TestSortinoRatio:
    def test_steady_loss_is_negative_not_zero(self):
        # Losing 1% every period: mean -0.01, downside deviation 0.01.
        assert sortino_ratio(pl.Series([-0.01] * 20)) == pytest.approx(-1.0 * ANNUAL)

    def test_identical_losses_do_not_look_risk_free(self):
        # Every losing period is -0.01, so their sample std is zero.
        returns = pl.Series([0.02, -0.01, 0.03, -0.01, -0.01, 0.01])
        assert sortino_ratio(returns) > 0.0

    def test_no_losses_is_capped(self):
        assert sortino_ratio(pl.Series([0.01] * 10)) == 100.0

    def test_no_losses_and_no_gains_is_zero(self):
        assert sortino_ratio(pl.Series([0.0] * 10)) == 0.0

    def test_empty_series_is_zero(self):
        assert sortino_ratio(pl.Series([], dtype=pl.Float64)) == 0.0

    def test_risk_free_rate_shifts_the_threshold(self):
        # A flat 1% per period is all downside against a 2% risk-free rate.
        assert sortino_ratio(pl.Series([0.01] * 10), risk_free_rate=0.02) == (
            pytest.approx(-1.0 * ANNUAL)
        )

    def test_accepts_a_dataframe_column(self):
        df = pl.DataFrame({"returns": [-0.01] * 20})
        assert sortino_ratio(df) == pytest.approx(sortino_ratio(df["returns"]))
