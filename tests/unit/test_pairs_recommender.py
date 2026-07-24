"""Tests for pairs_recommender half-life estimation."""

import polars as pl

from imst_quant.utils.pairs_recommender import _calculate_half_life


class TestCalculateHalfLife:
    def test_mean_reverting_spread_returns_finite_half_life(self):
        # Oscillating spread with negative beta (mean reverting)
        spread = pl.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        half_life = _calculate_half_life(spread)
        assert half_life >= 1.0
        assert half_life != float("inf")

    def test_constant_spread_returns_infinite_half_life(self):
        # Zero variance in lagged spread -> would divide by zero without a guard
        spread = pl.Series([2.0] * 10)
        assert _calculate_half_life(spread) == float("inf")
