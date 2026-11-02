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


def test_find_trading_pairs_detects_cointegrated_pair():
    # Used to return [] for every input: polars Series were passed into the
    # pandas-based cointegration API and the AttributeError was swallowed.
    import datetime as dt

    import numpy as np

    from imst_quant.utils.pairs_recommender import find_trading_pairs

    rng = np.random.default_rng(0)
    n = 200
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = 2 * x + rng.normal(0, 1, n)
    noise = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    df = pl.DataFrame({
        "date": dates * 3,
        "asset_id": ["X"] * n + ["Y"] * n + ["N"] * n,
        "close": np.r_[x, y, noise],
    })
    pairs = find_trading_pairs(df, min_correlation=0.0, lookback_days=0)
    found = {frozenset((p.asset1, p.asset2)): p for p in pairs}
    assert frozenset(("X", "Y")) in found
    xy = found[frozenset(("X", "Y"))]
    assert abs(abs(xy.hedge_ratio) - (2.0 if xy.asset1 == "Y" else 0.5)) < 0.1
    assert xy.signal in {"BUY", "SELL", "HOLD"}
