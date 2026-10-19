"""Tests for order flow analysis on dirty trade data."""

import numpy as np
import pandas as pd

from imst_quant.utils.order_flow import analyze_order_flow


def _trades(n=300):
    rng = np.random.default_rng(0)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, n)))
    volumes = pd.Series(rng.integers(1, 500, n).astype(float))
    return prices, volumes


def test_nan_price_or_volume_rows_are_skipped():
    prices, volumes = _trades()
    clean = analyze_order_flow(prices, volumes, vpin_bucket_size=2000)

    dirty_p, dirty_v = prices.copy(), volumes.copy()
    dirty_p.loc[len(dirty_p)] = np.nan
    dirty_v.loc[len(dirty_v)] = 100.0
    dirty_p.loc[len(dirty_p)] = 101.0
    dirty_v.loc[len(dirty_v)] = np.inf
    dirty = analyze_order_flow(dirty_p, dirty_v, vpin_bucket_size=2000)

    assert dirty == clean
    assert np.isfinite(dirty.vpin)
