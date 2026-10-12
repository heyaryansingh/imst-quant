"""Tests for cointegration utilities."""

import numpy as np
import pandas as pd

from imst_quant.utils.cointegration import calculate_hedge_ratio


def _pair():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(size=200)) + 50)
    y = 2.0 * x + 5 + rng.normal(scale=0.1, size=200)
    return y, x


def test_hedge_ratio_ignores_nan():
    y, x = _pair()
    y.iloc[10] = np.nan
    beta, _ = calculate_hedge_ratio(y, x)
    assert abs(beta - 2.0) < 0.05


def test_hedge_ratio_aligns_on_index():
    y, x = _pair()
    beta, _ = calculate_hedge_ratio(y.iloc[5:], x)
    assert abs(beta - 2.0) < 0.05
