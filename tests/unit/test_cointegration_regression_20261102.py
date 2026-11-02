import numpy as np
import pandas as pd

from imst_quant.utils.cointegration import calculate_half_life, rolling_hedge_ratio


def test_rolling_hedge_ratio_exact_beta():
    x = pd.Series(np.random.default_rng(0).normal(size=100))
    assert abs(rolling_hedge_ratio(2 * x, x, window=20).iloc[-1] - 2.0) < 1e-9


def test_half_life_nonzero_mean_spread():
    rng = np.random.default_rng(0)
    s = np.zeros(2000)
    s[0] = 100
    for i in range(1, 2000):
        s[i] = s[i - 1] - 0.1 * (s[i - 1] - 100) + rng.normal()
    assert 4 < calculate_half_life(pd.Series(s)) < 10
