"""EVT VaR and tail dependency must match their textbook definitions."""

import numpy as np
import polars as pl

from imst_quant.utils.tail_risk import extreme_value_at_risk, tail_dependency


def test_evt_var_matches_daily_quantile_for_normal_returns():
    r = pl.Series(np.random.default_rng(0).normal(0, 0.02, 5000))
    evt = extreme_value_at_risk(r, confidence_level=0.99)
    empirical = -r.quantile(0.01)
    # Old code took the 1% quantile of *monthly worst days* (~0.18 here)
    assert abs(evt - empirical) / empirical < 0.25


def test_evt_var_ignores_nan():
    vals = np.random.default_rng(1).normal(0, 0.02, 3000)
    vals[::50] = np.nan
    assert np.isfinite(extreme_value_at_risk(pl.Series(vals)))


def test_tail_dependency_is_conditional_probability():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    lower, upper = tail_dependency(pl.Series(a), pl.Series(b), quantile=0.95)
    # Independent assets: P(B in tail | A in tail) ~= 0.05
    assert 0.03 < lower < 0.07 and 0.03 < upper < 0.07
    lower, upper = tail_dependency(pl.Series(a), pl.Series(a), quantile=0.95)
    assert lower == 1.0 and upper == 1.0
