import numpy as np
import polars as pl

from imst_quant.utils.factor_analysis import estimate_factor_exposures


def test_collinear_factors_do_not_crash():
    a = np.random.default_rng(1).normal(0, 0.01, 50)
    r = pl.Series(np.random.default_rng(0).normal(0, 0.01, 50))
    out = estimate_factor_exposures(r, pl.DataFrame({"A": a, "B": 2 * a}))
    assert set(out.betas) == {"A", "B"}
