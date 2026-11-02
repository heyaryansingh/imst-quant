import numpy as np
import pandas as pd

from imst_quant.utils.portfolio_attribution import FactorAttribution


def test_factor_attribution_accepts_risk_free_series():
    idx = pd.date_range("2020-01-01", periods=30)
    f = pd.DataFrame({"f": np.random.default_rng(0).normal(size=30)}, index=idx)
    fa = FactorAttribution(f["f"] * 0.5, f, pd.Series(0.0, index=idx))
    betas, _, _ = fa.run_regression()
    assert np.isclose(betas["f"], 0.5)
