import numpy as np
import pandas as pd

from imst_quant.utils.portfolio_stress_testing import HistoricalScenarios, PortfolioStressTester


def test_stressed_var_uses_weighted_portfolio_returns():
    col = [0.01, -0.02, 0.015, -0.01, 0.0] * 10
    r = pd.DataFrame({"A": col, "B": col})
    t = PortfolioStressTester(pd.Series({"A": 1.0, "B": 1.0}), r)
    res = t.apply_scenario(HistoricalScenarios.RATE_SHOCK_2022)  # vol x1.5
    expected = np.percentile((r.mean(axis=1) - r.mean(axis=1).mean()) * 1.5 + r.mean(axis=1).mean(), 5)
    assert np.isclose(res["stressed_var_95"], expected)
