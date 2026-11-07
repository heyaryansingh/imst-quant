"""Regression: stress_test applies the scenario shock once, not every period."""

import pandas as pd
import pytest

from imst_quant.utils.advanced_risk import stress_test


def test_shock_applied_once():
    returns = pd.Series([0.01, -0.005, 0.002, 0.0] * 25)
    base_total = (1 + returns).prod() - 1
    res = stress_test(returns, {"crash": -0.30})["crash"]
    assert res["total_return"] == pytest.approx(0.7 * (1 + base_total) - 1)
    assert res["max_drawdown"] <= -0.30 + 1e-12
    assert res["max_drawdown"] > -0.35


def test_scenarios_ordered_by_severity():
    returns = pd.Series([0.001] * 50)
    res = stress_test(returns)
    totals = [res[k]["total_return"] for k in ("mild_correction", "correction", "bear_market", "crash", "black_swan")]
    assert totals == sorted(totals, reverse=True)
    assert res["mild_correction"]["total_return"] > -0.05
