"""Tests for RiskDashboard per-position VaR decomposition and beta."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from imst_quant.utils.risk_dashboard import RiskDashboard


def _dashboard(benchmark=False, seed=0, n=400):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    asset_returns = {
        "A": market + rng.normal(0, 0.005, n),
        "B": 0.5 * market + rng.normal(0, 0.01, n),
        "HEDGE": -market + rng.normal(0, 0.002, n),
    }
    weights = {"A": 0.5, "B": 0.45, "HEDGE": 0.05}
    portfolio = sum(weights[k] * pd.Series(v) for k, v in asset_returns.items())
    positions = pd.DataFrame(
        {
            "symbol": list(weights),
            "weight": list(weights.values()),
            "returns": [pd.Series(asset_returns[k]) for k in weights],
        }
    )
    return RiskDashboard(
        portfolio, positions, benchmark_returns=pd.Series(market) if benchmark else None
    ), portfolio, market, asset_returns


def test_var_contributions_sum_to_parametric_portfolio_var():
    dashboard, portfolio, _, _ = _dashboard()
    risks = dashboard.calculate_position_risk()

    expected = portfolio.mean() + stats.norm.ppf(0.05) * portfolio.std()
    assert sum(r.contribution_to_var for r in risks) == pytest.approx(expected)
    hedge = next(r for r in risks if r.symbol == "HEDGE")
    # The hedge offsets losses, so adding weight to it raises (improves) VaR.
    assert hedge.marginal_var > 0


def test_beta_matches_regression_slope():
    dashboard, _, market, asset_returns = _dashboard(benchmark=True)
    risks = dashboard.calculate_position_risk()

    slope = np.polyfit(market, asset_returns["A"], 1)[0]
    assert next(r for r in risks if r.symbol == "A").beta == pytest.approx(slope)
