"""Tests for the component VaR decomposition in calculate_portfolio_var."""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.var_calculator import VaRCalculator, calculate_portfolio_var


def _book(seed=0, n=500):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    returns = pd.DataFrame(
        {
            "A": market + rng.normal(0, 0.005, n),
            "B": 0.5 * market + rng.normal(0, 0.01, n),
            "HEDGE": -market + rng.normal(0, 0.002, n),
        }
    )
    positions = pd.DataFrame({"A": [0.6] * n, "B": [0.3] * n, "HEDGE": [0.1] * n})
    return positions, returns


@pytest.mark.parametrize("method", ["parametric", "historical"])
def test_component_var_sums_to_total(method):
    positions, returns = _book()
    result = calculate_portfolio_var(positions, returns, method=method)

    assert sum(result["component_var"].values()) == pytest.approx(result["total_var"])


def test_parametric_components_match_euler_and_hedge_is_negative():
    positions, returns = _book()
    result = calculate_portfolio_var(positions, returns, method="parametric")

    portfolio = (positions * returns).sum(axis=1)
    assert result["total_var"] == pytest.approx(
        VaRCalculator(portfolio, method="parametric").parametric_var()
    )
    # A short-beta position diversifies the book, so it carries negative risk.
    assert result["component_var"]["HEDGE"] < 0
    # Standalone VaRs ignore diversification and overstate the total.
    assert sum(result["standalone_var"].values()) > result["total_var"]
    assert result["marginal_var"] == result["standalone_var"]
