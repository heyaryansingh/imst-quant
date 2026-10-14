"""Tests for SLSQP-based mean_variance_optimize."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from imst_quant.utils.portfolio_optimization import (
    OptimizationObjective,
    estimate_covariance,
    estimate_expected_returns,
    mean_variance_optimize,
    portfolio_sharpe,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.normal([0.0008, 0.0003, 0.0001], [0.02, 0.01, 0.005], (500, 3)),
        columns=list("ABC"),
    )


def test_min_variance_matches_reference_solver(returns):
    cov = estimate_covariance(returns).values
    ref = minimize(
        lambda w: w @ cov @ w,
        np.ones(3) / 3,
        method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    result = mean_variance_optimize(returns, OptimizationObjective.MIN_VARIANCE)
    assert result.volatility == pytest.approx(np.sqrt(ref.fun), rel=1e-3)
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_target_risk_respects_volatility_cap(returns):
    result = mean_variance_optimize(
        returns, OptimizationObjective.TARGET_RISK, target_volatility=0.15
    )
    assert result.volatility <= 0.15 + 1e-6


def test_target_return_hits_target(returns):
    result = mean_variance_optimize(
        returns, OptimizationObjective.TARGET_RETURN, target_return=0.05
    )
    assert result.expected_return == pytest.approx(0.05, abs=1e-4)


def test_max_sharpe_beats_equal_weight(returns):
    result = mean_variance_optimize(returns, OptimizationObjective.MAX_SHARPE)
    w = np.ones(3) / 3
    equal = portfolio_sharpe(
        w, estimate_expected_returns(returns).values, estimate_covariance(returns).values
    )
    assert result.sharpe_ratio >= equal - 1e-9
    assert all(w >= -1e-9 for w in result.weights.values())
