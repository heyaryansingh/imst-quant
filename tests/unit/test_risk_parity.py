"""Tests for risk parity portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.risk_parity import (
    RiskParityOptimizer,
    TargetVolatilityRiskParity,
    calculate_risk_parity_with_constraints,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    """Four assets with deliberately different volatilities."""
    rng = np.random.default_rng(7)
    n = 500
    return pd.DataFrame(
        {
            "LOWVOL": rng.normal(0.0002, 0.004, n),
            "LOWVOL2": rng.normal(0.0002, 0.005, n),
            "MIDVOL": rng.normal(0.0003, 0.012, n),
            "HIGHVOL": rng.normal(0.0004, 0.040, n),
        }
    )


def test_erc_weights_sum_to_one_and_equalize_risk(returns):
    optimizer = RiskParityOptimizer(returns, method="equal_risk_contribution")
    weights = optimizer.optimize()

    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0).all()

    contributions = optimizer.calculate_risk_contributions(weights)
    # Every asset should carry roughly the same share of portfolio risk.
    shares = contributions / contributions.sum()
    assert shares.max() - shares.min() < 0.02


def test_hrp_weights_are_valid(returns):
    weights = RiskParityOptimizer(returns, method="hierarchical").optimize()

    assert list(weights.index) == list(returns.columns)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()


def test_hrp_underweights_the_riskiest_asset(returns):
    """Regression: bisection used to scale weights *up* with cluster variance."""
    weights = RiskParityOptimizer(returns, method="hierarchical").optimize()

    assert weights["HIGHVOL"] < weights["MIDVOL"]
    assert weights["HIGHVOL"] < weights["LOWVOL"]


def test_hrp_single_asset():
    single = pd.DataFrame({"ONLY": np.linspace(-0.01, 0.01, 50)})
    weights = RiskParityOptimizer(single, method="hierarchical").optimize()

    assert weights.to_dict() == {"ONLY": 1.0}


def test_hrp_handles_odd_asset_count(returns):
    """Recursive bisection pairs sub-clusters; an odd count must not break it."""
    weights = RiskParityOptimizer(returns[["LOWVOL", "MIDVOL", "HIGHVOL"]], method="hierarchical").optimize()

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert len(weights) == 3


def test_adaptive_respects_min_weight(returns):
    weights = RiskParityOptimizer(returns, method="adaptive").optimize(
        lookback_period=60, min_weight=0.15
    )

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0.1).all()


def test_unknown_method_raises(returns):
    with pytest.raises(ValueError, match="Unknown method"):
        RiskParityOptimizer(returns, method="nope").optimize()


def test_diversification_ratio_above_one(returns):
    optimizer = RiskParityOptimizer(returns)
    weights = optimizer.optimize()

    assert optimizer.calculate_diversification_ratio(weights) > 1.0


def test_target_volatility_scaling(returns):
    scaled, leverage = TargetVolatilityRiskParity(returns, target_volatility=0.10).optimize()

    realized = np.sqrt(scaled @ returns.cov() @ scaled) * np.sqrt(252)
    assert realized == pytest.approx(0.10, rel=1e-6)
    assert leverage > 0


def test_constrained_weights_respect_bounds(returns):
    weights = calculate_risk_parity_with_constraints(returns, min_weight=0.1, max_weight=0.4)

    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0.1 - 1e-6).all()
    assert (weights <= 0.4 + 1e-6).all()
