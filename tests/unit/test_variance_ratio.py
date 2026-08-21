"""Tests for the Lo-MacKinlay variance ratio test."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from imst_quant.utils.mean_reversion import _normal_cdf, variance_ratio_test


def _ou_prices(n: int = 1000, kappa: float = 0.5, seed: int = 0) -> np.ndarray:
    """Ornstein-Uhlenbeck prices: strongly mean-reverting by construction."""
    rng = np.random.default_rng(seed)
    prices = np.empty(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + kappa * (100.0 - prices[i - 1]) + rng.normal(0, 1)
    return prices


def _random_walk(n: int = 1000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1, n))


def test_detects_mean_reversion():
    """Regression: the standard error was sqrt(n) too large, so a series with
    VR well below 1 was still reported as a random walk."""
    result = variance_ratio_test(_ou_prices(), lag=4)

    assert result["vr"] < 0.8
    assert result["z_stat"] < -3
    assert result["p_value"] < 0.01
    assert result["is_random_walk"] is False


def test_does_not_reject_random_walk():
    result = variance_ratio_test(_random_walk(), lag=4)

    assert result["vr"] == pytest.approx(1.0, abs=0.15)
    assert result["p_value"] > 0.05
    assert result["is_random_walk"] is True


def test_short_series_returns_neutral_result():
    result = variance_ratio_test(np.linspace(100, 110, 6), lag=4)

    assert result == {"vr": 1.0, "z_stat": 0.0, "p_value": 1.0, "is_random_walk": True}


def test_constant_series_returns_neutral_result():
    result = variance_ratio_test(np.full(200, 100.0), lag=2)

    assert result["is_random_walk"] is True
    assert result["z_stat"] == 0.0


def test_p_value_stays_precise_in_the_tail():
    """The old Abramowitz-Stegun approximation floored at 0.0 past ~z=6."""
    result = variance_ratio_test(_ou_prices(kappa=0.9), lag=4)

    assert 0 < result["p_value"] < 1e-10


@pytest.mark.parametrize("z", [-8.0, -4.0, -1.96, 0.0, 1.0, 3.5])
def test_normal_cdf_matches_scipy(z):
    assert _normal_cdf(z) == pytest.approx(norm.cdf(z), abs=1e-12)


def test_normal_cdf_bounds():
    assert _normal_cdf(-40.0) >= 0.0
    assert _normal_cdf(40.0) == pytest.approx(1.0)
    assert not math.isnan(_normal_cdf(0.0))
