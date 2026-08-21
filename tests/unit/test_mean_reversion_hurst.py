"""Tests for the Hurst exponent in the mean reversion module."""

import numpy as np
import pytest

# Aliased: pytest would otherwise collect the imported `test_mean_reversion`
# helper as a test case and fail on its `prices` argument.
from imst_quant.utils.mean_reversion import (
    hurst_exponent,
    rolling_hurst,
)
from imst_quant.utils.mean_reversion import test_mean_reversion as run_mean_reversion_tests


def _random_walk(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1, n))


def _ou_prices(n: int = 2000, kappa: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = np.empty(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + kappa * (100.0 - prices[i - 1]) + rng.normal(0, 1)
    return prices


def test_random_walk_is_near_one_half():
    """Regression: R/S ran on price levels rather than increments, which added
    1 to the exponent and pinned a plain random walk at the 1.0 clamp."""
    h = hurst_exponent(_random_walk())

    assert 0.4 < h < 0.65
    assert h < 0.95


def test_mean_reverting_series_is_anti_persistent():
    assert hurst_exponent(_ou_prices()) < 0.5


def test_trending_series_scores_above_mean_reverting():
    rng = np.random.default_rng(5)
    trending = 100.0 + np.cumsum(rng.normal(0, 1, 2000) * 0.3 + 0.25)

    assert hurst_exponent(trending) > hurst_exponent(_ou_prices())


def test_short_series_falls_back_to_random_walk():
    assert hurst_exponent(np.linspace(100, 110, 10)) == 0.5


def test_result_is_clamped_to_unit_interval():
    for series in (_random_walk(), _ou_prices(), np.linspace(100, 200, 500)):
        assert 0.0 <= hurst_exponent(series) <= 1.0


def test_test_mean_reversion_flags_ou_process():
    result = run_mean_reversion_tests(_ou_prices())

    assert result.is_mean_reverting is True
    assert result.hurst_exponent < 0.5
    assert result.variance_ratio < 1.0


def test_test_mean_reversion_does_not_flag_random_walk():
    result = run_mean_reversion_tests(_random_walk())

    assert result.is_mean_reverting is False


def test_rolling_hurst_shape_and_range():
    df = rolling_hurst(_random_walk(n=400), window=100)

    assert df.height == 400 - 100 + 1
    values = df["hurst"].to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    # A window this short is noisy, but it must not sit pinned at the clamp.
    assert values.mean() < 0.9
