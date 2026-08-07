"""Tests for AlphaMetrics beta estimation and skill-vs-luck statistics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from imst_quant.utils.alpha_metrics import AlphaMetrics
from imst_quant.utils.deflated_sharpe import probabilistic_sharpe_ratio


def _series(n=500, true_beta=1.3, seed=0):
    rng = np.random.default_rng(seed)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, n))
    strategy = pd.Series(
        true_beta * benchmark.to_numpy() + rng.normal(0.0002, 0.005, n)
    )
    return strategy, benchmark


def test_beta_uses_one_consistent_normalization():
    """Beta is cov/var with matching ddof, not np.cov over np.var."""
    strategy, benchmark = _series()
    metrics = AlphaMetrics(strategy, benchmark)

    expected = np.cov(strategy, benchmark)[0, 1] / np.var(benchmark, ddof=1)

    assert metrics._calculate_beta() == pytest.approx(expected)


def test_beta_is_not_inflated_by_sample_size():
    """The old cov(ddof=1)/var(ddof=0) mix scaled beta by n / (n - 1)."""
    strategy, benchmark = _series(n=30)
    metrics = AlphaMetrics(strategy, benchmark)

    inflated = np.cov(strategy, benchmark)[0, 1] / np.var(benchmark)

    assert metrics._calculate_beta() < inflated
    assert metrics._calculate_beta() == pytest.approx(inflated * 29 / 30)


def test_beta_is_zero_for_flat_benchmark():
    """A benchmark with no variance yields beta 0 rather than a ZeroDivisionError."""
    strategy, _ = _series(n=50)
    metrics = AlphaMetrics(strategy, pd.Series([0.001] * 50))

    assert metrics._calculate_beta() == 0.0


def test_psr_matches_the_shared_implementation():
    """PSR delegates to deflated_sharpe rather than a units-mismatched formula."""
    strategy, benchmark = _series()
    metrics = AlphaMetrics(strategy, benchmark, risk_free_rate=0.02)

    result = metrics.calculate_skill_vs_luck(n_simulations=100, seed=7)

    excess = strategy - 0.02 / 252
    expected = probabilistic_sharpe_ratio(
        sharpe_ratio=float(excess.mean() / strategy.std()),
        benchmark_sharpe=0.0,
        n_observations=len(strategy),
        skewness=float(stats.skew(strategy)),
        kurtosis=float(stats.kurtosis(strategy, fisher=False)),
    )

    assert result["probabilistic_sharpe_ratio"] == pytest.approx(expected)


def test_psr_is_not_saturated_at_the_bounds():
    """The old formula collapsed to exactly 0.0 or 1.0 for realistic returns."""
    strategy, benchmark = _series()
    metrics = AlphaMetrics(strategy, benchmark, risk_free_rate=0.02)

    psr = metrics.calculate_skill_vs_luck(n_simulations=100, seed=7)[
        "probabilistic_sharpe_ratio"
    ]

    assert 0.0 < psr < 1.0


def test_skill_vs_luck_is_reproducible_with_a_seed():
    strategy, benchmark = _series()
    metrics = AlphaMetrics(strategy, benchmark)

    first = metrics.calculate_skill_vs_luck(n_simulations=50, seed=3)
    second = metrics.calculate_skill_vs_luck(n_simulations=50, seed=3)

    assert first == second


def test_skill_vs_luck_handles_zero_volatility():
    """A constant return series short-circuits instead of dividing by zero."""
    flat = pd.Series([0.001] * 100)
    _, benchmark = _series(n=100)
    metrics = AlphaMetrics(flat, benchmark)

    result = metrics.calculate_skill_vs_luck(n_simulations=10, seed=1)

    assert result["observed_sharpe"] == 0.0
    assert result["p_value"] == 1.0


def test_m2_alpha_handles_zero_volatility():
    """M2 returns 0 rather than raising or producing inf on a flat strategy."""
    flat = pd.Series([0.001] * 100)
    _, benchmark = _series(n=100)

    assert AlphaMetrics(flat, benchmark).calculate_m2_alpha() == 0.0
