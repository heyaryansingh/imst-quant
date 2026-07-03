"""Tests for Deflated Sharpe Ratio utilities."""

import numpy as np
import pytest

from imst_quant.utils.deflated_sharpe import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    estimated_sharpe_ratio_stderr,
    expected_max_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


class TestEstimatedSharpeRatioStderr:
    def test_normal_returns_matches_simple_formula(self):
        # Under normality (skew=0, kurtosis=3), stderr reduces to
        # sqrt((1 + 0.5*SR^2) / (n - 1)).
        stderr = estimated_sharpe_ratio_stderr(
            n_observations=101, sharpe_ratio=1.0, skewness=0.0, kurtosis=3.0
        )
        expected = np.sqrt((1 + 0.5 * 1.0**2) / 100)
        assert stderr == pytest.approx(expected)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            estimated_sharpe_ratio_stderr(n_observations=1, sharpe_ratio=1.0)


class TestExpectedMaxSharpeRatio:
    def test_more_trials_increases_expected_max(self):
        low = expected_max_sharpe_ratio(n_trials=2)
        high = expected_max_sharpe_ratio(n_trials=1000)
        assert high > low > 0

    def test_single_trial_is_zero(self):
        assert expected_max_sharpe_ratio(n_trials=1) == 0.0

    def test_invalid_trials_raises(self):
        with pytest.raises(ValueError):
            expected_max_sharpe_ratio(n_trials=0)


class TestProbabilisticSharpeRatio:
    def test_sharpe_above_benchmark_gives_high_psr(self):
        psr = probabilistic_sharpe_ratio(
            sharpe_ratio=2.0, benchmark_sharpe=0.0, n_observations=252
        )
        assert psr > 0.9

    def test_sharpe_at_benchmark_gives_half(self):
        psr = probabilistic_sharpe_ratio(
            sharpe_ratio=0.5, benchmark_sharpe=0.5, n_observations=100
        )
        assert psr == pytest.approx(0.5, abs=1e-6)


class TestDeflatedSharpeRatio:
    def test_more_trials_deflates_confidence(self):
        few_trials = deflated_sharpe_ratio(
            sharpe_ratio=1.5, n_observations=252, n_trials=1
        )
        many_trials = deflated_sharpe_ratio(
            sharpe_ratio=1.5, n_observations=252, n_trials=1000
        )
        assert many_trials["deflated_sharpe_ratio"] < few_trials["deflated_sharpe_ratio"]
        assert many_trials["expected_max_sharpe"] > few_trials["expected_max_sharpe"]

    def test_result_contains_expected_keys(self):
        result = deflated_sharpe_ratio(
            sharpe_ratio=1.0, n_observations=100, n_trials=10
        )
        assert set(result) == {
            "deflated_sharpe_ratio",
            "expected_max_sharpe",
            "sharpe_stderr",
            "n_trials",
            "n_observations",
        }
        assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0


class TestDeflatedSharpeRatioFromReturns:
    def test_computes_from_return_series(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.001, scale=0.01, size=252)
        result = deflated_sharpe_ratio_from_returns(
            returns, n_trials=5, periods_per_year=252
        )
        assert "sharpe_ratio" in result
        assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio_from_returns([0.01], n_trials=1)
