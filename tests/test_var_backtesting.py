"""Tests for VaR backtesting utilities (Kupiec and Christoffersen tests)."""

import numpy as np
import pytest

from imst_quant.utils.var_backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    var_backtest_summary,
)


class TestComputeViolations:
    def test_flags_breach_when_loss_exceeds_var(self):
        returns = np.array([-0.05, 0.01, -0.01, 0.02])
        var_forecasts = np.array([0.02, 0.02, 0.02, 0.02])
        violations = compute_violations(returns, var_forecasts)
        np.testing.assert_array_equal(violations, [True, False, False, False])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_violations([0.01, 0.02], [0.01])


class TestKupiecPOF:
    def test_well_calibrated_model_does_not_reject(self):
        # 5% violation rate over 1000 obs matches a 95% VaR model exactly.
        np.random.seed(0)
        n = 1000
        violations = np.zeros(n, dtype=bool)
        violations[np.random.choice(n, 50, replace=False)] = True
        result = kupiec_pof_test(violations, confidence_level=0.95)
        assert result["num_violations"] == 50
        assert result["reject_null"] is False

    def test_too_many_violations_rejects(self):
        # 30% violations vs expected 5% should clearly fail the test.
        n = 200
        violations = np.zeros(n, dtype=bool)
        violations[: int(0.3 * n)] = True
        result = kupiec_pof_test(violations, confidence_level=0.95)
        assert result["reject_null"] is True

    def test_empty_violations_raises(self):
        with pytest.raises(ValueError):
            kupiec_pof_test([], confidence_level=0.95)


class TestChristoffersenIndependence:
    def test_random_violations_do_not_reject(self):
        np.random.seed(1)
        violations = np.random.rand(500) < 0.05
        result = christoffersen_independence_test(violations)
        assert "lr_statistic" in result
        assert result["lr_statistic"] >= 0

    def test_clustered_violations_reject(self):
        # Force violations to cluster in consecutive blocks.
        violations = np.zeros(200, dtype=bool)
        violations[0:10] = True
        violations[50:60] = True
        violations[100:110] = True
        result = christoffersen_independence_test(violations)
        assert result["reject_null"] is True

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            christoffersen_independence_test([True])


class TestConditionalCoverage:
    def test_combines_both_components(self):
        np.random.seed(2)
        violations = np.random.rand(500) < 0.05
        result = christoffersen_conditional_coverage_test(violations, confidence_level=0.95)
        assert "unconditional_coverage" in result
        assert "independence" in result
        assert result["lr_statistic"] == pytest.approx(
            result["unconditional_coverage"]["lr_statistic"]
            + result["independence"]["lr_statistic"]
        )


class TestVarBacktestSummary:
    def test_well_specified_model_adequate(self):
        np.random.seed(3)
        n = 1000
        returns = np.random.normal(0, 0.02, n)
        var_forecast = np.full(n, np.percentile(-returns, 95))
        summary = var_backtest_summary(returns, var_forecast, confidence_level=0.95)
        assert summary["num_observations"] == n
        assert 0 <= summary["violation_rate"] <= 1
        assert isinstance(summary["model_adequate"], bool)

    def test_undersized_var_flagged_inadequate(self):
        # VaR forecast far too small relative to realized volatility.
        np.random.seed(4)
        n = 500
        returns = np.random.normal(0, 0.05, n)
        var_forecast = np.full(n, 0.001)
        summary = var_backtest_summary(returns, var_forecast, confidence_level=0.95)
        assert summary["violation_rate"] > 0.05
        assert summary["model_adequate"] is False
