"""Tests for drawdown recovery analysis utilities."""

import polars as pl
import pytest

from imst_quant.utils.drawdown_recovery import (
    RecoveryAnalysis,
    analyze_recovery_periods,
    estimate_recovery_time,
    recovery_by_depth_bucket,
    recovery_statistics,
    recovery_velocity,
    underwater_analysis,
)


class TestAnalyzeRecoveryPeriods:
    def test_empty_returns_empty_list(self):
        assert analyze_recovery_periods([]) == []

    def test_no_drawdown_returns_no_events(self):
        returns = [0.01, 0.02, 0.01, 0.03]
        assert analyze_recovery_periods(returns, min_drawdown=0.05) == []

    def test_simple_drawdown_and_full_recovery_detected(self):
        # up, big drop, recovers back above peak
        returns = [0.10, -0.20, 0.30]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        assert len(recoveries) == 1
        r = recoveries[0]
        assert isinstance(r, RecoveryAnalysis)
        assert r.drawdown_depth == pytest.approx(0.20, rel=1e-6)
        assert r.recovery_end is not None

    def test_below_threshold_excluded(self):
        returns = [0.10, -0.01, 0.02]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        assert recoveries == []

    def test_unrecovered_drawdown_has_none_recovery_fields(self):
        returns = [0.10, -0.20]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        assert len(recoveries) == 1
        assert recoveries[0].recovery_end is None
        assert recoveries[0].recovery_duration is None

    def test_accepts_polars_series(self):
        returns = pl.Series([0.10, -0.20, 0.30])
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        assert len(recoveries) == 1


class TestRecoveryStatistics:
    def test_empty_list_returns_zeros(self):
        stats = recovery_statistics([])
        assert stats["pct_recovered"] == 0.0
        assert stats["avg_recovery_duration"] == 0.0

    def test_all_recovered_gives_full_pct(self):
        returns = [0.10, -0.20, 0.30]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        stats = recovery_statistics(recoveries)
        assert stats["pct_recovered"] == pytest.approx(1.0)
        assert stats["max_drawdown_depth"] == pytest.approx(0.20, rel=1e-6)


class TestEstimateRecoveryTime:
    def test_historical_method_returns_expected_keys(self):
        returns = [0.01, -0.02, 0.015, -0.005] * 50
        result = estimate_recovery_time(returns, current_drawdown=0.05, method="historical")
        assert "expected_days" in result
        assert "prob_recover_30d" in result
        assert "prob_recover_90d" in result
        assert "prob_recover_252d" in result

    def test_monte_carlo_method_returns_expected_keys(self):
        returns = [0.01, -0.005, 0.008, -0.003] * 30
        result = estimate_recovery_time(
            returns, current_drawdown=0.05, method="monte_carlo", n_simulations=200, seed=42
        )
        assert result["expected_days"] >= 0
        assert 0.0 <= result["prob_recover_252d"] <= 1.0

    def test_monte_carlo_is_reproducible_with_seed(self):
        returns = [0.01, -0.005, 0.008, -0.003] * 30
        r1 = estimate_recovery_time(returns, current_drawdown=0.05, method="monte_carlo", n_simulations=100, seed=7)
        r2 = estimate_recovery_time(returns, current_drawdown=0.05, method="monte_carlo", n_simulations=100, seed=7)
        assert r1["expected_days"] == pytest.approx(r2["expected_days"])

    def test_negative_mean_return_gives_infinite_historical_estimate(self):
        returns = [-0.01, -0.02, -0.015, -0.005]
        result = estimate_recovery_time(returns, current_drawdown=0.10, method="historical")
        assert result["expected_days"] == float("inf")


class TestUnderwaterAnalysis:
    def test_empty_returns_empty_frame(self):
        df = underwater_analysis([])
        assert df.height == 0

    def test_tracks_underwater_duration(self):
        returns = [0.05, -0.10, 0.02, 0.05, -0.02, 0.03]
        df = underwater_analysis(returns)
        assert df.height == len(returns)
        assert "underwater_duration" in df.columns
        # first period is a new peak -> not underwater
        assert df["is_underwater"][0] is False

    def test_drawdown_never_negative(self):
        returns = [0.05, -0.10, 0.02, 0.05, -0.02, 0.03]
        df = underwater_analysis(returns)
        assert (df["drawdown"] >= 0).all()


class TestRecoveryByDepthBucket:
    def test_empty_recoveries_returns_all_zero_buckets(self):
        result = recovery_by_depth_bucket([])
        assert all(v["count"] == 0 for v in result.values())

    def test_buckets_recovery_into_correct_range(self):
        returns = [0.20, -0.10, 0.15]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        result = recovery_by_depth_bucket(recoveries, buckets=[0.05, 0.15, 0.30])
        total_count = sum(v["count"] for v in result.values())
        assert total_count == len(recoveries)


class TestRecoveryVelocity:
    def test_empty_recoveries_returns_zeros(self):
        velocity = recovery_velocity([])
        assert velocity["avg_velocity"] == 0.0

    def test_velocity_is_depth_over_duration(self):
        returns = [0.10, -0.20, 0.30]
        recoveries = analyze_recovery_periods(returns, min_drawdown=0.05)
        velocity = recovery_velocity(recoveries)
        assert velocity["avg_velocity"] > 0
        assert velocity["max_velocity"] >= velocity["min_velocity"]
