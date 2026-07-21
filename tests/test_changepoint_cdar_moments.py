"""Tests for change-point detection, CDaR, and higher-moment risk utilities."""

import numpy as np
import pytest

from imst_quant.utils.change_point import (
    analyze_change_points,
    cusum_mean_shift,
    icss_variance_breaks,
)
from imst_quant.utils.conditional_drawdown import (
    analyze_drawdown_risk,
    cdar_ratio,
    conditional_drawdown_at_risk,
    drawdown_at_risk,
    drawdown_series,
)
from imst_quant.utils.higher_moments import (
    analyze_higher_moments,
    cokurtosis,
    coskewness,
    downside_beta,
    upside_beta,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------- change point


class TestCusumMeanShift:
    def test_detects_upward_shift(self, rng):
        r = np.concatenate(
            [rng.normal(0, 0.01, 400), rng.normal(0.01, 0.01, 200)]
        )
        result = cusum_mean_shift(r)
        assert result["n_alarms"] >= 1
        first = result["alarms"][0]
        assert first["direction"] == "up"
        assert 400 <= first["index"] <= 520

    def test_no_alarms_on_stationary_series(self, rng):
        r = rng.normal(0, 0.01, 1000)
        assert cusum_mean_shift(r)["n_alarms"] == 0

    def test_detects_downward_shift(self, rng):
        r = np.concatenate(
            [rng.normal(0, 0.01, 400), rng.normal(-0.012, 0.01, 200)]
        )
        alarms = cusum_mean_shift(r)["alarms"]
        assert any(a["direction"] == "down" for a in alarms)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            cusum_mean_shift([0.01] * 5)
        with pytest.raises(ValueError):
            cusum_mean_shift(np.zeros(100))
        with pytest.raises(ValueError):
            cusum_mean_shift(np.random.default_rng(0).normal(0, 1, 100), threshold=-1)
        with pytest.raises(ValueError):
            cusum_mean_shift([[0.01, 0.02]] * 30)


class TestIcssVarianceBreaks:
    def test_detects_volatility_break(self, rng):
        r = np.concatenate(
            [rng.normal(0, 0.005, 500), rng.normal(0, 0.03, 500)]
        )
        result = icss_variance_breaks(r)
        assert result["n_breaks"] >= 1
        assert any(abs(b - 500) < 60 for b in result["break_indices"])
        # First and last segments should differ ~6x in vol
        segs = result["segments"]
        assert segs[-1]["volatility"] > 3 * segs[0]["volatility"]

    def test_no_breaks_on_homogeneous_series(self, rng):
        r = rng.normal(0, 0.01, 800)
        assert icss_variance_breaks(r)["n_breaks"] == 0

    def test_segments_partition_series(self, rng):
        r = np.concatenate(
            [rng.normal(0, 0.005, 300), rng.normal(0, 0.02, 300)]
        )
        segs = icss_variance_breaks(r)["segments"]
        assert segs[0]["start"] == 0
        assert segs[-1]["end"] == 600
        for prev, cur in zip(segs[:-1], segs[1:]):
            assert prev["end"] == cur["start"]

    def test_rejects_bad_input(self, rng):
        with pytest.raises(ValueError):
            icss_variance_breaks(rng.normal(0, 0.01, 30))
        with pytest.raises(ValueError):
            icss_variance_breaks(rng.normal(0, 0.01, 200), min_segment=5)


class TestAnalyzeChangePoints:
    def test_stable_series(self, rng):
        result = analyze_change_points(rng.normal(0, 0.01, 500))
        assert result["stability"] == "stable"

    def test_unstable_series(self, rng):
        r = np.concatenate(
            [rng.normal(0, 0.005, 400), rng.normal(0, 0.03, 400)]
        )
        assert analyze_change_points(r)["stability"] == "unstable"


# ---------------------------------------------------------------------- CDaR


class TestConditionalDrawdown:
    def test_drawdown_series_known_values(self):
        # +10%, -10%: equity 1.1 then 0.99 -> dd 0, 0.1
        dd = drawdown_series([0.10, -0.10])
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.10)

    def test_monotone_gains_have_zero_drawdown(self):
        dd = drawdown_series([0.01] * 50)
        assert np.all(dd == 0)

    def test_cdar_at_least_dar(self, rng):
        r = rng.normal(0.0002, 0.01, 1000)
        for alpha in (0.90, 0.95, 0.99):
            assert conditional_drawdown_at_risk(r, alpha) >= drawdown_at_risk(r, alpha)

    def test_dar_below_max_drawdown(self, rng):
        r = rng.normal(0.0002, 0.01, 1000)
        assert drawdown_at_risk(r, 0.95) <= drawdown_series(r).max()

    def test_cdar_ratio_positive_strategy(self, rng):
        r = rng.normal(0.001, 0.01, 1000)
        assert cdar_ratio(r) > 0

    def test_cdar_ratio_no_drawdowns(self):
        assert cdar_ratio([0.01] * 50) == float("inf")

    def test_analyze_summary(self, rng):
        r = rng.normal(0.0002, 0.01, 1000)
        result = analyze_drawdown_risk(r)
        assert result["max_drawdown"] >= result["levels"][-1]["cdar"] - 1e-12
        assert 0 <= result["time_underwater_pct"] <= 1
        assert result["n_observations"] == 1000
        assert [lv["alpha"] for lv in result["levels"]] == [0.90, 0.95, 0.99]

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            drawdown_at_risk([0.01] * 100, alpha=1.5)
        with pytest.raises(ValueError):
            drawdown_series([0.01, -1.5, 0.02] * 10)
        with pytest.raises(ValueError):
            analyze_drawdown_risk([0.01] * 100, alphas=())


# ------------------------------------------------------------- higher moments


class TestHigherMoments:
    def test_symmetric_asset_betas_close(self, rng):
        b = rng.normal(0, 0.01, 5000)
        a = 1.2 * b + rng.normal(0, 0.002, 5000)
        d = downside_beta(a, b)
        u = upside_beta(a, b)
        assert d == pytest.approx(1.2, abs=0.1)
        assert u == pytest.approx(1.2, abs=0.1)

    def test_crash_exposed_asset(self, rng):
        b = rng.normal(0, 0.01, 5000)
        # Amplify losses only -> downside beta > upside beta
        a = np.where(b < 0, 2.0 * b, 0.8 * b) + rng.normal(0, 0.001, 5000)
        result = analyze_higher_moments(a, b)
        assert result["downside_beta"] > result["upside_beta"]
        assert result["beta_asymmetry"] > 0.1
        assert result["assessment"] == "crash_exposed"

    def test_defensive_asset(self, rng):
        b = rng.normal(0, 0.01, 5000)
        a = np.where(b < 0, 0.5 * b, 1.2 * b) + rng.normal(0, 0.001, 5000)
        assert analyze_higher_moments(a, b)["assessment"] == "defensive"

    def test_coskewness_sign(self, rng):
        b = rng.normal(0, 0.01, 5000)
        # Asset loses when |b| is large -> negative coskewness
        a = -np.abs(b) + rng.normal(0, 0.001, 5000)
        assert coskewness(a, b) < 0

    def test_cokurtosis_of_leveraged_asset(self, rng):
        b = rng.standard_t(5, 5000) * 0.01
        a = 1.5 * b
        # Perfectly correlated: cokurtosis equals benchmark kurtosis (>0 raw)
        assert cokurtosis(a, b) > 0

    def test_rejects_bad_input(self, rng):
        b = rng.normal(0, 0.01, 100)
        with pytest.raises(ValueError):
            downside_beta(b[:50], b)
        with pytest.raises(ValueError):
            coskewness(np.zeros(100), b)
        with pytest.raises(ValueError):
            upside_beta(b, np.full(100, 0.01))  # no periods below/above mix
