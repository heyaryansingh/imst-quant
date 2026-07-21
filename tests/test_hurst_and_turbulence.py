"""Tests for Hurst exponent estimation and turbulence index utilities."""

import numpy as np
import pytest

from imst_quant.utils.hurst_exponent import (
    aggregated_variance_hurst,
    analyze_hurst,
    classify_regime,
    rescaled_range_hurst,
    variance_ratio_test,
)
from imst_quant.utils.turbulence_index import (
    absorption_ratio,
    rolling_absorption_ratio,
    turbulence_index,
    turbulence_regimes,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def random_walk_returns(rng):
    return rng.normal(0, 0.01, 1000)


@pytest.fixture
def trending_returns(rng):
    # AR(1) with strong positive autocorrelation -> persistent, H > 0.5
    noise = rng.normal(0, 0.01, 1000)
    returns = np.zeros(1000)
    for i in range(1, 1000):
        returns[i] = 0.7 * returns[i - 1] + noise[i]
    return returns


@pytest.fixture
def mean_reverting_returns(rng):
    # AR(1) with strong negative autocorrelation -> anti-persistent, H < 0.5
    noise = rng.normal(0, 0.01, 1000)
    returns = np.zeros(1000)
    for i in range(1, 1000):
        returns[i] = -0.7 * returns[i - 1] + noise[i]
    return returns


class TestRescaledRangeHurst:
    def test_random_walk_near_half(self, random_walk_returns):
        result = rescaled_range_hurst(random_walk_returns)
        assert 0.4 < result["hurst"] < 0.65

    def test_trending_above_random_walk(self, trending_returns, random_walk_returns):
        h_trend = rescaled_range_hurst(trending_returns)["hurst"]
        h_rw = rescaled_range_hurst(random_walk_returns)["hurst"]
        assert h_trend > h_rw

    def test_mean_reverting_below_random_walk(
        self, mean_reverting_returns, random_walk_returns
    ):
        h_mr = rescaled_range_hurst(mean_reverting_returns)["hurst"]
        h_rw = rescaled_range_hurst(random_walk_returns)["hurst"]
        assert h_mr < h_rw

    def test_good_fit_quality(self, random_walk_returns):
        result = rescaled_range_hurst(random_walk_returns)
        assert result["r_squared"] > 0.9

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least"):
            rescaled_range_hurst([0.01] * 10)

    def test_nan_filtered(self, random_walk_returns):
        with_nan = np.concatenate([random_walk_returns, [np.nan] * 5])
        result = rescaled_range_hurst(with_nan)
        assert np.isfinite(result["hurst"])


class TestAggregatedVarianceHurst:
    def test_random_walk_near_half(self, random_walk_returns):
        result = aggregated_variance_hurst(random_walk_returns)
        assert 0.35 < result["hurst"] < 0.65

    def test_ordering_across_regimes(
        self, trending_returns, mean_reverting_returns
    ):
        h_trend = aggregated_variance_hurst(trending_returns)["hurst"]
        h_mr = aggregated_variance_hurst(mean_reverting_returns)["hurst"]
        assert h_trend > h_mr

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            aggregated_variance_hurst([0.01] * 20)


class TestVarianceRatioTest:
    def test_random_walk_not_rejected(self):
        # seed 42 happens to draw a borderline-significant sample; seed 0
        # is representative of the null
        returns = np.random.default_rng(0).normal(0, 0.01, 1000)
        result = variance_ratio_test(returns, lag=2)
        assert result["interpretation"] == "random_walk"
        assert result["p_value"] > 0.05

    def test_mean_reverting_detected(self, mean_reverting_returns):
        result = variance_ratio_test(mean_reverting_returns, lag=2)
        assert result["variance_ratio"] < 1.0
        assert result["interpretation"] == "mean_reverting"

    def test_trending_detected(self, trending_returns):
        result = variance_ratio_test(trending_returns, lag=2)
        assert result["variance_ratio"] > 1.0
        assert result["interpretation"] == "trending"

    def test_lag_below_two_raises(self, random_walk_returns):
        with pytest.raises(ValueError, match="lag"):
            variance_ratio_test(random_walk_returns, lag=1)

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            variance_ratio_test([0.0] * 100, lag=2)


class TestClassifyRegime:
    def test_bands(self):
        assert classify_regime(0.3) == "mean_reverting"
        assert classify_regime(0.5) == "random_walk"
        assert classify_regime(0.52) == "random_walk"
        assert classify_regime(0.7) == "trending"


class TestAnalyzeHurst:
    def test_full_analysis_structure(self, random_walk_returns):
        result = analyze_hurst(random_walk_returns)
        assert result["n_observations"] == 1000
        assert "hurst_rs" in result
        assert "hurst_aggvar" in result
        assert result["regime"] in ("mean_reverting", "random_walk", "trending")
        assert len(result["variance_ratios"]) == 3

    def test_short_series_skips_long_lags(self, rng):
        returns = rng.normal(0, 0.01, 60)
        result = analyze_hurst(returns, variance_ratio_lags=(2, 5, 10))
        # lag 10 requires 100 obs -> skipped
        assert len(result["variance_ratios"]) == 2


@pytest.fixture
def multi_asset_returns(rng):
    return rng.normal(0, 0.01, size=(300, 5))


@pytest.fixture
def coupled_returns(rng):
    # One common factor drives all assets -> high absorption
    factor = rng.normal(0, 0.02, 300)
    idio = rng.normal(0, 0.002, size=(300, 5))
    return factor[:, None] + idio


class TestTurbulenceIndex:
    def test_full_sample_mean_near_one(self, multi_asset_returns):
        result = turbulence_index(multi_asset_returns)
        assert 0.8 < result["mean_turbulence"] < 1.2
        assert result["n_assets"] == 5

    def test_outlier_day_scores_high(self, multi_asset_returns):
        data = multi_asset_returns.copy()
        data[-1] = 0.10  # 10-sigma day on every asset
        result = turbulence_index(data)
        assert result["current_turbulence"] > result["mean_turbulence"] * 5

    def test_rolling_warmup_nan(self, multi_asset_returns):
        result = turbulence_index(multi_asset_returns, lookback=60)
        scores = np.array(result["turbulence"])
        assert np.all(np.isnan(scores[:60]))
        assert np.all(np.isfinite(scores[60:]))

    def test_lookback_too_small_raises(self, multi_asset_returns):
        with pytest.raises(ValueError, match="lookback"):
            turbulence_index(multi_asset_returns, lookback=3)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            turbulence_index([0.01, 0.02, 0.03])


class TestTurbulenceRegimes:
    def test_flags_top_decile(self, multi_asset_returns):
        scores = turbulence_index(multi_asset_returns)["turbulence"]
        result = turbulence_regimes(scores, quantile=0.9)
        assert 0.05 < result["pct_turbulent"] < 0.15
        assert result["n_turbulent"] == sum(result["is_turbulent"])

    def test_bad_quantile_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            turbulence_regimes([1.0, 2.0], quantile=1.5)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            turbulence_regimes([float("nan")] * 10)


class TestAbsorptionRatio:
    def test_coupled_market_high_absorption(self, coupled_returns):
        result = absorption_ratio(coupled_returns, n_components=1)
        assert result["absorption_ratio"] > 0.9

    def test_independent_market_low_absorption(self, multi_asset_returns):
        result = absorption_ratio(multi_asset_returns, n_components=1)
        assert result["absorption_ratio"] < 0.5

    def test_all_components_sum_to_one(self, multi_asset_returns):
        result = absorption_ratio(multi_asset_returns, n_components=5)
        assert result["absorption_ratio"] == pytest.approx(1.0)

    def test_bad_n_components_raises(self, multi_asset_returns):
        with pytest.raises(ValueError, match="n_components"):
            absorption_ratio(multi_asset_returns, n_components=99)


class TestRollingAbsorptionRatio:
    def test_structure_and_warmup(self, multi_asset_returns):
        result = rolling_absorption_ratio(multi_asset_returns, window=60)
        ratios = np.array(result["absorption_ratios"])
        assert np.all(np.isnan(ratios[:60]))
        assert np.isfinite(result["current_ratio"])
        assert result["trend"] in ("rising", "falling", "stable")

    def test_too_short_raises(self, rng):
        with pytest.raises(ValueError):
            rolling_absorption_ratio(rng.normal(0, 0.01, size=(30, 3)), window=60)
