"""Tests for triple-barrier labeling and fractional differentiation."""

import numpy as np
import pytest

from imst_quant.utils.fractional_diff import (
    ffd_weights,
    frac_diff_ffd,
    memory_vs_stationarity,
)
from imst_quant.utils.triple_barrier import (
    ewm_volatility,
    label_distribution,
    triple_barrier_labels,
)


# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------


class TestEwmVolatility:
    def test_length_matches_input(self):
        prices = np.linspace(100, 110, 50)
        vol = ewm_volatility(prices)
        assert vol.size == 50
        assert np.isnan(vol[0])
        assert np.all(np.isfinite(vol[1:]))

    def test_constant_prices_zero_vol(self):
        vol = ewm_volatility(np.full(30, 100.0))
        assert np.allclose(vol[1:], 0.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            ewm_volatility([100.0])

    def test_bad_span_raises(self):
        with pytest.raises(ValueError):
            ewm_volatility([100.0, 101.0], span=1)


class TestTripleBarrierLabels:
    def _trending_up(self, n=100):
        return 100.0 * np.cumprod(1 + np.full(n, 0.01))

    def test_uptrend_labels_positive(self):
        prices = self._trending_up()
        result = triple_barrier_labels(
            prices, events=range(0, 70), pt_mult=1.0, sl_mult=1.0,
            max_holding=20, volatility=np.full(prices.size, 0.02),
        )
        assert np.all(result["labels"] == 1)
        assert np.all(result["returns"] > 0)

    def test_downtrend_labels_negative(self):
        prices = 100.0 * np.cumprod(1 - np.full(100, 0.01))
        result = triple_barrier_labels(
            prices, events=range(0, 70), pt_mult=1.0, sl_mult=1.0,
            max_holding=20, volatility=np.full(prices.size, 0.02),
        )
        assert np.all(result["labels"] == -1)

    def test_flat_series_vertical_barrier(self):
        prices = np.full(50, 100.0)
        result = triple_barrier_labels(
            prices, max_holding=5, volatility=np.full(50, 0.02)
        )
        assert np.all(result["labels"] == 0)
        # Vertical touch is exactly max_holding bars out (except near end)
        assert result["touch_indices"][0] == result["event_indices"][0] + 5

    def test_barrier_levels(self):
        prices = np.full(20, 100.0)
        result = triple_barrier_labels(
            prices, pt_mult=2.0, sl_mult=1.0, volatility=np.full(20, 0.05)
        )
        assert np.allclose(result["upper_barriers"], 110.0)
        assert np.allclose(result["lower_barriers"], 95.0)

    def test_explicit_events_subset(self):
        prices = self._trending_up(60)
        result = triple_barrier_labels(
            prices, events=[10, 20, 30], volatility=np.full(60, 0.02)
        )
        assert list(result["event_indices"]) == [10, 20, 30]

    def test_out_of_range_events_raise(self):
        with pytest.raises(ValueError):
            triple_barrier_labels(
                np.full(10, 100.0), events=[50], volatility=np.full(10, 0.02)
            )

    def test_invalid_multipliers_raise(self):
        with pytest.raises(ValueError):
            triple_barrier_labels(np.full(10, 100.0), pt_mult=0)

    def test_zero_vol_events_skipped(self):
        # Default EWM vol on constant prices is zero -> no labelable events
        with pytest.raises(ValueError):
            triple_barrier_labels(np.full(30, 100.0))


class TestLabelDistribution:
    def test_counts_and_proportions(self):
        summary = label_distribution([1, 1, -1, 0, 0, 0])
        assert summary["counts"] == {"upper": 2, "lower": 1, "vertical": 3}
        assert summary["proportions"]["vertical"] == pytest.approx(0.5)
        assert summary["directional_share"] == pytest.approx(0.5)
        assert summary["imbalance_ratio"] == pytest.approx(3.0)

    def test_single_class_infinite_imbalance(self):
        summary = label_distribution([1, 1, 1])
        assert summary["imbalance_ratio"] == float("inf")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            label_distribution([])


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------


class TestFfdWeights:
    def test_d_zero_identity(self):
        w = ffd_weights(0.0)
        assert w.size == 1
        assert w[0] == 1.0

    def test_d_one_first_difference(self):
        w = ffd_weights(1.0)
        assert np.allclose(w, [1.0, -1.0])

    def test_weights_decay(self):
        w = ffd_weights(0.5, threshold=1e-5)
        assert w[0] == 1.0
        assert np.all(np.abs(w[1:]) >= 1e-5)
        # Magnitudes shrink monotonically after w_1
        mags = np.abs(w[1:])
        assert np.all(np.diff(mags) <= 0)

    def test_negative_d_raises(self):
        with pytest.raises(ValueError):
            ffd_weights(-0.1)

    def test_bad_threshold_raises(self):
        with pytest.raises(ValueError):
            ffd_weights(0.5, threshold=0)


class TestFracDiffFfd:
    def test_d_one_equals_diff(self):
        series = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
        result = frac_diff_ffd(series, d=1.0)
        assert np.allclose(result["values"], np.diff(series))
        assert result["start_index"] == 1

    def test_d_zero_identity(self):
        series = np.array([1.0, 2.0, 3.0])
        result = frac_diff_ffd(series, d=0.0)
        assert np.allclose(result["values"], series)

    def test_output_length(self):
        rng = np.random.default_rng(7)
        series = np.cumsum(rng.normal(0, 1, 500)) + 100
        result = frac_diff_ffd(series, d=0.4, threshold=1e-3)
        assert result["values"].size == series.size - result["width"] + 1

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            frac_diff_ffd([1.0, 2.0], d=0.5, threshold=1e-6)

    def test_intermediate_d_preserves_memory(self):
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0.05, 1, 1000)) + 100
        low_d = frac_diff_ffd(series, d=0.3, threshold=1e-3)
        aligned = series[low_d["start_index"]:]
        corr = np.corrcoef(low_d["values"], aligned)[0, 1]
        full_diff = np.diff(series)
        corr_full = np.corrcoef(full_diff, series[1:])[0, 1]
        # Partial differentiation keeps more level memory than full diff
        assert corr > corr_full


class TestMemoryVsStationarity:
    def test_profile_shape_and_monotone_memory(self):
        rng = np.random.default_rng(3)
        series = np.cumsum(rng.normal(0, 1, 800)) + 50
        result = memory_vs_stationarity(series, d_grid=(0.0, 0.5, 1.0))
        assert result["n_obs"] == 800
        assert len(result["profiles"]) == 3
        corrs = [p["correlation_with_original"] for p in result["profiles"]]
        # d=0 is perfectly correlated with itself; memory falls as d rises
        assert corrs[0] == pytest.approx(1.0)
        assert corrs[0] > corrs[1] > corrs[2]
