"""Tests for the feature scaling module.

Tests cover:
- StandardScaler (z-score) fit/transform/inverse round trip
- MinMaxScaler range and zero-variance column handling
- RobustScaler median/IQR scaling and zero-IQR handling
- winsorize outlier clipping
- unfitted-column error paths
"""

import json

import polars as pl
import pytest

from imst_quant.utils.feature_scaling import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    winsorize,
)


def test_standard_scaler_fit_transform_mean_zero_std_one():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df, ["a"])

    assert scaled["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["a"].std() == pytest.approx(1.0, abs=1e-9)


def test_standard_scaler_inverse_transform_round_trip():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df, ["a"])
    restored = scaler.inverse_transform(scaled, ["a"])

    for orig, back in zip(df["a"].to_list(), restored["a"].to_list()):
        assert orig == pytest.approx(back, abs=1e-9)


def test_standard_scaler_transform_before_fit_raises():
    df = pl.DataFrame({"a": [1.0, 2.0]})
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform(df, ["a"])


def test_standard_scaler_save_load_round_trip(tmp_path):
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    scaler = StandardScaler()
    scaler.fit(df, ["a"])

    path = tmp_path / "scaler.json"
    scaler.save(path)
    assert json.loads(path.read_text())["mean"]["a"] == pytest.approx(2.0)

    loaded = StandardScaler().load(path)
    assert loaded.mean_["a"] == pytest.approx(scaler.mean_["a"])
    assert loaded.std_["a"] == pytest.approx(scaler.std_["a"])


def test_minmax_scaler_scales_to_zero_one_range():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    scaled = MinMaxScaler().fit_transform(df, ["a"])

    assert scaled["a"].min() == pytest.approx(0.0)
    assert scaled["a"].max() == pytest.approx(1.0)


def test_minmax_scaler_constant_column_scales_to_zero():
    df = pl.DataFrame({"a": [5.0, 5.0, 5.0]})
    scaled = MinMaxScaler().fit_transform(df, ["a"])

    assert scaled["a"].to_list() == [0.0, 0.0, 0.0]


def test_robust_scaler_centers_on_median_and_resists_outlier():
    df = pl.DataFrame({"price": [100.0, 102.0, 105.0, 1000.0]})
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df, ["price"])

    assert scaler.median_["price"] == pytest.approx(103.5)
    # the outlier should not compress the non-outlier values toward 0
    assert abs(scaled["price"][0]) > 0


def test_robust_scaler_zero_iqr_scales_to_zero():
    df = pl.DataFrame({"a": [5.0, 5.0, 5.0, 5.0]})
    scaled = RobustScaler().fit_transform(df, ["a"])

    assert scaled["a"].to_list() == [0.0, 0.0, 0.0, 0.0]


def test_winsorize_clips_extreme_values():
    df = pl.DataFrame({"returns": [-0.5, -0.02, 0.01, 0.02, 0.6]})
    result = winsorize(df, ["returns"], lower=0.2, upper=0.8)

    assert result["returns"].min() > df["returns"].min()
    assert result["returns"].max() < df["returns"].max()
    # middle values are untouched
    assert result["returns"][2] == pytest.approx(0.01)
