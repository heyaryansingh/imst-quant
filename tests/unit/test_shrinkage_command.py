"""Tests for the `shrinkage` CLI command."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import (
    _minimum_variance_weights,
    cmd_shrinkage,
    create_parser,
)


def _features_file(tmp_path, n_dates=60, n_assets=6, short_history=None, seed=0):
    """Write a long-format features parquet with a one-factor return panel."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    factor = rng.normal(0, 0.01, n_dates)

    rows = []
    for j in range(n_assets):
        series = factor * (0.5 + 0.3 * j) + rng.normal(0, 0.004 + 0.002 * j, n_dates)
        for date, value in zip(dates, series):
            rows.append({"date": date, "asset_id": f"A{j}", "return_1d": float(value)})

    if short_history:
        for date, value in zip(dates[:short_history], rng.normal(0, 0.01, short_history)):
            rows.append({"date": date, "asset_id": "SHORTY", "return_1d": float(value)})

    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_shrinkage_command_is_registered():
    args = create_parser().parse_args(["shrinkage", "--features", "x.parquet"])

    assert args.command == "shrinkage"
    assert args.method == "compare"
    assert args.return_col == "return_1d"
    assert args.min_obs == 20
    assert args.intensity is None
    assert args.long_only is False


def test_shrinkage_compares_all_estimators(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(["shrinkage", "--features", str(path), "--json"])
    assert cmd_shrinkage(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_assets"] == 6
    assert payload["n_observations"] == 60
    assert set(payload["methods"]) == {"ledoit_wolf", "oas", "identity"}
    for entry in payload["methods"].values():
        assert 0.0 <= entry["shrinkage_intensity"] <= 1.0
        assert entry["solvable"] is True
    # Only the identity-target estimators are guaranteed to improve conditioning;
    # a constant-correlation target can leave it slightly worse.
    for name in ("oas", "identity"):
        assert (
            payload["methods"][name]["condition_number"]
            <= payload["sample"]["condition_number"] + 1e-9
        )


def test_shrinkage_drops_assets_below_min_obs(tmp_path, capsys):
    path = _features_file(tmp_path, short_history=5)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--min-obs", "20", "--json"]
    )
    assert cmd_shrinkage(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dropped_assets"] == ["SHORTY"]
    assert payload["n_assets"] == 6
    # Without the drop, the overlap would collapse to the 5 shared dates.
    assert payload["n_observations"] == 60


def test_shrinkage_single_method_and_fixed_intensity(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(
        [
            "shrinkage",
            "--features",
            str(path),
            "--method",
            "identity",
            "--intensity",
            "0.4",
            "--json",
        ]
    )
    assert cmd_shrinkage(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert list(payload["methods"]) == ["identity"]
    assert payload["methods"]["identity"]["shrinkage_intensity"] == pytest.approx(0.4)


def test_shrinkage_long_only_weights_are_non_negative(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--long-only", "--method", "oas", "--json"]
    )
    assert cmd_shrinkage(args) == 0

    payload = json.loads(capsys.readouterr().out)
    entry = payload["methods"]["oas"]
    assert entry["short_count"] == 0
    assert entry["gross_leverage"] == pytest.approx(1.0)
    assert all(item["weight"] >= 0 for item in entry["top_weights"])


def test_shrinkage_handles_more_assets_than_dates(tmp_path, capsys):
    """The sample covariance is singular here; the shrunk ones must still solve."""
    path = _features_file(tmp_path, n_dates=8, n_assets=12, seed=3)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--min-obs", "5", "--json"]
    )
    assert cmd_shrinkage(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_observations"] < payload["n_assets"]
    assert payload["methods"]["oas"]["solvable"] is True


def test_shrinkage_rejects_intensity_for_other_methods(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--intensity", "0.3", "--json"]
    )
    assert cmd_shrinkage(args) == 1
    assert "--intensity only applies" in capsys.readouterr().out


def test_shrinkage_rejects_out_of_range_intensity(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--method", "identity",
         "--intensity", "1.5"]
    )
    assert cmd_shrinkage(args) == 1
    assert "must be in [0, 1]" in capsys.readouterr().out


def test_shrinkage_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(
        ["shrinkage", "--features", str(tmp_path / "nope.parquet")]
    )
    assert cmd_shrinkage(args) == 1
    assert "not found" in capsys.readouterr().out


def test_shrinkage_reports_missing_column(tmp_path, capsys):
    path = _features_file(tmp_path)

    args = create_parser().parse_args(
        ["shrinkage", "--features", str(path), "--return-col", "missing"]
    )
    assert cmd_shrinkage(args) == 1
    assert "not found in features file" in capsys.readouterr().out


def test_shrinkage_needs_two_assets(tmp_path, capsys):
    path = _features_file(tmp_path, n_assets=1)

    args = create_parser().parse_args(["shrinkage", "--features", str(path)])
    assert cmd_shrinkage(args) == 1
    assert "at least 2 assets" in capsys.readouterr().out


class TestMinimumVarianceWeights:
    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = _minimum_variance_weights(cov)
        assert weights.sum() == pytest.approx(1.0)

    def test_lower_variance_asset_gets_more_weight(self):
        cov = np.diag([0.01, 0.04])
        weights = _minimum_variance_weights(cov)
        # For a diagonal covariance the weights are inverse-variance.
        np.testing.assert_allclose(weights, [0.8, 0.2])

    def test_long_only_clips_and_renormalizes(self):
        cov = np.array([[0.04, 0.055], [0.055, 0.09]])
        weights = _minimum_variance_weights(cov, long_only=True)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_singular_covariance_returns_none(self):
        cov = np.zeros((3, 3))
        assert _minimum_variance_weights(cov) is None
