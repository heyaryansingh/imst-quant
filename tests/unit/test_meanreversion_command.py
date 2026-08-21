"""Tests for the `meanreversion` CLI command."""

import json

import numpy as np
import polars as pl

from imst_quant.cli import cmd_meanreversion, create_parser


def _ou_file(tmp_path, n=600, kappa=0.5, seed=0, col="close", name="prices.parquet"):
    """Write an Ornstein-Uhlenbeck price series: mean-reverting by construction."""
    rng = np.random.default_rng(seed)
    prices = np.empty(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + kappa * (100.0 - prices[i - 1]) + rng.normal(0, 1)

    path = tmp_path / name
    pl.DataFrame({col: prices}).write_parquet(path)
    return path


def _random_walk_file(tmp_path, n=600, seed=1, name="rw.parquet"):
    rng = np.random.default_rng(seed)
    path = tmp_path / name
    pl.DataFrame({"close": 100.0 + np.cumsum(rng.normal(0, 1, n))}).write_parquet(path)
    return path


def test_command_is_registered():
    args = create_parser().parse_args(["meanreversion", "--prices", "x.parquet"])

    assert args.command == "meanreversion"
    assert args.price_col == "close"
    assert args.rolling_window is None
    assert args.json is False


def test_reports_mean_reversion_for_ou_series(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(_ou_file(tmp_path))]
    )

    assert cmd_meanreversion(args) == 0

    out = capsys.readouterr().out
    assert "Mean Reversion Analysis" in out
    assert "Mean reverting:" in out
    assert "YES" in out


def test_json_output_is_parseable(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(_ou_file(tmp_path)), "--json"]
    )

    assert cmd_meanreversion(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["observations"] == 600
    assert payload["is_mean_reverting"] is True
    assert payload["hurst_exponent"] < 0.5
    assert payload["variance_ratio"] < 1.0
    assert "rolling_hurst" not in payload


def test_random_walk_is_not_flagged(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(_random_walk_file(tmp_path)), "--json"]
    )

    assert cmd_meanreversion(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["is_mean_reverting"] is False


def test_rolling_window_adds_hurst_summary(tmp_path, capsys):
    args = create_parser().parse_args(
        [
            "meanreversion",
            "--prices",
            str(_ou_file(tmp_path)),
            "--rolling-window",
            "120",
            "--json",
        ]
    )

    assert cmd_meanreversion(args) == 0

    roll = json.loads(capsys.readouterr().out)["rolling_hurst"]
    assert roll["window"] == 120
    assert roll["min"] <= roll["mean"] <= roll["max"]
    assert 0.0 <= roll["pct_mean_reverting"] <= 1.0


def test_custom_price_column(tmp_path, capsys):
    path = _ou_file(tmp_path, col="spread")
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(path), "--price-col", "spread", "--json"]
    )

    assert cmd_meanreversion(args) == 0
    assert json.loads(capsys.readouterr().out)["observations"] == 600


def test_missing_file_is_reported(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(tmp_path / "nope.parquet")]
    )

    assert cmd_meanreversion(args) == 1
    assert "not found" in capsys.readouterr().out


def test_missing_column_is_reported(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(_ou_file(tmp_path)), "--price-col", "vwap"]
    )

    assert cmd_meanreversion(args) == 1
    out = capsys.readouterr().out
    assert "'vwap' not found" in out
    assert "close" in out


def test_too_few_observations_is_reported(tmp_path, capsys):
    path = tmp_path / "short.parquet"
    pl.DataFrame({"close": np.linspace(100, 110, 10)}).write_parquet(path)
    args = create_parser().parse_args(["meanreversion", "--prices", str(path)])

    assert cmd_meanreversion(args) == 1
    assert "at least 30 observations" in capsys.readouterr().out


def test_rolling_window_larger_than_series_is_reported(tmp_path, capsys):
    args = create_parser().parse_args(
        [
            "meanreversion",
            "--prices",
            str(_ou_file(tmp_path, n=100)),
            "--rolling-window",
            "500",
        ]
    )

    assert cmd_meanreversion(args) == 1
    assert "exceeds" in capsys.readouterr().out


def test_rolling_window_below_two_is_reported(tmp_path, capsys):
    args = create_parser().parse_args(
        ["meanreversion", "--prices", str(_ou_file(tmp_path)), "--rolling-window", "1"]
    )

    assert cmd_meanreversion(args) == 1
    assert "at least 2" in capsys.readouterr().out
