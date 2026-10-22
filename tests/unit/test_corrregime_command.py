"""Tests for the `corrregime` CLI command."""

import datetime as dt
import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_corrregime, create_parser


def _features_file(tmp_path, n_dates=200, break_at=None, seed=0):
    """Three assets driven by one factor; after ``break_at`` C decouples."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    common = rng.normal(0, 0.01, n_dates)
    series = {k: common + rng.normal(0, 0.003, n_dates) for k in ("A", "B", "C")}
    if break_at is not None:
        series["C"][break_at:] = rng.normal(0, 0.01, n_dates - break_at)
    rows = [
        {"date": d, "asset_id": name, "return_1d": float(v)}
        for name, values in series.items()
        for d, v in zip(dates, values)
    ]
    rows.append({"date": dates[-1], "asset_id": "A", "return_1d": float("nan")})
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(capsys, path, *extra):
    args = create_parser().parse_args(
        ["corrregime", "--features", str(path), *extra]
    )
    code = cmd_corrregime(args)
    return code, capsys.readouterr().out


def test_corrregime_is_registered_with_defaults():
    args = create_parser().parse_args(["corrregime"])
    assert (args.window, args.baseline, args.crisis) == (60, 252, 0.7)


def test_corrregime_flags_crisis_and_stable_structure(tmp_path, capsys):
    code, out = _run(capsys, _features_file(tmp_path), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["regime"] == "crisis"
    assert payload["assets"] == 3
    assert payload["pc1_variance_share"] > 0.8
    assert payload["stability"]["compared"] is True
    assert payload["stability"]["is_stable"] is True


def test_corrregime_flags_pairs_that_decouple(tmp_path, capsys):
    path = _features_file(tmp_path, break_at=140)
    code, out = _run(capsys, path, "--json")
    assert code == 0
    payload = json.loads(out)
    pairs = {tuple(sorted(p["pair"])) for p in payload["stability"]["unstable_pairs"]}
    assert pairs == {("A", "C"), ("B", "C")}
    assert payload["regime"] != "crisis"


def test_corrregime_text_output(tmp_path, capsys):
    code, out = _run(capsys, _features_file(tmp_path))
    assert code == 0
    assert "Correlation regime: CRISIS" in out


@pytest.mark.parametrize(
    "extra",
    [
        ["--window", "5"],
        ["--baseline", "60"],
        ["--elevated", "0.8"],
        ["--change-threshold", "0"],
    ],
)
def test_corrregime_rejects_bad_arguments(tmp_path, capsys, extra):
    code, out = _run(capsys, _features_file(tmp_path), *extra)
    assert code == 1
    assert out.startswith("Error")


def test_corrregime_missing_file(tmp_path, capsys):
    code, out = _run(capsys, tmp_path / "nope.parquet")
    assert code == 1
    assert "not found" in out
