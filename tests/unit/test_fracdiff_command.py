"""Tests for the `fracdiff` CLI command and ffd_weights validation."""

import datetime as dt
import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_fracdiff, create_parser
from imst_quant.utils.fractional_diff import ffd_weights


def _features_file(tmp_path, n_dates=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    rows = [
        {"date": d, "asset_id": "RW", "return_1d": float(v)}
        for d, v in zip(dates, rng.normal(0.0005, 0.01, n_dates))
    ]
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    path = _features_file(tmp_path)
    args = create_parser().parse_args(
        ["fracdiff", "--features", str(path), "--json", *extra]
    )
    code = cmd_fracdiff(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_fracdiff_command_is_registered():
    args = create_parser().parse_args(["fracdiff"])
    assert args.command == "fracdiff"
    assert args.d is None
    assert args.threshold == 1e-4


def test_fracdiff_search_finds_d_that_is_stationary_and_keeps_memory(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys)
    assert code == 0
    info = payload["assets"]["RW"]

    assert payload["mode"] == "min_d_search"
    # A random-walk level is non-stationary at d=0 but stationary by d=1.
    assert 0 < info["d"] <= 1.0
    assert info["stationary"] is True
    assert info["adf_pvalue"] < 0.05
    assert -1 <= info["correlation_with_original"] <= 1


def test_fracdiff_fixed_d_reports_that_d(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys, "--d", "1.0")
    assert code == 0
    info = payload["assets"]["RW"]
    assert payload["mode"] == "fixed"
    assert info["d"] == 1.0
    assert info["window_width"] == 2
    assert info["stationary"] is True


def test_fracdiff_missing_file_and_bad_args(tmp_path, capsys):
    args = create_parser().parse_args(
        ["fracdiff", "--features", str(tmp_path / "nope.parquet")]
    )
    assert cmd_fracdiff(args) == 1
    assert "not found" in capsys.readouterr().out

    code, out = _run(tmp_path, capsys, "--threshold", "0")
    assert code == 1
    assert "--threshold" in out


def test_ffd_weights_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        ffd_weights(float("nan"))
    with pytest.raises(ValueError):
        ffd_weights(0.5, threshold=float("nan"))
    assert ffd_weights(0.5).size < 10000
