"""Tests for the `correlation` CLI command."""

import json
import sys
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_correlation, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _features(n=120):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, n)
    series = {
        "AAA": base,
        "BBB": base + rng.normal(0, 0.002, n),  # strongly correlated with AAA
        "CCC": rng.normal(0, 0.01, n),
    }
    rows = []
    for asset, values in series.items():
        values = values.copy()
        values[[3, n // 3]] = np.nan
        for i, v in enumerate(values):
            rows.append((date(2024, 1, 1) + timedelta(days=i), asset, float(v)))
    return pl.DataFrame(rows, schema=["date", "asset_id", "return_1d"], orient="row")


def _run(tmp_path, capsys, frame, *extra):
    path = tmp_path / "features.parquet"
    frame.write_parquet(path)
    args = create_parser().parse_args(["correlation", "--features", str(path), *extra])
    code = cmd_correlation(args)
    return code, capsys.readouterr().out


def test_summary_ignores_nan_returns(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _features(), "--json")
    assert code == 0
    payload = json.loads(out)  # strict JSON: no NaN tokens
    assert payload["total_pairs"] == 3
    pairs = {(p["asset1"], p["asset2"]) for p in payload["highly_correlated_pairs"]}
    assert ("AAA", "BBB") in pairs


def test_summary_honors_spearman_method(tmp_path, capsys):
    frame = _features()
    _, pearson = _run(tmp_path, capsys, frame, "--json")
    _, spearman = _run(tmp_path, capsys, frame, "--json", "--method", "spearman")
    assert json.loads(pearson)["avg_correlation"] != json.loads(spearman)["avg_correlation"]


def test_summary_with_too_few_observations_does_not_crash(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _features(n=10), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["total_pairs"] == 0
    assert payload["avg_correlation"] is None


def test_rolling_pair_ignores_nan_returns(tmp_path, capsys):
    code, out = _run(
        tmp_path, capsys, _features(), "--asset1", "AAA", "--asset2", "BBB", "--json"
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["avg_correlation"] > 0.8
