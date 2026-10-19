"""Tests for the `distribution` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog


from imst_quant.cli import cmd_distribution, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    returns = np.random.default_rng(3).normal(0.0, 0.01, 200)
    path = tmp_path / "features.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["distribution", "--features", str(path), *extra])
    code = cmd_distribution(args)
    return code, capsys.readouterr().out


def test_distribution_json_with_normality_and_tails(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--normality", "--tails", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["distribution"]["n_observations"] == 200
    assert isinstance(payload["normality_tests"]["is_normal_5pct"], bool)
    assert payload["tail_analysis"]["var_99"] <= payload["tail_analysis"]["var_95"]


def test_distribution_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--normality")
    assert code == 0
    assert "Jarque-Bera" in out


def test_distribution_drops_nan_and_filters_asset_id(tmp_path, capsys):
    path = tmp_path / "features.parquet"
    rng = np.random.default_rng(5)
    pl.DataFrame({
        "asset_id": ["A"] * 50 + ["B"] * 50,
        "returns": np.concatenate([[np.nan, np.inf], rng.normal(0, 0.01, 48), rng.normal(0, 0.01, 50)]),
    }).write_parquet(path)
    args = create_parser().parse_args(["distribution", "--features", str(path), "--asset", "A", "--json"])
    assert cmd_distribution(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distribution"]["n_observations"] == 48
    assert np.isfinite(payload["distribution"]["max"])


def test_distribution_rejects_unknown_asset(tmp_path, capsys):
    path = tmp_path / "features.parquet"
    pl.DataFrame({"asset_id": ["A"] * 5, "returns": [0.01] * 5}).write_parquet(path)
    args = create_parser().parse_args(["distribution", "--features", str(path), "--asset", "Z"])
    assert cmd_distribution(args) == 1
    assert "at least 3" in capsys.readouterr().out
