"""Tests for the `montecarlo` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog


from imst_quant.cli import cmd_montecarlo, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    returns = np.random.default_rng(7).normal(0.0005, 0.01, 120)
    path = tmp_path / "features.parquet"
    pl.DataFrame({"return_1d": returns}).write_parquet(path)
    args = create_parser().parse_args(
        ["montecarlo", "--features", str(path), "--simulations", "400", "--horizon", "10",
         "--seed", "1", *extra]
    )
    code = cmd_montecarlo(args)
    return code, capsys.readouterr().out


@pytest.mark.parametrize("method", ["historical", "parametric", "gbm"])
def test_montecarlo_json(tmp_path, capsys, method):
    code, out = _run(tmp_path, capsys, "--method", method, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["method"] == method
    assert payload["expected_shortfall"] >= payload["var"] > 0
    assert set(payload["percentiles"]) >= {"5", "50", "95"}


def test_montecarlo_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys)
    assert code == 0
    assert "Expected Shortfall" in out
