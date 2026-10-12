"""Tests for the `signal-corr` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_signal_corr, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    r = np.random.default_rng(1).normal(size=100)
    noise = np.random.default_rng(2).normal(size=100)
    df = pl.DataFrame({
        "a": r[1:],  # leads returns by one bar
        "b": r[1:] * 2,  # duplicate of a
        "c": noise[1:],
        "returns": r[:-1],
    })
    path = tmp_path / "signals.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["signal-corr", "--data", str(path), *extra])
    code = cmd_signal_corr(args)
    return code, capsys.readouterr().out


def test_signal_corr_json(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--signals", "a,b,c", "--returns-col", "returns", "--json")
    assert code == 0
    payload = json.loads(out)
    assert [(p["a"], p["b"]) for p in payload["redundant_pairs"]] == [("a", "b")]
    assert payload["lead_lag"]["a"]["best_lag"] == 1
    assert 0.0 <= payload["diversification"] <= 1.0


def test_signal_corr_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--signals", "a,c")
    assert code == 0
    assert "No redundant pairs" in out


@pytest.mark.parametrize("extra", [
    ["--signals", "a"],
    ["--signals", "a,a"],
    ["--signals", "a,nope"],
    ["--signals", "a,b", "--threshold", "0"],
])
def test_signal_corr_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, *extra)
    assert code == 1
    assert "Error" in out
