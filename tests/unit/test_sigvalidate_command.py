"""Tests for the `sigvalidate` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_sigvalidate, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, signal, returns, *extra):
    path = tmp_path / "sig.parquet"
    pl.DataFrame({"signal": signal, "returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["sigvalidate", "--data", str(path), *extra])
    code = cmd_sigvalidate(args)
    return code, capsys.readouterr().out


def test_perfect_signal_passes(tmp_path, capsys):
    rng = np.random.default_rng(0)
    r = np.abs(rng.normal(0.01, 0.005, 300))
    # Signal at t-1 earns r at t; stay long the whole time (low turnover)
    code, out = _run(tmp_path, capsys, [1.0] * 300, r.tolist(), "--json")
    payload = json.loads(out)
    assert code == 0 and payload["passed"] is True
    assert payload["hit_rate"] == 1.0
    assert payload["consistency"]["autocorr_lag1"] is None  # constant signal


def test_losing_signal_fails_with_exit_2(tmp_path, capsys):
    r = [0.01] * 100
    code, out = _run(tmp_path, capsys, [-1.0] * 100, r, "--json")
    assert code == 2 and json.loads(out)["checks"]["hit_rate_check"] is False


def test_rejects_non_ternary_signal(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [0.5, 1.0, -1.0, 0.0], [0.01] * 4)
    assert code == 1 and "-1, 0 or 1" in out


def test_text_report(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [1.0, -1.0, 0.0] * 20, [0.01, -0.01, 0.0] * 20)
    assert "Signal Validation" in out and "Verdict" in out
