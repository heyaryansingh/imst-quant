"""Tests for the `consistency` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_consistency, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, frame, *extra):
    path = tmp_path / "signals.parquet"
    frame.write_parquet(path)
    args = create_parser().parse_args(["consistency", "--data", str(path), *extra])
    code = cmd_consistency(args)
    return code, capsys.readouterr().out


def _frame(n=250):
    rng = np.random.default_rng(0)
    signal = rng.choice([-1.0, 1.0], n)
    returns = signal * 0.01 + rng.normal(0, 0.002, n)
    returns[200:] = rng.normal(0, 0.01, max(n - 200, 0))
    return pl.DataFrame({"signal": signal, "returns": returns})


def test_consistency_json_reports_late_degradation(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _frame(), "--returns-col", "returns", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_observations"] == 250
    assert 0.0 <= payload["consistency_score"] <= 1.0
    assert 200 in [e["period_start"] for e in payload["degradation_events"]]


def test_consistency_persistent_signal_scores_higher(tmp_path, capsys):
    sticky = pl.DataFrame({"signal": [1.0] * 20 + [-1.0] * 20})
    choppy = pl.DataFrame({"signal": [1.0, -1.0] * 20})
    _, a = _run(tmp_path, capsys, sticky, "--json")
    _, b = _run(tmp_path, capsys, choppy, "--json")
    assert json.loads(a)["consistency_score"] > json.loads(b)["consistency_score"]


def test_consistency_text_skips_degradation_on_short_data(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _frame(100))
    assert code == 0
    assert "skipped (need 200 rows" in out


@pytest.mark.parametrize("extra", [["--window", "1"], ["--returns-col", "nope"]])
def test_consistency_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, _frame(), *extra)
    assert code == 1
    assert "Error" in out
