"""Tests for the `leadlag` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_leadlag, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, n=400, *extra):
    rng = np.random.default_rng(0)
    sentiment = rng.normal(size=n)
    returns = np.r_[0.0, 0.5 * sentiment[:-1]] + rng.normal(size=n)
    path = tmp_path / "s.parquet"
    pl.DataFrame({"sentiment": sentiment, "returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["leadlag", "--data", str(path), *extra])
    code = cmd_leadlag(args)
    return code, capsys.readouterr().out


def test_leadlag_json_detects_sentiment_lead(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, 400, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["sentiment_causes_returns"]
    assert not payload["returns_cause_sentiment"]
    assert payload["optimal_lag"] == 1
    assert len(payload["sentiment_to_returns"]) == 5


def test_leadlag_text_output(tmp_path, capsys):
    code, out = _run(tmp_path, capsys)
    assert code == 0
    assert "Sentiment leads returns" in out


@pytest.mark.parametrize(
    "extra", [["--max-lag", "0"], ["--significance", "1.5"], ["--returns-col", "x"]]
)
def test_leadlag_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, 400, *extra)
    assert code == 1
    assert "Error" in out


def test_leadlag_needs_enough_rows(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, 20)
    assert code == 1
    assert "at least 25" in out
