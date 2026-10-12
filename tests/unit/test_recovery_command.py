"""Tests for the `recovery` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_recovery, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["recovery", "--data", str(path), *extra])
    code = cmd_recovery(args)
    return code, capsys.readouterr().out


RETURNS = [0.01, -0.10, 0.12, 0.01, 0.0, -0.05, 0.01, 0.01, 0.01, 0.03, -0.20, 0.01]


def test_recovery_json(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, RETURNS, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_observations"] == len(RETURNS)
    assert [r["trough_idx"] for r in payload["recoveries"]] == [1, 5]
    assert payload["patterns"]["Incomplete"] == 1
    assert 0.0 <= payload["efficiency_score"] <= 1.0


def test_recovery_text_flags_underwater(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, RETURNS)
    assert code == 0
    assert "Efficiency score" in out
    assert "Still underwater" in out


@pytest.mark.parametrize("extra", [["--returns-col", "nope"]])
def test_recovery_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, RETURNS, *extra)
    assert code == 1
    assert "Error" in out
