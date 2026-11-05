"""Tests for the `drift` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_drift, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, rows, *extra):
    path = tmp_path / "pf.parquet"
    pl.DataFrame(rows, schema=["symbol", "weight", "target_weight"], orient="row").write_parquet(path)
    args = create_parser().parse_args(["drift", "--portfolio", str(path), *extra])
    code = cmd_drift(args)
    return code, capsys.readouterr().out


def test_drift_triggers_and_orders(tmp_path, capsys):
    rows = [("AAPL", 0.45, 0.33), ("GOOG", 0.25, 0.34), ("MSFT", 0.30, 0.33)]
    code, out = _run(tmp_path, capsys, rows, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["trigger"] is True
    orders = {o["asset"]: o for o in payload["orders"]}
    assert orders["AAPL"]["side"] == "SELL"
    assert math.isclose(orders["AAPL"]["dollar_amount"], 12_000.0)
    assert orders["GOOG"]["side"] == "BUY"
    assert math.isclose(payload["estimated_cost"], (12_000 + 9_000 + 3_000) * 0.001)


def test_exit_position_relative_drift_is_null(tmp_path, capsys):
    rows = [("AAPL", 0.5, 0.5), ("OLD", 0.02, 0.0), ("MSFT", 0.48, 0.5)]
    code, out = _run(tmp_path, capsys, rows, "--json")
    assert code == 0
    drift = {d["asset"]: d for d in json.loads(out)["drift_metrics"]}
    assert drift["OLD"]["relative_drift"] is None


def test_hold_text_output(tmp_path, capsys):
    rows = [("AAPL", 0.51, 0.5), ("MSFT", 0.49, 0.5)]
    code, out = _run(tmp_path, capsys, rows)
    assert code == 0
    assert "hold" in out


def test_rejects_nan_weight(tmp_path, capsys):
    rows = [("AAPL", float("nan"), 0.5), ("MSFT", 0.5, 0.5)]
    code, out = _run(tmp_path, capsys, rows)
    assert code == 1
    assert "non-finite" in out


def test_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(["drift", "--portfolio", str(tmp_path / "nope.parquet")])
    assert cmd_drift(args) == 1
