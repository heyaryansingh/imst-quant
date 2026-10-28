"""Tests for the `compare` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_compare, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, data, *extra):
    path = tmp_path / "strategies.parquet"
    pl.DataFrame(data).write_parquet(path)
    args = create_parser().parse_args(["compare", "--data", str(path), *extra])
    code = cmd_compare(args)
    return code, capsys.readouterr().out


DATA = {
    "steady": [0.01, 0.005, 0.01, 0.004, 0.01, 0.006],
    "choppy": [0.05, -0.06, 0.04, -0.05, 0.06, -0.03],
    "bust": [0.1, -1.0, 0.0, 0.0, 0.0, 0.0],
}


def test_compare_json_ranks_by_sharpe(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, DATA, "--json")
    assert code == 0
    payload = json.loads(out)
    names = [r["strategy"] for r in payload["ranking"]]
    assert names[0] == "steady"
    bust = next(r for r in payload["ranking"] if r["strategy"] == "bust")
    assert bust["cagr"] == -1.0


def test_compare_sort_by_drawdown_and_column_subset(tmp_path, capsys):
    code, out = _run(
        tmp_path, capsys, DATA, "--json", "--columns", "choppy,bust", "--sort-by", "max_drawdown"
    )
    assert code == 0
    assert [r["strategy"] for r in json.loads(out)["ranking"]] == ["choppy", "bust"]


def test_compare_text_and_errors(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, DATA)
    assert code == 0 and "steady" in out and "Calmar" in out

    code, out = _run(tmp_path, capsys, DATA, "--columns", "nope")
    assert code == 1 and "Missing columns" in out

    code, out = _run(tmp_path, capsys, {"a": [0.01, None, None]})
    assert code == 1 and "at least 2" in out
