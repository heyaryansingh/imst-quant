"""Tests for aroon and the `aroon` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_aroon, create_parser
from imst_quant.utils.technical_indicators import aroon


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


HL = {"high": [1.0, 2, 3, 2.5, 2.4], "low": [0.5, 1.5, 2.5, 1.0, 1.2]}


def test_aroon_by_hand():
    out = aroon(pl.DataFrame(HL), period=2)
    assert out["aroon_up"].to_list() == [None, None, 100, 50, 0]
    assert out["aroon_down"].to_list() == [None, None, 0, 100, 50]
    assert out["aroon_osc"].to_list() == [None, None, 100, -50, -50]


def test_flat_ties_to_most_recent_and_bad_period():
    flat = pl.DataFrame({"high": [5.0] * 4, "low": [4.0] * 4})
    assert aroon(flat, period=3)["aroon_osc"].to_list()[-1] == 0
    assert aroon(flat, period=3)["aroon_up"].to_list()[-1] == 100
    assert aroon(flat, period=10)["aroon_up"].null_count() == 4
    with pytest.raises(ValueError):
        aroon(flat, period=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "hl.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["aroon", "--data", str(path), *extra])
    code = cmd_aroon(args)
    return code, capsys.readouterr().out


def test_cli_json_regime_and_crossovers(tmp_path, capsys):
    rising = pl.DataFrame({"high": [float(i) for i in range(1, 12)], "low": [i - 0.5 for i in range(1, 12)]})
    res = json.loads(_run(tmp_path, capsys, rising, "--period", "5", "--json")[1])
    assert res["regime"] == "strong_up"
    assert (res["aroon_up"], res["aroon_down"]) == (100, 0)
    assert res["bars_since_low"] == 5
    assert res["crossovers"] == 0
    res = json.loads(_run(tmp_path, capsys, pl.DataFrame(HL), "--period", "2", "--json")[1])
    assert res["crossovers"] == 1


def test_cli_errors(tmp_path, capsys):
    df = pl.DataFrame(HL)
    assert _run(tmp_path, capsys, df, "--period", "5")[0] == 1
    assert _run(tmp_path, capsys, df.drop("low"))[0] == 1
    assert _run(tmp_path, capsys, df, "--period", "0")[0] == 1
    code, out = _run(tmp_path, capsys, df, "--period", "2")
    assert code == 0 and "Aroon" in out
