"""Tests for chandelier_exit and the `chandelier` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_chandelier, create_parser
from imst_quant.trading.trailing_stop import chandelier_exit


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_levels_match_hand_computation():
    df = pl.DataFrame({"high": [11.0, 12, 13, 12], "low": [9.0, 10, 11, 8], "close": [10.0, 11, 12, 9]})
    out = chandelier_exit(df, period=2, multiplier=1.0)
    # Last bar: TR=4, Wilder ATR 2 -> 3; HH(2)=13, LL(2)=8
    assert out["long_stop"].to_list() == [None, 10.0, 11.0, 10.0]
    assert out["short_stop"].to_list() == [None, 11.0, 12.0, 11.0]


def test_rejects_bad_params():
    df = pl.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]})
    with pytest.raises(ValueError):
        chandelier_exit(df, period=0)
    with pytest.raises(ValueError):
        chandelier_exit(df, multiplier=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["chandelier", "--data", str(path), *extra])
    code = cmd_chandelier(args)
    return code, capsys.readouterr().out


def test_cli_json_flags_long_exit_and_skips_bad_bars(tmp_path, capsys):
    df = pl.DataFrame({
        "high": [11.0, 12, math.nan, 13, 12],
        "low": [9.0, 10, 10, 11, 8],
        "close": [10.0, 11, 11, 12, 9],
    })
    code, out = _run(tmp_path, capsys, df, "--period", "2", "--mult", "1", "--json")
    assert code == 0
    res = json.loads(out)
    assert res["n_bars"] == 4  # NaN bar dropped
    assert res["long_stop"] == pytest.approx(10.0)
    assert res["long_exit"] is True  # close 9 <= stop 10
    assert res["short_exit"] is False


def test_cli_errors(tmp_path, capsys):
    df = pl.DataFrame({"high": [11.0], "low": [9.0], "close": [10.0]})
    assert _run(tmp_path, capsys, df, "--period", "5")[0] == 1
    assert _run(tmp_path, capsys, df.drop("low"))[0] == 1
