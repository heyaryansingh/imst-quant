"""Tests for supertrend and the `supertrend` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_supertrend, create_parser
from imst_quant.utils.technical_indicators import supertrend


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


OHLC = {
    "high": [11.0, 12, 13, 12, 9, 8, 12, 14],
    "low": [9.0, 10, 11, 8, 6, 5, 9, 12],
    "close": [10.0, 11, 12, 9, 7, 6, 11, 13],
}


def test_bands_ratchet_and_flip_by_hand():
    out = supertrend(pl.DataFrame(OHLC), period=2, multiplier=1.0)
    assert out["st_direction"].to_list() == [None, 1, 1, -1, -1, -1, 1, 1]
    assert out["supertrend"].to_list() == [None, 9.0, 10.0, 13.0, 10.5, 9.5, 6.0, 9.25]


def test_rejects_bad_params():
    df = pl.DataFrame({"high": [1.0], "low": [1.0], "close": [1.0]})
    with pytest.raises(ValueError):
        supertrend(df, period=0)
    with pytest.raises(ValueError):
        supertrend(df, multiplier=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["supertrend", "--data", str(path), *extra])
    code = cmd_supertrend(args)
    return code, capsys.readouterr().out


def test_cli_json_reports_trend_and_skips_bad_bars(tmp_path, capsys):
    rows = {k: v[:3] + [math.nan if k == "close" else v[2]] + v[3:] for k, v in OHLC.items()}
    code, out = _run(tmp_path, capsys, pl.DataFrame(rows), "--period", "2", "--mult", "1", "--json")
    assert code == 0
    res = json.loads(out)
    assert res["n_bars"] == 8  # NaN bar dropped
    assert res["trend"] == "up"
    assert res["supertrend"] == pytest.approx(9.25)
    assert res["flips"] == 2
    assert res["bars_in_trend"] == 2
    assert res["flipped_this_bar"] is False


def test_cli_errors(tmp_path, capsys):
    df = pl.DataFrame({"high": [11.0], "low": [9.0], "close": [10.0]})
    assert _run(tmp_path, capsys, df, "--period", "5")[0] == 1
    assert _run(tmp_path, capsys, df.drop("low"))[0] == 1
    assert _run(tmp_path, capsys, df, "--mult", "0")[0] == 1
