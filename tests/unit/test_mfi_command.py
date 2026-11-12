"""Tests for money_flow_index and the `mfi` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_mfi, create_parser
from imst_quant.utils.technical_indicators import money_flow_index


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


MIX = pl.DataFrame({
    "high": [2.0, 3, 2.5, 4], "low": [1.0, 2, 1.5, 3],
    "close": [1.5, 2.5, 2.0, 3.5], "volume": [10.0, 10, 20, 10],
})


def test_mfi_by_hand():
    # tp = 1.5, 2.5, 2.0, 3.5 ; flow = 15, 25, 40, 35
    out = money_flow_index(MIX, period=2)["mfi"].to_list()
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(100 * 25 / 65)
    assert out[3] == pytest.approx(100 * 35 / 75)


def test_flat_reads_50_and_bad_period():
    flat = pl.DataFrame({"high": [2.0] * 4, "low": [1.0] * 4, "close": [1.5] * 4, "volume": [5.0] * 4})
    assert money_flow_index(flat, period=2)["mfi"].to_list() == [None, None, 50.0, 50.0]
    with pytest.raises(ValueError):
        money_flow_index(flat, period=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlcv.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["mfi", "--data", str(path), *extra])
    code = cmd_mfi(args)
    return code, capsys.readouterr().out


def test_cli_json_overbought_streak(tmp_path, capsys):
    mids = [float(i) for i in range(10, 22)]
    up = pl.DataFrame({"high": [m + 0.5 for m in mids], "low": [m - 0.5 for m in mids],
                       "close": mids, "volume": [100.0] * 12})
    res = json.loads(_run(tmp_path, capsys, up, "--period", "3", "--json")[1])
    assert res["mfi"] == 100.0 and res["zone"] == "overbought"
    assert res["bars_in_zone"] == 9 and res["pct_overbought"] == 1.0
    assert res["mfi_change"] == 0.0


def test_cli_errors(tmp_path, capsys):
    assert _run(tmp_path, capsys, MIX, "--period", "5")[0] == 1
    assert _run(tmp_path, capsys, MIX.drop("volume"))[0] == 1
    assert _run(tmp_path, capsys, MIX, "--period", "0")[0] == 1
    assert _run(tmp_path, capsys, MIX, "--period", "2", "--overbought", "40")[0] == 1
    code, out = _run(tmp_path, capsys, MIX, "--period", "2")
    assert code == 0 and "Money Flow Index" in out and "OVERSOLD" not in out
