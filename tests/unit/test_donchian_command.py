"""Tests for donchian_channels and the `donchian` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_donchian, create_parser
from imst_quant.utils.technical_indicators import donchian_channels


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


OHLC = {
    "high": [10.0, 11, 10.5, 13, 12, 9],
    "low": [9.0, 10, 9.5, 12, 11, 7],
    "close": [9.5, 10.5, 10.0, 12.8, 11.5, 7.5],
}


def test_channel_and_breakouts_by_hand():
    out = donchian_channels(pl.DataFrame(OHLC), window=2)
    assert out["dc_upper"].to_list() == [None, 11, 11, 13, 13, 12]
    assert out["dc_lower"].to_list() == [None, 9, 9.5, 9.5, 11, 7]
    # close 12.8 > prior upper 11 -> up; close 7.5 < prior lower 11 -> down
    assert out["dc_breakout"].to_list() == [None, None, 0, 1, 0, -1]
    with pytest.raises(ValueError):
        donchian_channels(pl.DataFrame(OHLC), window=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["donchian", "--data", str(path), *extra])
    code = cmd_donchian(args)
    return code, capsys.readouterr().out


def test_cli_json_skips_bad_bars(tmp_path, capsys):
    rows = {k: v[:2] + [math.nan if k == "high" else 5.0] + v[2:] for k, v in OHLC.items()}
    code, out = _run(tmp_path, capsys, pl.DataFrame(rows), "--window", "2", "--json")
    assert code == 0
    res = json.loads(out)
    assert res["n_bars"] == 6
    assert res["breakout_now"] == "down"
    assert res["bars_since_breakout"] == 0
    assert (res["breakouts_up"], res["breakouts_down"]) == (1, 1)
    assert res["position_pct"] == pytest.approx(0.1)


def test_cli_text_and_errors(tmp_path, capsys):
    df = pl.DataFrame(OHLC)
    code, out = _run(tmp_path, capsys, df, "--window", "2")
    assert code == 0 and "Donchian" in out
    assert _run(tmp_path, capsys, df, "--window", "6")[0] == 1
    assert _run(tmp_path, capsys, df.drop("close"))[0] == 1
    assert _run(tmp_path, capsys, df, "--window", "0")[0] == 1
