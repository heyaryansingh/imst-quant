"""Tests for chaikin_money_flow / choppiness_index and the `cmf` / `chop` CLI commands."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_chop, cmd_cmf, create_parser
from imst_quant.utils.technical_indicators import chaikin_money_flow, choppiness_index


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


BARS = pl.DataFrame({
    "high": [2.0, 3, 2.5, 4], "low": [1.0, 2, 1.5, 3],
    "close": [2.0, 2.25, 1.5, 3.5], "volume": [10.0, 20, 10, 10],
})


def test_cmf_by_hand():
    # mfm = 1, -0.5, -1, 0 ; mfv = 10, -10, -10, 0
    out = chaikin_money_flow(BARS, period=2)["cmf"].to_list()
    assert out[0] is None
    assert out[1:] == pytest.approx([0.0, -20 / 30, -10 / 20])


def test_cmf_zero_range_and_zero_volume():
    df = pl.DataFrame({"high": [1.0, 1], "low": [1.0, 1], "close": [1.0, 1], "volume": [0.0, 0]})
    assert chaikin_money_flow(df, period=1)["cmf"].to_list() == [None, None]
    with pytest.raises(ValueError):
        chaikin_money_flow(df, period=0)


def test_chop_by_hand_and_flat():
    # TR: bar 0 = high - low; later bars max(h - l, |h - prev c|, |l - prev c|)
    out = choppiness_index(BARS, period=3)["chop"].to_list()
    tr = [1.0, 1.0, 1.0, 2.5]
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(100 * math.log10(sum(tr[:3]) / 2.0) / math.log10(3))
    assert out[3] == pytest.approx(100 * math.log10(sum(tr[1:]) / 2.5) / math.log10(3))
    flat = pl.DataFrame({"high": [1.0] * 3, "low": [1.0] * 3, "close": [1.0] * 3})
    assert choppiness_index(flat, period=2)["chop"].to_list() == [None, None, None]
    with pytest.raises(ValueError):
        choppiness_index(flat, period=1)


def _run(cmd, name, tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlcv.parquet"
    df.write_parquet(path)
    code = cmd(create_parser().parse_args([name, "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def _trend(n=20, closes_at_high=True):
    mids = [10.0 + i for i in range(n)]
    return pl.DataFrame({
        "high": [m + 0.5 for m in mids], "low": [m - 0.5 for m in mids],
        "close": [m + (0.5 if closes_at_high else -0.5) for m in mids], "volume": [100.0] * n,
    })


def test_cmf_cli_buying_and_divergence(tmp_path, capsys):
    res = json.loads(_run(cmd_cmf, "cmf", tmp_path, capsys, _trend(), "--period", "5", "--json")[1])
    assert res["cmf"] == pytest.approx(1.0) and res["pressure"] == "buying"
    assert res["bars_in_pressure"] == 16 and res["zero_crosses"] == 0 and res["divergence"] == "none"
    # Price rising while closes slide from the high to the low: bearish divergence + a zero cross.
    df = _trend(closes_at_high=True)
    df = df.with_columns(pl.Series("close", [h - (h - lo) * i / 19 for i, (h, lo) in
                                             enumerate(zip(df["high"], df["low"]))]))
    res = json.loads(_run(cmd_cmf, "cmf", tmp_path, capsys, df, "--period", "3", "--json")[1])
    assert res["pressure"] == "selling" and res["divergence"] == "bearish"
    assert res["zero_crosses"] == 1


def test_chop_cli_trend_vs_chop(tmp_path, capsys):
    res = json.loads(_run(cmd_chop, "chop", tmp_path, capsys, _trend(), "--period", "5", "--json")[1])
    assert res["regime"] == "trending" and res["pct_trending"] == 1.0
    zig = pl.DataFrame({"high": [11.0, 12] * 10, "low": [9.0, 10] * 10, "close": [10.0, 11] * 10})
    res = json.loads(_run(cmd_chop, "chop", tmp_path, capsys, zig, "--period", "10", "--json")[1])
    assert res["regime"] == "choppy" and res["bars_in_regime"] == 11


def test_cli_errors(tmp_path, capsys):
    assert _run(cmd_cmf, "cmf", tmp_path, capsys, BARS, "--period", "5")[0] == 1
    assert _run(cmd_cmf, "cmf", tmp_path, capsys, BARS.drop("volume"))[0] == 1
    assert _run(cmd_cmf, "cmf", tmp_path, capsys, BARS, "--period", "2", "--threshold", "1")[0] == 1
    assert _run(cmd_chop, "chop", tmp_path, capsys, BARS, "--period", "1")[0] == 1
    assert _run(cmd_chop, "chop", tmp_path, capsys, BARS, "--period", "2", "--trending", "70")[0] == 1
    code, out = _run(cmd_chop, "chop", tmp_path, capsys, BARS, "--period", "2")
    assert code == 0 and "Choppiness Index" in out
    code, out = _run(cmd_cmf, "cmf", tmp_path, capsys, BARS, "--period", "2")
    assert code == 0 and "Chaikin Money Flow" in out
