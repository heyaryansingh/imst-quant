"""Tests for ichimoku and the `ichimoku` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_ichimoku, create_parser
from imst_quant.utils.technical_indicators import ichimoku


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


HL = pl.DataFrame({"high": [2.0, 4, 6, 8], "low": [1.0, 3, 5, 7]})


def test_ichimoku_by_hand_and_no_lookahead():
    out = ichimoku(HL, tenkan=1, kijun=2, senkou=3, displacement=1)
    assert out["ichi_tenkan"].to_list() == [1.5, 3.5, 5.5, 7.5]
    assert out["ichi_kijun"].to_list() == [None, 2.5, 4.5, 6.5]
    assert out["ichi_lead_b"].to_list() == [None, None, 3.5, 5.5]
    assert out["ichi_lead_a"].to_list() == [None, 3.0, 5.0, 7.0]
    # Cloud in force at bar t is the lead from bar t-1.
    assert out["ichi_senkou_a"].to_list() == [None, None, 3.0, 5.0]
    assert out["ichi_senkou_b"].to_list() == [None, None, None, 3.5]
    # Changing the last bar must not change the cloud at the last bar.
    bumped = HL.with_columns(pl.Series("high", [2.0, 4, 6, 80]))
    assert ichimoku(bumped, tenkan=1, kijun=2, senkou=3, displacement=1)["ichi_senkou_b"][-1] == 3.5


def test_bad_window():
    with pytest.raises(ValueError):
        ichimoku(HL, displacement=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["ichimoku", "--data", str(path), *extra])
    code = cmd_ichimoku(args)
    return code, capsys.readouterr().out


SMALL = ("--tenkan", "2", "--kijun", "3", "--senkou", "4", "--displacement", "2")


def _trend(step):
    mids = [50.0 + step * i for i in range(12)]
    return pl.DataFrame({"high": [m + 0.5 for m in mids], "low": [m - 0.5 for m in mids], "close": mids})


def test_cli_json_bullish_and_bearish(tmp_path, capsys):
    res = json.loads(_run(tmp_path, capsys, _trend(1.0), *SMALL, "--json")[1])
    assert res["price_vs_cloud"] == "above" and res["tk_state"] == "bullish"
    assert res["chikou"] == "above" and res["bias"] == "bullish"
    assert res["cloud_color"] == "green" and res["distance_to_cloud_pct"] > 0
    res = json.loads(_run(tmp_path, capsys, _trend(-1.0), *SMALL, "--json")[1])
    assert res["bias"] == "bearish" and res["distance_to_cloud_pct"] < 0
    assert res["bars_since_tk_cross"] is None


def test_cli_errors(tmp_path, capsys):
    df = _trend(1.0)
    assert _run(tmp_path, capsys, df)[0] == 1  # 12 bars < 52 + 26
    assert _run(tmp_path, capsys, df.drop("close"), *SMALL)[0] == 1
    assert _run(tmp_path, capsys, df, *SMALL[:-1], "0")[0] == 1
    code, out = _run(tmp_path, capsys, df, *SMALL)
    assert code == 0 and "Ichimoku" in out and "BULLISH" in out
