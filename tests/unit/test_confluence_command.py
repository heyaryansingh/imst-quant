"""Tests for indicator_confluence and the `confluence` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_confluence, create_parser
from imst_quant.utils.technical_indicators import indicator_confluence


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _bars(n=60, step=1.0, volume=True):
    mids = [50.0 + step * i for i in range(n)]
    data = {"h": [m + 0.5 for m in mids], "l": [m - 0.5 for m in mids],
            "c": [m + (0.4 if step > 0 else -0.4) for m in mids]}
    if volume:
        data["v"] = [100.0] * n
    return pl.DataFrame(data)


def _std(df):
    return df.rename({"h": "high", "l": "low", "c": "close", **({"v": "volume"} if "v" in df.columns else {})})


def test_uptrend_trend_votes_bull_and_overbought_votes_bear():
    res = indicator_confluence(_std(_bars()))
    v = res["votes"]
    assert [v[k] for k in ("macd", "aroon", "vortex", "supertrend", "cmf")] == [1, 1, 1, 1, 1]
    assert v["rsi"] == -1 and v["mfi"] == -1  # contrarian: a steady climb is overbought
    assert res["n_votes"] == 7 and res["score"] == pytest.approx(3 / 7)
    assert not res["choppy"]


def test_downtrend_mirrors_and_no_volume_skips_flow_votes():
    res = indicator_confluence(_std(_bars(step=-0.5, volume=False)))
    assert "mfi" not in res["votes"] and res["n_votes"] == 5
    assert res["score"] == pytest.approx((-4 + 1) / 5)


def test_choppy_market_halves_trend_votes():
    zig = pl.DataFrame({"high": [11.0, 12] * 30, "low": [9.0, 10] * 30, "close": [10.0, 11] * 30})
    res = indicator_confluence(zig)
    assert res["choppy"] and res["chop"] > 61.8
    cast = {k: v for k, v in res["votes"].items() if v is not None}
    w = {k: (1.0 if k == "rsi" else 0.5) for k in cast}
    assert res["score"] == pytest.approx(sum(w[k] * v for k, v in cast.items()) / sum(w.values()))


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlcv.parquet"
    df.write_parquet(path)
    argv = ["confluence", "--data", str(path), "--high-col", "h", "--low-col", "l",
            "--close-col", "c", "--volume-col", "v", *extra]
    return cmd_confluence(create_parser().parse_args(argv)), capsys.readouterr().out


def test_cli_json_and_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _bars(), "--json")
    res = json.loads(out)
    assert code == 0 and res["bias"] == "bullish" and res["n_bars"] == 60
    code, out = _run(tmp_path, capsys, _bars(volume=False), "--threshold", "0.9")
    assert code == 0 and "NEUTRAL" in out and "MFI/CMF skipped" in out


def test_cli_errors(tmp_path, capsys):
    assert _run(tmp_path, capsys, _bars(n=20))[0] == 1  # too few bars for MACD
    assert _run(tmp_path, capsys, _bars(), "--threshold", "0")[0] == 1
    assert _run(tmp_path, capsys, _bars().drop("c"))[0] == 1
    code = cmd_confluence(create_parser().parse_args(["confluence", "--data", str(tmp_path / "nope.parquet")]))
    assert code == 1
