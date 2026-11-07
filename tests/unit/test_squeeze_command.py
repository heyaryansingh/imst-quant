"""Tests for keltner_channels, bollinger_squeeze and the `squeeze` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_squeeze, create_parser
from imst_quant.utils.technical_indicators import bollinger_squeeze, keltner_channels


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _frame():
    # Swinging, then compressed, then a breakout up.
    c = [100 + 10 * math.sin(i / 2) for i in range(20)]
    c += [110 + 0.1 * (i % 2) for i in range(15)]
    c += [110 + 3 * i for i in range(6)]
    return pl.DataFrame({"close": c, "high": [x + 1 for x in c], "low": [x - 1 for x in c]})


def test_keltner_constant_range():
    df = pl.DataFrame({"high": [11.0] * 4, "low": [9.0] * 4, "close": [10.0] * 4})
    out = keltner_channels(df, window=2, multiplier=1.0)
    assert out["kc_middle"].to_list() == [None, 10.0, 10.0, 10.0]
    assert out["kc_upper"].to_list() == [None, 12.0, 12.0, 12.0]
    assert out["kc_lower"].to_list() == [None, 8.0, 8.0, 8.0]


def test_squeeze_detects_compression_then_fires():
    out = bollinger_squeeze(_frame(), window=10)
    on = out["squeeze_on"].to_list()
    assert on[:9] == [None] * 9
    assert not any(on[9:25])  # swinging market: no squeeze
    assert all(on[33:37])  # compressed stretch
    fired = [i for i, f in enumerate(out["squeeze_fired"].to_list()) if f]
    assert fired == [37]


def test_rejects_bad_params():
    df = _frame()
    with pytest.raises(ValueError):
        bollinger_squeeze(df, window=1)
    with pytest.raises(ValueError):
        keltner_channels(df, multiplier=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["squeeze", "--data", str(path), *extra])
    code = cmd_squeeze(args)
    return code, capsys.readouterr().out


def test_cli_json_reports_fire_and_skips_bad_bars(tmp_path, capsys):
    df = _frame().head(38)
    bad = pl.DataFrame({"close": [math.nan], "high": [1.0], "low": [1.0]})
    df = pl.concat([df.head(5), bad, df.tail(33)])
    code, out = _run(tmp_path, capsys, df, "--window", "10", "--json")
    assert code == 0
    res = json.loads(out)
    assert res["n_bars"] == 38
    assert res["squeeze_on"] is False
    assert res["squeeze_fired"] is True
    assert res["bias"] == "up"
    assert res["squeeze_episodes"] == 1


def test_cli_errors(tmp_path, capsys):
    df = _frame()
    assert _run(tmp_path, capsys, df.head(3), "--window", "10")[0] == 1
    assert _run(tmp_path, capsys, df.drop("high"))[0] == 1
    assert _run(tmp_path, capsys, df, "--kc-mult", "0")[0] == 1
