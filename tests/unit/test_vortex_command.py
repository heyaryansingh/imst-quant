"""Tests for vortex and the `vortex` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_vortex, create_parser
from imst_quant.utils.technical_indicators import vortex


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


UP = pl.DataFrame({"high": [2.0, 3, 4], "low": [1.0, 2, 3], "close": [1.5, 2.5, 3.5]})


def test_vortex_by_hand():
    out = vortex(UP, period=2)
    # VM+ = |3-1| + |4-2| = 4, VM- = 0, TR = 1.5 + 1.5 = 3
    assert out["vi_plus"].to_list()[:2] == [None, None]
    assert out["vi_plus"][2] == pytest.approx(4 / 3)
    assert out["vi_minus"][2] == 0.0
    assert out["vi_diff"][2] == pytest.approx(4 / 3)


def test_flat_window_is_null_and_bad_period():
    flat = pl.DataFrame({"high": [2.0] * 4, "low": [2.0] * 4, "close": [2.0] * 4})
    assert vortex(flat, period=2)["vi_plus"].null_count() == 4
    with pytest.raises(ValueError):
        vortex(flat, period=0)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["vortex", "--data", str(path), *extra])
    code = cmd_vortex(args)
    return code, capsys.readouterr().out


def _zigzag():
    # 6 bars rising then 6 falling: one VI cross.
    mids = [float(i) for i in range(10, 16)] + [float(i) for i in range(14, 8, -1)]
    return pl.DataFrame({"high": [m + 0.5 for m in mids], "low": [m - 0.5 for m in mids], "close": mids})


def test_cli_json_trend_and_cross(tmp_path, capsys):
    res = json.loads(_run(tmp_path, capsys, _zigzag(), "--period", "3", "--json")[1])
    assert res["trend"] == "down"
    assert res["crossovers"] == 1
    assert res["bars_since_cross"] is not None and not res["cross_now"]
    assert res["vi_minus"] > res["vi_plus"]


def test_cli_errors(tmp_path, capsys):
    assert _run(tmp_path, capsys, UP, "--period", "5")[0] == 1
    assert _run(tmp_path, capsys, UP.drop("close"))[0] == 1
    assert _run(tmp_path, capsys, UP, "--period", "0")[0] == 1
    flat = pl.DataFrame({"high": [2.0] * 4, "low": [2.0] * 4, "close": [2.0] * 4})
    assert _run(tmp_path, capsys, flat, "--period", "2")[0] == 1
    code, out = _run(tmp_path, capsys, UP, "--period", "2")
    assert code == 0 and "Vortex" in out
