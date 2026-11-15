"""Tests for connors_rsi and the `crsi` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_crsi, create_parser
from imst_quant.utils.technical_indicators import connors_rsi


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def test_streak_and_rank():
    out = connors_rsi(pl.DataFrame({"close": [10.0, 11, 12, 12, 11, 10, 9]}), rank_period=2)
    assert out["crsi_streak"].to_list() == [0, 1, 2, 0, -1, -2, -3]
    assert out["crsi"][2] is None and out["crsi"][3] is not None
    vals = connors_rsi(_prices())["crsi"].drop_nulls()
    assert vals.min() >= 0 and vals.max() <= 100
    with pytest.raises(ValueError):
        connors_rsi(_prices(), rank_period=0)


def test_selloff_reads_oversold():
    closes = list(np.linspace(100, 120, 150)) + list(np.linspace(119, 100, 8))
    out = connors_rsi(pl.DataFrame({"close": closes}))
    assert out["crsi"][-1] < 10  # fresh 8-bar losing streak after a long uptrend


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "px.parquet"
    df.write_parquet(path)
    code = cmd_crsi(create_parser().parse_args(["crsi", "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def test_crsi_command(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _prices(), "--json")
    res = json.loads(out)
    assert code == 0 and 0 <= res["crsi"] <= 100 and res["zone"] in ("overbought", "oversold", "neutral")
    code, out = _run(tmp_path, capsys, _prices())
    assert code == 0 and "Connors RSI" in out
    assert _run(tmp_path, capsys, _prices(50))[0] == 1  # shorter than rank warmup
    assert _run(tmp_path, capsys, _prices(), "--oversold", "95")[0] == 1
