"""Tests for ultimate_oscillator, fisher_transform, ppo and their CLI commands."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_fisher, cmd_ppo, cmd_uo, create_parser
from imst_quant.utils.technical_indicators import fisher_transform, ppo, ultimate_oscillator


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _ohlc(n=300, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    spread = close * rng.uniform(0.002, 0.01, n)
    return pl.DataFrame({"high": close + spread, "low": close - spread, "close": close})


def test_uo_bounds_and_trend():
    vals = ultimate_oscillator(_ohlc())["uo"].drop_nulls()
    assert vals.min() >= 0 and vals.max() <= 100
    assert ultimate_oscillator(_ohlc(drift=0.01))["uo"][-1] > 60   # closes near highs
    assert ultimate_oscillator(_ohlc(drift=-0.01))["uo"][-1] < 40
    flat = pl.DataFrame({"high": [5.0] * 40, "low": [5.0] * 40, "close": [5.0] * 40})
    assert ultimate_oscillator(flat)["uo"][-1] == pytest.approx(50.0)  # zero range -> neutral
    with pytest.raises(ValueError):
        ultimate_oscillator(_ohlc(), periods=(0, 14, 28))


def test_fisher_extremes_and_trigger():
    out = fisher_transform(_ohlc(drift=0.01))
    assert out["fisher"][-1] > 1.5                       # sustained highs push to extreme
    assert out["fisher_trigger"][-1] == out["fisher"][-2]
    assert fisher_transform(_ohlc(drift=-0.01))["fisher"][-1] < -1.5
    with pytest.raises(ValueError):
        fisher_transform(_ohlc(), period=1)


def test_ppo_matches_macd_ratio_and_is_scale_free():
    df = _ohlc()
    a = ppo(df)["ppo"].to_numpy()
    b = ppo(df.with_columns(pl.col("close") * 1000))["ppo"].to_numpy()
    np.testing.assert_allclose(a, b)
    hist = ppo(df)
    np.testing.assert_allclose(hist["ppo_hist"], hist["ppo"] - hist["ppo_signal"])
    with pytest.raises(ValueError):
        ppo(df, fast=26, slow=12)


def _run(cmd, name, tmp_path, capsys, df, *extra):
    path = tmp_path / "px.parquet"
    df.write_parquet(path)
    code = cmd(create_parser().parse_args([name, "--data", str(path), *extra]))
    return code, capsys.readouterr().out


@pytest.mark.parametrize("cmd,name,title", [
    (cmd_uo, "uo", "Ultimate Oscillator"),
    (cmd_fisher, "fisher", "Fisher Transform"),
    (cmd_ppo, "ppo", "Percentage Price Oscillator"),
])
def test_commands(cmd, name, title, tmp_path, capsys):
    code, out = _run(cmd, name, tmp_path, capsys, _ohlc(), "--json")
    assert code == 0 and json.loads(out)["n_bars"] == 300
    code, out = _run(cmd, name, tmp_path, capsys, _ohlc())
    assert code == 0 and title in out
    assert _run(cmd, name, tmp_path, capsys, _ohlc(5))[0] == 1  # too short


def test_command_validation(tmp_path, capsys):
    assert _run(cmd_uo, "uo", tmp_path, capsys, _ohlc(), "--periods", "14", "7", "28")[0] == 1
    assert _run(cmd_fisher, "fisher", tmp_path, capsys, _ohlc(), "--extreme", "0")[0] == 1
    assert _run(cmd_ppo, "ppo", tmp_path, capsys, _ohlc(), "--fast", "30")[0] == 1
    code, out = _run(cmd_uo, "uo", tmp_path, capsys, _ohlc(drift=0.01), "--overbought", "60", "--json")
    assert json.loads(out)["zone"] == "overbought"
