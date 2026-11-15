"""Tests for stoch_rsi / trix indicators and the `stochrsi` / `trix` CLI commands."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import _last_cross, cmd_stochrsi, cmd_trix, create_parser
from imst_quant.utils.technical_indicators import stoch_rsi, trix


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _prices(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def test_stoch_rsi_bounded_and_flat_window_neutral():
    out = stoch_rsi(_prices()).drop_nulls("stochrsi_d")
    assert out["stochrsi_k"].min() >= 0 and out["stochrsi_k"].max() <= 100
    flat = stoch_rsi(pl.DataFrame({"close": [10.0] * 40}))
    assert flat["stochrsi_d"].drop_nulls().to_list() == pytest.approx([50.0] * 23)
    with pytest.raises(ValueError):
        stoch_rsi(_prices(), stoch_period=0)


def test_trix_sign_follows_trend():
    up = trix(pl.DataFrame({"close": np.linspace(100, 200, 60)}), period=5)
    down = trix(pl.DataFrame({"close": np.linspace(200, 100, 60)}), period=5)
    assert up["trix"][-1] > 0 > down["trix"][-1]
    assert up["trix"][0] is None
    with pytest.raises(ValueError):
        trix(_prices(), period=0)


def test_last_cross_skips_touches():
    assert _last_cross([1, 0, -1, -2]) == ("down", 1, 1)
    assert _last_cross([-1, 1, 0, -1, 1]) == ("up", 0, 3)
    assert _last_cross([0, 1, 2]) == (None, None, 0)


def _run(tmp_path, capsys, cmd, df, *extra):
    path = tmp_path / "px.parquet"
    df.write_parquet(path)
    code = cmd(create_parser().parse_args([cmd.__name__[4:], "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def test_stochrsi_command_json_and_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, cmd_stochrsi, _prices(), "--json")
    res = json.loads(out)
    assert code == 0 and 0 <= res["k"] <= 100
    assert res["zone"] in ("overbought", "oversold", "neutral")
    assert res["signal"] in ("buy", "sell", "none")
    code, out = _run(tmp_path, capsys, cmd_stochrsi, _prices())
    assert code == 0 and "Stochastic RSI" in out


def test_stochrsi_command_rejects_bad_input(tmp_path, capsys):
    assert _run(tmp_path, capsys, cmd_stochrsi, _prices(), "--oversold", "90")[0] == 1
    assert _run(tmp_path, capsys, cmd_stochrsi, _prices(10))[0] == 1  # too short to warm up


def test_trix_command_uptrend(tmp_path, capsys):
    df = pl.DataFrame({"close": np.linspace(100, 200, 80)})
    code, out = _run(tmp_path, capsys, cmd_trix, df, "--period", "5", "--json")
    res = json.loads(out)
    assert code == 0 and res["momentum"] == "bullish" and res["last_zero_cross"] is None
    assert _run(tmp_path, capsys, cmd_trix, df, "--signal", "0")[0] == 1


def test_top_level_help_renders():
    # argparse %-formats subcommand help; a stray "%K" crashed `--help`.
    assert "stochrsi" in create_parser().format_help()
