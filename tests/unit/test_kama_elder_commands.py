"""Tests for kaufman_efficiency / elder_ray and the `kama` / `elder` CLI commands."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_elder, cmd_kama, create_parser
from imst_quant.utils.technical_indicators import elder_ray, kaufman_efficiency


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_er_and_kama_by_hand():
    df = pl.DataFrame({"close": [1.0, 2, 1, 2, 4]})
    out = kaufman_efficiency(df, period=2, fast=2, slow=30)
    er = out["er"].to_list()
    assert er[:2] == [None, None]
    assert er[2:] == pytest.approx([0.0, 0.0, 3 / 3])
    fast_sc, slow_sc = 2 / 3, 2 / 31
    k = [None, None, 1.0]
    k.append(k[2] + slow_sc**2 * (2 - k[2]))
    k.append(k[3] + (fast_sc) ** 2 * (4 - k[3]))
    assert out["kama"].to_list()[:2] == [None, None]
    assert out["kama"].to_list()[2:] == pytest.approx(k[2:])


def test_er_flat_is_zero_and_validation():
    flat = pl.DataFrame({"close": [5.0] * 4})
    out = kaufman_efficiency(flat, period=2)
    assert out["er"].to_list() == [None, None, 0.0, 0.0]
    assert out["kama"].to_list() == [None, None, 5.0, 5.0]
    short = kaufman_efficiency(pl.DataFrame({"close": [1.0, 2.0]}), period=3)
    assert short["kama"].to_list() == [None, None]
    with pytest.raises(ValueError):
        kaufman_efficiency(flat, period=0)
    with pytest.raises(ValueError):
        kaufman_efficiency(flat, fast=30, slow=30)


def test_elder_ray_by_hand():
    df = pl.DataFrame({"high": [11.0, 13, 12], "low": [9.0, 10, 10], "close": [10.0, 12, 11]})
    out = elder_ray(df, period=3)
    # EMA span 3 (alpha 0.5, min 3 samples): 10 -> 11 -> 11
    assert out["ema"].to_list() == [None, None, 11.0]
    assert out["bull_power"].to_list() == [None, None, 1.0]
    assert out["bear_power"].to_list() == [None, None, -1.0]
    with pytest.raises(ValueError):
        elder_ray(df, period=0)


def _run(cmd, name, tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    code = cmd(create_parser().parse_args([name, "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def _bars(closes):
    return pl.DataFrame({"high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
                         "close": closes})


def test_kama_cli_trend(tmp_path, capsys):
    code, out = _run(cmd_kama, "kama", tmp_path, capsys, _bars([10.0 + i for i in range(30)]), "--json")
    assert code == 0
    res = json.loads(out)
    assert res["er"] == pytest.approx(1.0)
    assert res["regime"] == "efficient"
    assert res["price_vs_kama"] == "above" and res["kama_direction"] == "up"
    assert res["crosses"] == 0 and res["bars_since_cross"] is None


def test_kama_cli_choppy_and_errors(tmp_path, capsys):
    closes = [10.0 + (i % 2) for i in range(30)]
    code, out = _run(cmd_kama, "kama", tmp_path, capsys, _bars(closes), "--json")
    assert code == 0
    res = json.loads(out)
    assert res["regime"] == "noisy" and res["crosses"] > 0
    code, out = _run(cmd_kama, "kama", tmp_path, capsys, _bars(closes), "--fast", "5", "--slow", "5")
    assert code == 1 and "--fast" in out
    code, out = _run(cmd_kama, "kama", tmp_path, capsys, _bars(closes[:5]))
    assert code == 1 and "clean bars" in out


def _with(df, col, idx_vals):
    expr = pl.col(col)
    for i, v in idx_vals.items():
        expr = pl.when(pl.int_range(pl.len()) == i).then(v).otherwise(expr)
    return df.with_columns(expr.alias(col))


def test_elder_cli_setups(tmp_path, capsys):
    # Uptrend whose last two bars dip below the EMA, the second less deeply.
    up = _with(_bars([10.0 + i for i in range(32)]), "low", {30: 25.0, 31: 30.0})
    code, out = _run(cmd_elder, "elder", tmp_path, capsys, up, "--json")
    assert code == 0
    res = json.loads(out)
    assert res["ema_trend"] == "rising"
    assert res["bear_power"] < 0 and res["bear_power_change"] > 0
    assert res["setup"] == "buy"
    # Mirror: downtrend whose last two bars spike above the EMA, the second less.
    down = _with(_bars([50.0 - i for i in range(32)]), "high", {30: 35.0, 31: 30.0})
    code, out = _run(cmd_elder, "elder", tmp_path, capsys, down)
    assert code == 0 and "Setup: SELL" in out
    code, out = _run(cmd_elder, "elder", tmp_path, capsys, _bars([10.0 + i for i in range(32)]), "--json")
    assert json.loads(out)["setup"] == "none"
