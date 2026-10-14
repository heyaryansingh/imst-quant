"""Tests for the `volatility` CLI command."""

import json

import numpy as np
import polars as pl

from imst_quant.cli import cmd_volatility, create_parser


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["volatility", "--features", str(path), *extra])
    code = cmd_volatility(args)
    return code, capsys.readouterr().out


def _two_assets():
    r = np.random.default_rng(0).normal(0, 0.01, 200)
    return pl.DataFrame({"asset": ["A"] * 100 + ["B"] * 100, "returns": np.r_[r[:100], r[100:] * 5]})


def test_multiple_assets_require_asset(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _two_assets(), "--json")
    assert code == 1 and "--asset" in out


def test_forecast_honors_span(tmp_path, capsys):
    df = _two_assets()
    _, out = _run(tmp_path, capsys, df, "--asset", "A", "--forecast", "--span", "60", "--json")
    p = json.loads(out)
    assert p["forecast"]["current"] == p["current_volatility"]


def test_short_history_json_is_valid(tmp_path, capsys):
    df = pl.DataFrame({"returns": np.random.default_rng(1).normal(0, 0.01, 10)})
    code, out = _run(tmp_path, capsys, df, "--method", "compare", "--json")
    assert code == 0
    p = json.loads(out, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert p["volatility_comparison"]["historical_63"] is None


def test_nan_returns_dropped(tmp_path, capsys):
    r = np.random.default_rng(2).normal(0, 0.01, 60)
    clean = _run(tmp_path, capsys, pl.DataFrame({"returns": r}), "--method", "historical", "--json")[1]
    dirty = _run(tmp_path, capsys, pl.DataFrame({"returns": np.r_[r, np.nan]}), "--method", "historical", "--json")[1]
    assert json.loads(clean) == json.loads(dirty)
