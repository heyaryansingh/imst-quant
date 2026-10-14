"""Tests for the `liquidity` CLI command."""

import json
import math

import polars as pl

from imst_quant.cli import cmd_liquidity, create_parser


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["liquidity", "--features", str(path), "--json", *extra])
    code = cmd_liquidity(args)
    return code, capsys.readouterr().out


def test_single_row_with_order_size(tmp_path, capsys):
    df = pl.DataFrame({"asset_id": ["A"], "close": [10.0], "volume": [100.0], "return_1d": [0.01]})
    code, out = _run(tmp_path, capsys, df, "--order-size", "10")
    assert code == 0
    assert json.loads(out)["market_impact"]["volatility"] == 0.02


def test_nan_values_ignored(tmp_path, capsys):
    nan = float("nan")
    df = pl.DataFrame({
        "asset_id": ["A"] * 5,
        "close": [10.0] * 5,
        "volume": [100.0, nan, 100.0, 100.0, 100.0],
        "return_1d": [0.01, nan, -0.01, 0.02, -0.02],
    })
    code, out = _run(tmp_path, capsys, df, "--order-size", "10")
    assert code == 0
    payload = json.loads(out)
    assert payload["avg_daily_volume"] == 100.0
    assert not math.isnan(payload["market_impact"]["volatility"])


def test_asset_without_id_column_errors(tmp_path, capsys):
    df = pl.DataFrame({"close": [10.0] * 5, "volume": [100.0] * 5})
    code, out = _run(tmp_path, capsys, df, "--asset", "A")
    assert code == 1
    assert "asset" in out.lower()
