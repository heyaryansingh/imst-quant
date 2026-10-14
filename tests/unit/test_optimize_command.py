"""Tests for the `optimize` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_optimize, create_parser


@pytest.fixture
def long_panel(tmp_path):
    rng = np.random.default_rng(0)
    frames = [
        pl.DataFrame(
            {
                "date": np.arange(200),
                "asset_id": asset,
                "return_1d": rng.normal(mu, 0.01, 200),
                "volume": rng.normal(1e6, 1e4, 200),
            }
        )
        for asset, mu in [("AAA", 0.001), ("BBB", 0.0), ("CCC", -0.0005)]
    ]
    path = tmp_path / "features.parquet"
    pl.concat(frames).write_parquet(path)
    return path


def _run(capsys, path, *extra):
    args = create_parser().parse_args(["optimize", "--features", str(path), "--json", *extra])
    code = cmd_optimize(args)
    return code, capsys.readouterr().out


def test_long_panel_is_pivoted_to_assets(capsys, long_panel):
    code, out = _run(capsys, long_panel, "--method", "hrp")
    assert code == 0
    assert set(json.loads(out)["weights"]) == {"AAA", "BBB", "CCC"}


def test_target_volatility_is_applied(capsys, long_panel):
    code, out = _run(capsys, long_panel, "--target-volatility", "0.12")
    assert code == 0
    assert json.loads(out)["volatility"] <= 0.12 + 1e-6


def test_conflicting_targets_rejected(capsys, long_panel):
    code, out = _run(capsys, long_panel, "--target-return", "0.1", "--target-volatility", "0.1")
    assert code == 1
    assert "Error" in out
