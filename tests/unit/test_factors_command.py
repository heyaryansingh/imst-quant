"""Tests for the `factors` CLI command and stdout encoding safety."""

import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_factors, create_parser


@pytest.fixture
def panel(tmp_path):
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 250)
    frames = [
        pl.DataFrame(
            {
                "date": np.arange(250),
                "asset_id": asset,
                "return_1d": beta * common + rng.normal(0, 0.01, 250),
            }
        )
        for asset, beta in [("AAA", 1.5), ("BBB", 1.0), ("CCC", 0.5)]
    ]
    path = tmp_path / "features.parquet"
    pl.concat(frames).write_parquet(path)
    return path


def _run(capsys, path, *extra):
    args = create_parser().parse_args(["factors", "--features", str(path), *extra])
    code = cmd_factors(args)
    return code, capsys.readouterr().out


def test_exposures_json_is_clean_and_not_self_regression(capsys, panel):
    code, out = _run(capsys, panel, "--asset", "AAA", "--json")
    assert code == 0
    payload = json.loads(out)  # no progress line mixed into stdout
    # Asset is regressed on the cross-sectional market, not on itself
    assert payload["r_squared"] < 0.99
    assert payload["betas"]["MKT"] > 1.0


def test_panel_requires_asset(capsys, panel):
    code, out = _run(capsys, panel)
    assert code == 1
    assert "--asset" in out


def test_reports_survive_cp1252_stdout(panel):
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    proc = subprocess.run(
        [sys.executable, "-m", "imst_quant.cli", "health"],
        capture_output=True,
        env=env,
    )
    assert b"UnicodeEncodeError" not in proc.stderr
