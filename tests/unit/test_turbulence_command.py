"""Tests for the `turbulence` CLI command's rolling absorption output."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_turbulence, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_rolling_absorption_detects_coupling(tmp_path, capsys):
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.01, (200, 5))
    factor = rng.normal(0, 0.02, (100, 1))
    coupled = factor + rng.normal(0, 0.002, (100, 5))
    path = tmp_path / "wide.parquet"
    m = np.vstack([calm, coupled])
    pl.DataFrame({f"a{i}": m[:, i] for i in range(5)}).write_parquet(path)

    args = create_parser().parse_args(
        ["turbulence", "--returns", str(path), "--ar-window", "60", "--json"]
    )
    assert cmd_turbulence(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rolling_absorption"]["window"] == 60
    assert out["rolling_absorption"]["current_ratio"] > 0.9


def test_no_rolling_block_without_flag(tmp_path, capsys):
    path = tmp_path / "wide.parquet"
    m = np.random.default_rng(1).normal(0, 0.01, (100, 3))
    pl.DataFrame({f"a{i}": m[:, i] for i in range(3)}).write_parquet(path)
    args = create_parser().parse_args(["turbulence", "--returns", str(path), "--json"])
    assert cmd_turbulence(args) == 0
    assert "rolling_absorption" not in json.loads(capsys.readouterr().out)
