"""Tests for the `pairs` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_pairs, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_pairs_scan_json_is_parseable(tmp_path, capsys):
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 300)) + 100
    path = tmp_path / "features.parquet"
    pl.DataFrame({"A": x, "B": 2 * x + rng.normal(0, 1, 300)}).write_parquet(path)
    args = create_parser().parse_args(["pairs", "--features", str(path), "--scan", "--json"])
    assert cmd_pairs(args) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)
