"""Tests for the `changepoint` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_changepoint, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, returns):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(
        ["changepoint", "--returns", str(path), "--return-col", "returns", "--json"]
    )
    code = cmd_changepoint(args)
    return code, capsys.readouterr().out


def test_changepoint_skips_nan_and_reports_original_row_indices(tmp_path, capsys):
    calm = np.random.default_rng(1).normal(0, 0.01, 150)
    wild = np.random.default_rng(2).normal(0.01, 0.04, 150)
    returns = np.r_[[np.nan] * 10, calm, wild].tolist()
    returns[5] = None

    code, out = _run(tmp_path, capsys, returns)
    assert code == 0
    payload = json.loads(out)
    assert payload["n_dropped"] == 10
    assert payload["n_observations"] == 300
    segments = payload["variance"]["segments"]
    assert segments[0]["start"] == 10  # first finite row, not 0
    assert segments[-1]["end"] == 310  # length of the input file
    # Break lands near the calm/wild boundary at file row 160.
    assert any(150 <= b <= 175 for b in payload["variance"]["break_indices"])
