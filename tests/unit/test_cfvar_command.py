"""Tests for the `cfvar` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_cfvar, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["cfvar", "--returns", str(path), *extra])
    code = cmd_cfvar(args)
    return code, capsys.readouterr().out


def test_cfvar_reports_fat_tail_adjustment(tmp_path, capsys):
    r = np.random.default_rng(0).standard_t(3, 500) * 0.01
    code, out = _run(tmp_path, capsys, r, "--confidence", "0.99", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_observations"] == 500
    assert payload["excess_kurtosis"] > 0
    assert not payload["is_normal"]
    # Fat tails at 99% should widen VaR beyond the Gaussian estimate.
    assert payload["cornish_fisher_var"] > payload["parametric_var"]
    assert payload["cornish_fisher_cvar"] >= payload["cornish_fisher_var"]


def test_cfvar_text_output_and_nan_rows(tmp_path, capsys):
    r = np.append(np.random.default_rng(1).normal(0, 0.01, 100), np.nan)
    code, out = _run(tmp_path, capsys, r)
    assert code == 0
    assert "100 obs" in out
    assert "Cornish-Fisher" in out


@pytest.mark.parametrize("extra", [["--confidence", "1.5"], ["--return-col", "nope"]])
def test_cfvar_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, [0.01, -0.02, 0.0, 0.03, -0.01], *extra)
    assert code == 1
    assert "Error" in out


def test_cfvar_needs_four_observations(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [0.01, -0.02, None])
    assert code == 1
    assert "at least 4" in out
