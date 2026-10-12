"""Tests for the `regime` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_regime, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    rng = np.random.default_rng(1)
    returns = np.r_[rng.normal(0.002, 0.005, 120), rng.normal(-0.003, 0.03, 80)]
    path = tmp_path / "features.parquet"
    pl.DataFrame({"return_1d": returns}).write_parquet(path)
    args = create_parser().parse_args(["regime", "--features", str(path), *extra])
    code = cmd_regime(args)
    return code, capsys.readouterr().out


@pytest.mark.parametrize("kind", ["volatility", "trend", "combined"])
def test_regime_json(tmp_path, capsys, kind):
    code, out = _run(tmp_path, capsys, "--type", kind, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["current_regime"] in payload["statistics"]
    assert sum(s["count"] for s in payload["statistics"].values()) <= 200
    for probs in payload["transition_matrix"].values():
        assert np.isclose(sum(probs.values()), 1.0)


def test_regime_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys)
    assert code == 0
    assert "Current Regime:" in out
    assert "Transition Probabilities:" in out
