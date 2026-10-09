"""Tests for the `decay` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_decay, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, n=300, *extra):
    rng = np.random.default_rng(0)
    signal = rng.choice([-1.0, 1.0], n)
    returns = np.roll(signal, 1) * 0.01 + rng.normal(0, 0.005, n)
    path = tmp_path / "signals.parquet"
    pl.DataFrame({"signal": signal, "returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["decay", "--data", str(path), *extra])
    code = cmd_decay(args)
    return code, capsys.readouterr().out


def test_decay_json_shows_one_bar_edge(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, 300, "--horizons", "1,5", "--json")
    assert code == 0
    payload = json.loads(out)
    curve = payload["decay_curve"]
    assert curve["horizons"] == [1, 5]
    assert curve["ic_values"][0] > 0.5
    assert curve["is_significant_at"][0]
    assert payload["staleness"]["recent_hit_rate"] > 0.8


def test_decay_text_output(tmp_path, capsys):
    code, out = _run(tmp_path, capsys)
    assert code == 0
    assert "Half-life" in out


@pytest.mark.parametrize(
    "extra", [["--horizons", "a,b"], ["--horizons", "0"], ["--window", "5"], ["--signal-col", "x"]]
)
def test_decay_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, 300, *extra)
    assert code == 1
    assert "Error" in out


def test_decay_needs_enough_rows(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, 50)
    assert code == 1
    assert "at least 70" in out
