"""Tests for the `circuitbreaker` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_circuitbreaker, create_parser


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["circuitbreaker", "--returns", str(path), *extra])
    code = cmd_circuitbreaker(args)
    return code, capsys.readouterr().out


def test_breaker_halts_and_caps_crash(tmp_path, capsys):
    # Steady losses of 2%/bar: raw strategy loses ~64% over 50 bars.
    r = [-0.02] * 50
    code, out = _run(tmp_path, capsys, r, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["halts"] == 1
    assert payload["current_action"] == "halt"
    assert payload["current_position_scale"] == 0.0
    assert payload["raw"]["max_drawdown"] == pytest.approx(1 - 0.98 ** 50)
    # Managed side stops losing once halted near the 20% level.
    assert payload["managed"]["max_drawdown"] < 0.25
    assert sum(payload["time_in_action"].values()) == pytest.approx(1.0)


def test_breaker_is_no_op_without_drawdown(tmp_path, capsys):
    r = list(np.random.default_rng(0).uniform(0.0, 0.01, 100))
    r[3] = float("nan")
    code, out = _run(tmp_path, capsys, r, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["observations"] == 99
    assert payload["managed"] == payload["raw"]
    assert payload["average_exposure"] == 1.0


def test_breaker_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [-0.02] * 50)
    assert code == 0
    assert "Now: HALT" in out


@pytest.mark.parametrize("extra", [
    ["--thresholds", "0.1,0.05,0.15,0.2"],
    ["--thresholds", "0.05,0.1,0.2"],
    ["--thresholds", "a,b,c,d"],
    ["--cooldown", "-1"],
])
def test_breaker_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, [0.01] * 10, *extra)
    assert code == 1
    assert "Error" in out
