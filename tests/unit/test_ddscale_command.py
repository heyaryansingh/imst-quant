"""Tests for the `ddscale` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_ddscale, create_parser


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["ddscale", "--returns", str(path), *extra])
    code = cmd_ddscale(args)
    return code, capsys.readouterr().out


def test_scaling_cuts_drawdown_and_kill_switch_flattens(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [-0.02] * 50, "--json")
    assert code == 0
    p = json.loads(out)
    assert set(p["scaled"]) == {"linear", "convex", "concave", "step"}
    assert p["raw"]["max_drawdown"] == pytest.approx(1 - 0.98 ** 50)
    for m in p["scaled"].values():
        assert m["max_drawdown"] < p["raw"]["max_drawdown"]
        assert m["current_scale"] == 0.0
        assert m["time_halted"] > 0
        assert m["average_exposure"] < 1.0


def test_no_drawdown_is_no_op(tmp_path, capsys):
    r = list(np.random.default_rng(0).uniform(0.0, 0.01, 100))
    r[5] = float("nan")
    code, out = _run(tmp_path, capsys, r, "--method", "linear", "--json")
    assert code == 0
    p = json.loads(out)
    assert p["observations"] == 99
    lin = p["scaled"]["linear"]
    assert lin["average_exposure"] == 1.0
    assert lin["total_return"] == pytest.approx(p["raw"]["total_return"])


def test_text_output(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [0.01, -0.03, 0.02, -0.05, 0.01] * 10)
    assert code == 0
    assert "Best Sharpe:" in out and "Concave" in out


@pytest.mark.parametrize("extra", [
    ["--max-drawdown", "0"], ["--floor", "1.5"], ["--kill-switch", "-0.1"],
])
def test_rejects_bad_input(tmp_path, capsys, extra):
    code, _ = _run(tmp_path, capsys, [0.01] * 10, *extra)
    assert code == 1
