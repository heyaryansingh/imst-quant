"""Tests for the `breakeven` CLI command and signal_cost_sensitivity."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_breakeven, create_parser
from imst_quant.utils.transaction_costs import signal_cost_sensitivity


def _run(tmp_path, capsys, frame, *extra):
    path = tmp_path / "signals.parquet"
    frame.write_parquet(path)
    args = create_parser().parse_args(["breakeven", "--data", str(path), *extra])
    code = cmd_breakeven(args)
    out = capsys.readouterr().out
    return code, out


def test_breakeven_cost_zeroes_net_pnl():
    rng = np.random.default_rng(0)
    pos = np.sign(rng.normal(size=500))
    ret = pos * 0.002 + rng.normal(0, 0.01, 500)
    result = signal_cost_sensitivity(pos, ret, [0.0])

    be = result["breakeven_cost_bps"]
    assert be is not None and be > 0
    at_be = signal_cost_sensitivity(pos, ret, [be])["scenarios"][0]
    assert abs(at_be["total_pnl"]) < 1e-9
    assert result["scenarios"][0]["sharpe"] == pytest.approx(result["gross"]["sharpe"])


def test_breakeven_none_without_edge_or_turnover():
    ret = np.array([0.01, -0.02, 0.01])
    assert signal_cost_sensitivity(np.ones(3), -np.abs(ret), [5.0])["breakeven_cost_bps"] is None
    assert signal_cost_sensitivity(np.zeros(3), ret, [5.0])["breakeven_cost_bps"] is None


def test_command_sorts_by_date_and_skips_nan(tmp_path, capsys):
    dates = [dt.date(2024, 1, d) for d in (4, 1, 3, 2, 5)]
    frame = pl.DataFrame({
        "date": dates,
        "signal": [-1.0, 1.0, -1.0, 1.0, float("nan")],
        "returns": [-0.02, 0.01, -0.01, 0.02, 0.05],
    })
    code, out = _run(tmp_path, capsys, frame, "--json", "--costs", "10,0")
    assert code == 0
    payload = json.loads(out)
    # Date order: +1,+1,-1,-1 -> turnover 1+0+2+0 = 3, gross pnl 0.06
    assert payload["n_periods"] == 4
    assert payload["total_turnover"] == pytest.approx(3.0)
    assert payload["breakeven_cost_bps"] == pytest.approx(200.0)
    assert [s["cost_bps"] for s in payload["scenarios"]] == [0.0, 10.0]


def test_command_text_and_bad_costs(tmp_path, capsys):
    frame = pl.DataFrame({"signal": [1.0, -1.0, 1.0], "returns": [0.01, -0.01, 0.01]})
    code, out = _run(tmp_path, capsys, frame)
    assert code == 0
    assert "Breakeven Cost" in out

    code, out = _run(tmp_path, capsys, frame, "--costs", "5,abc")
    assert code == 1
    assert "--costs" in out

    code, out = _run(tmp_path, capsys, frame, "--costs", "-1")
    assert code == 1
