"""Tests for the `risk-metrics` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_risk_metrics, create_parser


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(
        ["risk-metrics", "--returns", str(path), "--json", *extra]
    )
    code = cmd_risk_metrics(args)
    return code, capsys.readouterr().out


def test_first_period_loss_counts_as_drawdown(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [-0.5, 0.1])
    assert code == 0
    assert json.loads(out)["max_drawdown"] == pytest.approx(0.5)


def test_nan_returns_are_ignored(tmp_path, capsys):
    clean = [0.01, -0.02, 0.03, 0.01]
    _, a = _run(tmp_path, capsys, clean)
    _, b = _run(tmp_path, capsys, [0.01, float("nan"), -0.02, None, 0.03, 0.01])
    assert json.loads(a) == pytest.approx(json.loads(b))
