"""Tests for the `risk-metrics` CLI command."""

import json
import math

import numpy as np
import polars as pl

from imst_quant.cli import cmd_risk_metrics, create_parser


def _run(tmp_path, capsys, returns):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["risk-metrics", "--returns", str(path), "--json"])
    code = cmd_risk_metrics(args)
    return code, capsys.readouterr().out


def test_risk_metrics_drops_non_finite_returns(tmp_path, capsys):
    r = np.random.default_rng(0).normal(0.001, 0.01, 300)
    r[[10, 50]] = np.nan
    r[100] = np.inf
    code, out = _run(tmp_path, capsys, r)
    assert code == 0
    metrics = json.loads(out)
    assert all(math.isfinite(v) for v in metrics.values())


def test_risk_metrics_rejects_all_non_finite(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [float("nan"), float("inf"), None])
    assert code == 1
    assert "finite" in out
