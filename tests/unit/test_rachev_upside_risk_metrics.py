"""Tests for rachev_ratio, upside_potential_ratio and the extended risk-metrics command."""

import json
import math
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_risk_metrics, create_parser
from imst_quant.utils.risk_metrics import calculate_all_metrics, rachev_ratio, upside_potential_ratio


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_rachev():
    r = pl.Series([-0.04, -0.01, 0.0, 0.01, 0.02])
    assert rachev_ratio(r, alpha=0.2, beta=0.2) == pytest.approx(0.5)
    assert rachev_ratio(pl.Series([0.01, 0.02])) == math.inf
    assert rachev_ratio(pl.Series([-0.01, -0.02])) == 0.0
    sym = pl.Series(np.concatenate([-np.arange(1, 101), np.arange(1, 101)]) / 1000)
    assert rachev_ratio(sym) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rachev_ratio(r, alpha=0)


def test_upside_potential():
    assert upside_potential_ratio(pl.Series([0.02, -0.02, 0.02, -0.02])) == pytest.approx(0.5 ** 0.5)
    assert upside_potential_ratio(pl.Series([0.01, 0.0])) == math.inf
    assert upside_potential_ratio(pl.Series([-0.01, float("nan")])) == 0.0


def test_all_metrics_has_new_keys():
    m = calculate_all_metrics(pl.Series(np.random.default_rng(0).normal(0.001, 0.01, 300)))
    for k in ("omega", "gain_to_pain", "burke", "tail_ratio", "rachev", "upside_potential"):
        assert math.isfinite(m[k])


def test_command_json_is_valid_without_losses(tmp_path, capsys):
    path = tmp_path / "r.parquet"
    pl.DataFrame({"returns": [0.01, 0.02, 0.005, 0.01]}).write_parquet(path)
    assert cmd_risk_metrics(create_parser().parse_args(["risk-metrics", "--returns", str(path), "--json"])) == 0
    out = json.loads(capsys.readouterr().out)  # inf would make this raise
    assert out["omega"] is None and out["rachev"] is None
    assert cmd_risk_metrics(create_parser().parse_args(["risk-metrics", "--returns", str(path)])) == 0
    assert "Rachev Ratio" in capsys.readouterr().out
