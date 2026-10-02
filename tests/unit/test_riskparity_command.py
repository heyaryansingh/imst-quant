"""Tests for the `riskparity` CLI command."""

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_riskparity, create_parser
from imst_quant.utils.risk_parity import RiskParityOptimizer


def _features_file(tmp_path, n_dates=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    vols = {"CALM": 0.004, "MID": 0.012, "WILD": 0.04}
    rows = [
        {"date": d, "asset_id": name, "return_1d": float(v)}
        for name, vol in vols.items()
        for d, v in zip(dates, rng.normal(0, vol, n_dates))
    ]
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    path = _features_file(tmp_path)
    args = create_parser().parse_args(
        ["riskparity", "--features", str(path), "--json", *extra]
    )
    code = cmd_riskparity(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_riskparity_command_is_registered():
    args = create_parser().parse_args(["riskparity"])
    assert args.command == "riskparity"
    assert args.method == "equal_risk_contribution"
    assert args.lookback == 252


def test_riskparity_equalizes_risk_and_underweights_volatile_asset(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys)
    assert code == 0

    assets = payload["assets"]
    assert abs(sum(a["weight"] for a in assets.values()) - 1.0) < 1e-6
    assert abs(sum(a["risk_share"] for a in assets.values()) - 1.0) < 1e-6
    assert assets["CALM"]["weight"] > assets["MID"]["weight"] > assets["WILD"]["weight"]
    shares = [a["risk_share"] for a in assets.values()]
    assert max(shares) - min(shares) < 0.02
    assert payload["observations"] == 252
    assert payload["portfolio_vol_annualized"] > 0


@pytest.mark.parametrize("method", ["hierarchical", "adaptive"])
def test_riskparity_other_methods_produce_valid_weights(tmp_path, capsys, method):
    code, payload = _run(tmp_path, capsys, "--method", method)
    assert code == 0
    weights = [a["weight"] for a in payload["assets"].values()]
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


def test_riskparity_missing_file_and_bad_lookback(tmp_path, capsys):
    args = create_parser().parse_args(
        ["riskparity", "--features", str(tmp_path / "nope.parquet")]
    )
    assert cmd_riskparity(args) == 1
    assert "not found" in capsys.readouterr().out

    code, out = _run(tmp_path, capsys, "--lookback", "5")
    assert code == 1
    assert "--lookback" in out


def test_adaptive_uses_available_history_when_shorter_than_lookback():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 30), "B": rng.normal(0, 0.03, 30)}
    )
    weights = RiskParityOptimizer(returns, method="adaptive").optimize(
        lookback_period=60
    )
    assert not weights.isna().any()
    assert weights.sum() == pytest.approx(1.0)
    assert weights["A"] > weights["B"]
