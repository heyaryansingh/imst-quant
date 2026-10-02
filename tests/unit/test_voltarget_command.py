"""Tests for the `voltarget` CLI command and vol-targeting helpers."""

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_voltarget, create_parser
from imst_quant.utils.volatility_targeting import (
    VolatilityTargeter,
    VolTargetConfig,
    vol_targeted_returns,
)


def _features_file(tmp_path, n_dates=250, short_history=None, seed=0):
    """Write a features parquet with a low-vol, high-vol and regime-shift asset."""
    rng = np.random.default_rng(seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    half = n_dates // 2
    series = {
        "CALM": rng.normal(0, 0.003, n_dates),
        "WILD": rng.normal(0, 0.03, n_dates),
        "SHIFT": np.concatenate(
            [rng.normal(0, 0.005, half), rng.normal(0, 0.04, n_dates - half)]
        ),
    }
    rows = [
        {"date": d, "asset_id": name, "return_1d": float(v)}
        for name, values in series.items()
        for d, v in zip(dates, values)
    ]
    if short_history:
        rows += [
            {"date": d, "asset_id": "TINY", "return_1d": 0.001}
            for d in dates[:short_history]
        ]
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    # Mirror main(), which routes structlog to stderr so --json stays parseable.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra, **file_kwargs):
    path = _features_file(tmp_path, **file_kwargs)
    args = create_parser().parse_args(
        ["voltarget", "--features", str(path), "--json", *extra]
    )
    code = cmd_voltarget(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_voltarget_command_is_registered():
    args = create_parser().parse_args(["voltarget"])
    assert args.command == "voltarget"
    assert args.target_vol == 0.15
    assert args.lookback == 20
    assert args.max_leverage == 2.0


def test_voltarget_scales_exposure_inversely_to_vol(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys)
    assert code == 0

    calm, wild = payload["assets"]["CALM"], payload["assets"]["WILD"]
    # CALM (~5% vol) needs leverage, capped at 2x; WILD (~48%) is cut well below 1x.
    assert calm["target_exposure"] == 2.0
    assert wild["target_exposure"] < 0.5
    assert wild["rebalance_needed"] is True
    weights = payload["inverse_vol_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["CALM"] > weights["WILD"]


def test_voltarget_backtest_pulls_vol_toward_target(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys, "--max-leverage", "10")
    assert code == 0

    for name in ("WILD", "SHIFT"):
        bt = payload["assets"][name]["backtest"]
        assert abs(bt["scaled_vol"] - 0.15) < abs(bt["unscaled_vol"] - 0.15)
    # Stationary vol is hit closely; SHIFT overshoots while the window catches up.
    assert abs(payload["assets"]["WILD"]["backtest"]["scaled_vol"] - 0.15) < 0.03


def test_voltarget_skips_short_history_and_rejects_bad_input(tmp_path, capsys):
    code, payload = _run(tmp_path, capsys, short_history=10)
    assert code == 0
    assert payload["skipped_assets"] == ["TINY"]
    assert "TINY" not in payload["assets"]

    code, out = _run(tmp_path, capsys, "--target-vol", "0")
    assert code == 1
    assert "--target-vol" in out


def test_vol_targeted_returns_uses_only_prior_window():
    returns = pd.Series([0.01, -0.01] * 15 + [0.5])
    config = VolTargetConfig(lookback_days=10)
    result = vol_targeted_returns(returns, config)

    assert result["exposure"].iloc[:10].isna().all()
    # The 0.5 shock on the last bar must not shrink the exposure applied to it.
    assert result["exposure"].iloc[-1] == result["exposure"].iloc[-2]


def test_realized_vol_ignores_nan_gaps_and_schedule_handles_short_history():
    targeter = VolatilityTargeter(VolTargetConfig(lookback_days=5))
    returns = pd.Series([np.nan, np.nan, np.nan, 0.01, np.nan])
    assert targeter.calculate_realized_vol(returns) == targeter.config.vol_floor

    schedule = targeter.generate_rebalance_schedule(
        pd.Series([0.01, -0.02, 0.015]), current_exposure=1.0, forecast_days=3
    )
    assert schedule["forecast_vol"].notna().all()
    assert schedule["target_exposure"].notna().all()
