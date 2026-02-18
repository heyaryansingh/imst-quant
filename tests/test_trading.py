"""Tests for trading (Phase 8) and backtest (Phase 9)."""

import polars as pl
import pytest

from imst_quant.trading import (
    FixedThresholdPolicy,
    DynamicThresholdPolicy,
    run_backtest,
)


def test_fixed_threshold_policy():
    """TRAD-01: Fixed threshold produces signals."""
    policy = FixedThresholdPolicy(thresholds=[0.5, 0.6])
    policy.fit([0.7, 0.8, 0.3], [0.01, 0.02, -0.01])
    assert policy.signal(0.8) == 1
    assert policy.signal(0.2) == -1
    assert policy.signal(0.5) == 0


def test_dynamic_threshold_policy():
    """TRAD-02: Dynamic threshold updates."""
    policy = DynamicThresholdPolicy(initial=0.5)
    assert policy.signal(0.7) == 1
    policy.update(0.7, 0.01)
    assert policy.threshold >= 0.2


def test_run_backtest(tmp_path):
    """BACK-01: Backtest computes PnL."""
    df = pl.DataFrame({
        "date": ["2024-01-15", "2024-01-16"],
        "asset_id": ["AAPL", "AAPL"],
        "return_1d": [0.01, -0.005],
    })
    df.write_parquet(tmp_path / "features.parquet")
    preds = pl.DataFrame({
        "date": ["2024-01-15", "2024-01-16"],
        "asset_id": ["AAPL", "AAPL"],
        "prob_up": [0.7, 0.3],
    })
    result = run_backtest(tmp_path / "features.parquet", preds)
    assert "total_pnl" in result
    assert "sharpe" in result
    assert "trades" in result
