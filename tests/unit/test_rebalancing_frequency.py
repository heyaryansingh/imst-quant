"""optimize_rebalancing_frequency simulates real drift instead of a fixed model."""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.rebalancing import optimize_rebalancing_frequency


def _long(rets: np.ndarray, symbols):
    rows = [
        {"date": t, "symbol": s, "returns": float(rets[t, i])}
        for t in range(rets.shape[0]) for i, s in enumerate(symbols)
    ]
    return pl.DataFrame(rows)


def test_identical_assets_never_drift():
    r = np.random.default_rng(0).normal(0.0005, 0.01, (300, 1))
    df = _long(np.hstack([r, r]), ["A", "B"])
    res = optimize_rebalancing_frequency(df, {"A": 0.5, "B": 0.5}, cost_bps_per_rebalance=10)
    assert res["annual_rebalancing_cost"] == pytest.approx(0.0, abs=1e-12)
    assert res["tracking_error"] == pytest.approx(0.0, abs=1e-9)


def test_uses_data_and_charges_turnover():
    rng = np.random.default_rng(1)
    rets = np.column_stack([rng.normal(0.001, 0.03, 500), rng.normal(0.0, 0.005, 500)])
    df = _long(rets, ["A", "B"])
    daily = optimize_rebalancing_frequency(df, {"A": 0.5, "B": 0.5}, 50, test_frequencies=[1])
    slow = optimize_rebalancing_frequency(df, {"A": 0.5, "B": 0.5}, 50, test_frequencies=[120])
    assert daily["annual_rebalancing_cost"] > slow["annual_rebalancing_cost"] > 0
    assert slow["tracking_error"] > daily["tracking_error"]


def test_missing_symbol_raises():
    df = _long(np.zeros((5, 1)), ["A"])
    with pytest.raises(ValueError):
        optimize_rebalancing_frequency(df, {"A": 0.5, "Z": 0.5})
