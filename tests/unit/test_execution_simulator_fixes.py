import numpy as np
from imst_quant.utils import execution_simulator as es
import polars as pl
from datetime import datetime, timedelta


def test_partial_fill_does_not_overshoot_position(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # always partial
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 0.5)
    t0 = datetime(2024, 1, 1)
    sig = pl.DataFrame({"timestamp": [t0, t0 + timedelta(days=1)], "signal": [1, 1]})
    px = pl.DataFrame({"timestamp": sig["timestamp"], "close": [100.0, 100.0], "volume": [1e9, 1e9]})
    fills, _ = es.simulate_execution_scenario(sig, px, initial_capital=10000, position_size=1.0)
    net = sum(q if s == "buy" else -q for q, s in zip(fills["quantity"], fills["side"]))
    assert net <= 100.0 + 1e-9  # target is 100 shares
