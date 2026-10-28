"""Signal backtest metric definitions: entry cost, hit rate, information ratio."""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.signal_backtest import backtest_signal, compare_signals

IDX = pd.date_range("2024-01-01", periods=5)


def test_entry_from_flat_is_charged():
    signal = pd.Series([1, 1, 1, 1, 1], index=IDX)
    returns = pd.Series([0.0] * 5, index=IDX)
    res = backtest_signal(signal, returns, transaction_cost=0.01)
    assert res.total_return == pytest.approx(-0.01)


def test_hit_rate_ignores_flat_periods():
    signal = pd.Series([1, 0, 0, 0, 0], index=IDX)
    returns = pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=IDX)
    # One active period, and it was right
    assert backtest_signal(signal, returns).hit_rate == 1.0


def test_buy_and_hold_signal_has_zero_information_ratio():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=300)
    returns = pd.Series(rng.normal(0.001, 0.01, 300), index=idx)
    out = compare_signals({"long": pd.Series(1.0, index=idx)}, returns)
    # Old code divided by a vol *difference* ~0, exploding the ratio
    assert out.information_ratios[0] == 0.0
