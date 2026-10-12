"""Tests for signal correlation utilities."""

import numpy as np
import polars as pl

from imst_quant.utils.signal_correlation import (
    analyze_signal_lead_lag,
    calculate_signal_correlation_matrix,
    calculate_signal_diversification,
)


def test_lead_lag_distinguishes_future_from_past_returns():
    r = np.random.default_rng(0).normal(size=200)
    # signal_t equals the next bar's return: predictive at lag +1 only
    df = pl.DataFrame({"s": r[1:], "returns": r[:-1]})
    lags = analyze_signal_lead_lag(df, "s", max_lag=2)
    assert lags[1] > 0.99
    assert abs(lags[-1]) < 0.3


def test_constant_signal_is_uncorrelated_not_nan():
    df = pl.DataFrame({"flat": [1.0] * 5, "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    corr = calculate_signal_correlation_matrix(df, ["flat", "b"])
    assert corr["b"][0] == 0.0
    assert calculate_signal_diversification(df, ["flat", "b"]) == 1.0
