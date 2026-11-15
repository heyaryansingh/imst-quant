"""Constant series must not leak NaN correlations into divergence or dashboard output."""

import math
import warnings

import numpy as np
import pandas as pd

from imst_quant.sentiment.advanced_scoring import compute_sentiment_divergence
from imst_quant.utils.risk_dashboard import RiskDashboard


def test_divergence_constant_inputs_is_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_sentiment_divergence([0.5, 0.5, 0.5], [0.01, -0.02, 0.03]) == 0.0
        assert compute_sentiment_divergence([0.1, 0.5, 0.9], [0.0, 0.0, 0.0]) == 0.0
    assert compute_sentiment_divergence([0.1, 0.5, 0.9], [0.03, 0.0, -0.03]) > 0.99


def test_dashboard_correlation_finite_with_flat_benchmark():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 100))
    positions = pd.DataFrame({"symbol": ["A", "B"], "weight": [0.5, 0.5], "returns": [0.01, 0.02]})
    m = RiskDashboard(returns, positions, benchmark_returns=pd.Series(np.zeros(100))).generate_dashboard()
    assert m.correlation_to_benchmark == 0.0
    m = RiskDashboard(returns, positions, benchmark_returns=returns * 2).generate_dashboard()
    assert math.isclose(m.correlation_to_benchmark, 1.0)
