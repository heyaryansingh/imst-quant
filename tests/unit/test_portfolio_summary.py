import math

import numpy as np
import pandas as pd

from imst_quant.utils.portfolio_summary import PortfolioSummary


def test_short_position_pnl_pct_has_profit_sign():
    s = PortfolioSummary([{"symbol": "X", "quantity": -100, "entry": 50, "current": 40}], pd.Series([1.0, 1.0]))
    (pm,) = s.calculate_position_metrics()
    assert pm.unrealized_pnl == 1000
    assert math.isclose(pm.unrealized_pnl_pct, 20.0)
    assert math.isclose(s.calculate_portfolio_metrics().total_pnl_pct, 20.0)


def test_weekly_and_monthly_returns_span_5_and_21_periods():
    values = pd.Series(np.arange(1, 31, dtype=float))  # 30 values: 1..30
    m = PortfolioSummary([], values).calculate_portfolio_metrics()
    assert math.isclose(m.weekly_return, (30 / 25 - 1) * 100)
    assert math.isclose(m.monthly_return, (30 / 9 - 1) * 100)


def test_beta_of_benchmark_against_itself_is_one():
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 10))
    values = pd.Series(np.concatenate([[100.0], 100 * np.cumprod(1 + bench.values)]))
    returns = values.pct_change().dropna().reset_index(drop=True)
    s = PortfolioSummary([], values, benchmark_returns=returns)
    assert math.isclose(s._calculate_beta(returns), 1.0)
