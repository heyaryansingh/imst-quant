"""Tests for GARCH forecasting and the volatility cone."""

import numpy as np
import pandas as pd

from imst_quant.utils.volatility_forecast import volatility_cone, volatility_forecast


def _returns(n: int = 300) -> pd.Series:
    return pd.Series(np.random.default_rng(0).normal(0, 0.01, n))


def test_cone_skipped_window_does_not_shift_labels():
    returns = _returns(100)
    cone = volatility_cone(returns, windows=[252, 5])
    assert cone.windows == [5]
    expected = returns.iloc[-5:].std() * np.sqrt(252)
    assert np.isclose(cone.current[0], expected)


def test_garch_forecast_reacts_to_shock_and_reverts():
    returns = _returns()
    returns.iloc[-1] = -0.08  # crash on the last bar
    fc = volatility_forecast(returns, method="garch")
    long_run = returns.std() * np.sqrt(252)
    # The shock is not yet in sigma_T but must drive the 1-day forecast.
    assert fc.forecast_1d > fc.current
    assert fc.forecast_1d > fc.forecast_5d > fc.forecast_21d > long_run
    # alpha + beta = 0.95 -> half-life ln(0.5)/ln(0.95)
    assert np.isclose(fc.half_life, np.log(0.5) / np.log(0.95))


def test_garch_forecast_rises_toward_long_run_from_calm():
    returns = _returns()
    returns.iloc[-40:] = 0.0001  # vol collapsed recently
    fc = volatility_forecast(returns, method="garch")
    assert fc.forecast_1d < fc.forecast_5d < fc.forecast_21d
