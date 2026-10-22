"""Tests for Black-Scholes pricing, Greeks and implied volatility."""

import numpy as np
import pytest

from imst_quant.utils.options_strategies import (
    BlackScholesModel,
    ImpliedVolatilityCalculator,
    OptionType,
)

CALL, PUT = OptionType.CALL, OptionType.PUT


def test_put_call_parity():
    s, k, t, v, r = 100.0, 105.0, 0.5, 0.25, 0.04
    c = BlackScholesModel.price(s, k, t, v, r, option_type=CALL)
    p = BlackScholesModel.price(s, k, t, v, r, option_type=PUT)
    assert c - p == pytest.approx(s - k * np.exp(-r * t))


@pytest.mark.parametrize("spot,kind,expected", [
    (110, CALL, 1.0), (90, CALL, 0.0), (90, PUT, -1.0), (110, PUT, 0.0),
])
def test_expired_delta(spot, kind, expected):
    assert BlackScholesModel.greeks(spot, 100, 0, 0.2, 0.05, option_type=kind)["delta"] == expected


def test_implied_vol_round_trip():
    price = BlackScholesModel.price(100, 95, 0.25, 0.35, 0.03, option_type=PUT)
    iv = ImpliedVolatilityCalculator.calculate_iv(price, 100, 95, 0.25, 0.03, option_type=PUT)
    assert iv == pytest.approx(0.35, abs=1e-4)
