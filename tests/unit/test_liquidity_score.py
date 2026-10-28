"""Tests for liquidity scoring utilities."""

import pandas as pd
import pytest

from imst_quant.utils.liquidity_score import AmihudIlliquidity


def test_amihud_skips_zero_volume_days():
    returns = pd.Series([0.01, -0.02, 0.01, 0.005])
    volumes = pd.Series([1e6, 1e6, 0.0, 1e6])
    prices = pd.Series([100.0] * 4)

    daily = AmihudIlliquidity.daily_illiquidity(returns, volumes, prices)
    assert pd.isna(daily.iloc[2])

    avg = AmihudIlliquidity.average_illiquidity(returns, volumes, prices)
    assert avg == pytest.approx((0.01 + 0.02 + 0.005) / 3 / 1e8)
