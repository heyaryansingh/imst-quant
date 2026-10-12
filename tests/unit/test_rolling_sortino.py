"""Tests for the rolling Sortino ratio's downside deviation."""

import numpy as np
import pandas as pd
import pytest

from imst_quant.utils.rolling_performance import calculate_rolling_sortino


def test_constant_losses_are_not_nan_or_infinite():
    result = calculate_rolling_sortino(pd.Series([-0.01] * 10), window=5, annualization_factor=1)
    assert result.iloc[-1] == pytest.approx(-1.0)


def test_matches_rms_shortfall_definition():
    returns = pd.Series([0.10, -0.05, 0.03, -0.02, 0.04])
    result = calculate_rolling_sortino(returns, window=5, annualization_factor=1)
    expected = 0.02 / np.sqrt((0.05**2 + 0.02**2) / 5)
    assert result.iloc[-1] == pytest.approx(expected)


def test_window_without_losses_is_nan():
    result = calculate_rolling_sortino(pd.Series([0.01] * 6), window=3)
    assert result.isna().all()
