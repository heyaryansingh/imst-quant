"""Tests for VaRCalculator input validation."""

import pandas as pd
import pytest

from imst_quant.utils.var_calculator import VaRCalculator


@pytest.mark.parametrize("level", [0, 1, 95, -0.5])
def test_rejects_confidence_outside_unit_interval(level):
    with pytest.raises(ValueError, match="confidence_level"):
        VaRCalculator(pd.Series([0.01, -0.02, 0.03]), level, "parametric")


def test_historical_var_is_positive_loss():
    returns = pd.Series([-0.05, -0.01, 0.0, 0.01, 0.02] * 20)
    assert VaRCalculator(returns, 0.95).calculate_var() > 0
