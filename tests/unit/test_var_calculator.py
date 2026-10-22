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


@pytest.mark.parametrize("method", ["parametric", "historical", "monte_carlo"])
def test_inf_returns_are_ignored(method):
    clean = pd.Series([-0.05, -0.01, 0.0, 0.01, 0.02] * 20)
    dirty = pd.concat([clean, pd.Series([float("inf"), float("-inf"), float("nan")])])
    calc = VaRCalculator(dirty, 0.95, method)
    var, cvar = calc.calculate_var(), calc.conditional_var()
    assert pd.notna(var) and pd.notna(cvar)
    if method != "monte_carlo":
        ref = VaRCalculator(clean, 0.95, method)
        assert var == pytest.approx(ref.calculate_var())
        assert cvar == pytest.approx(ref.conditional_var())
