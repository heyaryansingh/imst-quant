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


def test_flat_window_sharpe_and_ir_are_nan_not_inf():
    import numpy as np
    import pandas as pd

    from imst_quant.utils.rolling_performance import (
        calculate_rolling_information_ratio,
        calculate_rolling_sharpe,
    )

    r = pd.Series([0.01, -0.02, 0.03, 0.001, 0.001, 0.001, 0.001, 0.001])
    sharpe = calculate_rolling_sharpe(r, 4)
    assert np.isfinite(sharpe.iloc[3]) and sharpe.iloc[-2:].isna().all()
    ir = calculate_rolling_information_ratio(r, r * 0.0 + 0.0005, 4)
    assert not np.isinf(ir).any() and ir.iloc[-2:].isna().all()
