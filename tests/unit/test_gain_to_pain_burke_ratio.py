import polars as pl
import pytest

from imst_quant.utils.risk_metrics import burke_ratio, gain_to_pain_ratio


def test_gain_to_pain_ratio_basic():
    returns = pl.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    # sum = 0.03, pain = 0.03
    assert gain_to_pain_ratio(returns) == pytest.approx(1.0)


def test_gain_to_pain_ratio_no_losses():
    returns = pl.Series([0.01, 0.02, 0.03])
    assert gain_to_pain_ratio(returns) == float("inf")


def test_gain_to_pain_ratio_no_gains():
    returns = pl.Series([-0.01, -0.02])
    # sum = -0.03, pain = 0.03 -> ratio -1.0
    assert gain_to_pain_ratio(returns) == pytest.approx(-1.0)


def test_gain_to_pain_ratio_dataframe_input():
    df = pl.DataFrame({"returns": [0.02, -0.02]})
    assert gain_to_pain_ratio(df) == pytest.approx(0.0)


def test_burke_ratio_no_drawdown():
    returns = pl.Series([0.01, 0.01, 0.01])
    assert burke_ratio(returns) == 100.0


def test_burke_ratio_penalizes_drawdown_path():
    calm = pl.Series([0.01, -0.001, 0.01, -0.001, 0.01])
    choppy = pl.Series([0.05, -0.04, 0.05, -0.04, 0.05])
    # Same-ish mean excess return, choppier drawdown path should score lower.
    assert burke_ratio(choppy) < burke_ratio(calm)


def test_burke_ratio_flat_returns():
    returns = pl.Series([0.0, 0.0, 0.0])
    assert burke_ratio(returns) == 0.0
