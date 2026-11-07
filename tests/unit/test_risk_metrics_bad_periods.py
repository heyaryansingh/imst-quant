"""Standalone risk metrics ignore null/NaN/inf periods instead of returning NaN."""

import math

import polars as pl
import pytest

from imst_quant.utils import risk_metrics as rm

CLEAN = [0.01, -0.02, 0.015, -0.005, 0.02]
DIRTY = [0.01, math.nan, -0.02, None, 0.015, math.inf, -0.005, 0.02]

FUNCS = [
    rm.sharpe_ratio, rm.sortino_ratio, rm.max_drawdown, rm.value_at_risk,
    rm.calmar_ratio, rm.calculate_omega_ratio, rm.calculate_tail_ratio,
    rm.gain_to_pain_ratio, rm.burke_ratio,
]


@pytest.mark.parametrize("fn", FUNCS, ids=lambda f: f.__name__)
def test_matches_clean_series(fn):
    clean = fn(pl.Series(CLEAN))
    dirty = fn(pl.Series(DIRTY, dtype=pl.Float64))
    assert not math.isnan(dirty)
    assert dirty == pytest.approx(clean)


def test_dataframe_input_cleaned_too():
    df = pl.DataFrame({"returns": pl.Series(DIRTY, dtype=pl.Float64)})
    assert rm.calculate_omega_ratio(df) == pytest.approx(rm.calculate_omega_ratio(pl.Series(CLEAN)))
