"""Contrarian/oscillator signals must hold (0) on a NaN close, not trade."""

import numpy as np
import polars as pl
import pytest

from imst_quant.trading import signals as S


@pytest.mark.parametrize("fn,col", [
    (S.mean_reversion_signal, "reversion_signal"),
    (S.rsi_signal, "rsi_signal"),
    (S.bollinger_band_signal, "bb_signal"),
    (S.bollinger_bands_signal, "bb_signal"),
    (S.stochastic_signal, "stoch_signal"),
])
def test_nan_close_holds(fn, col):
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 60))
    c[45] = np.nan
    df = pl.DataFrame({"close": c, "high": c + 1, "low": c - 1})
    sig = fn(df)[col]
    assert sig[45] == 0
    assert sig.drop_nulls().abs().sum() > 0  # still trades on clean bars
