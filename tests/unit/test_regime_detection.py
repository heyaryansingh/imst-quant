"""Regression tests for regime_detection look-ahead bias."""

import numpy as np
import polars as pl

from imst_quant.utils.regime_detection import detect_volatility_regime


def test_vol_regime_labels_do_not_use_future_data():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.01, 80)
    wild = rng.normal(0, 0.08, 40)
    full = detect_volatility_regime(pl.DataFrame({"return_1d": np.r_[calm, wild]}))
    prefix = detect_volatility_regime(pl.DataFrame({"return_1d": calm}))
    assert (
        full["vol_regime"].head(80).to_list() == prefix["vol_regime"].to_list()
    )
