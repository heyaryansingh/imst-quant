"""Tests for Cornish-Fisher VaR edge cases."""

import numpy as np
import polars as pl

from imst_quant.utils.cornish_fisher_var import (
    cornish_fisher_cvar,
    cornish_fisher_var,
    risk_summary,
)


def test_nan_returns_are_ignored_not_propagated():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.01, 250)
    dirty = pl.Series("r", np.append(clean, [np.nan, np.inf]))
    assert cornish_fisher_var(dirty) == cornish_fisher_var(pl.Series("r", clean))
    summary = risk_summary(dirty)
    assert np.isfinite(summary.parametric_var)
    assert np.isfinite(summary.cornish_fisher_cvar)


def test_cvar_never_below_var_when_tail_is_empty():
    # Fat tails at 99.9% push the CF quantile past every observed loss.
    a = np.random.default_rng(1).standard_t(3, 250) * 0.01
    r = pl.Series("r", a)
    var = cornish_fisher_var(r, 0.999)
    assert var > -a.min()  # beyond the worst sample, so the empirical tail is empty
    assert cornish_fisher_cvar(r, 0.999) >= var
