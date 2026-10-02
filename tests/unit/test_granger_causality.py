"""Tests for granger_causality null handling."""

import numpy as np
import polars as pl

from imst_quant.utils.granger_causality import granger_causality_test


def test_nulls_do_not_shift_x_against_y():
    # x Granger-causes y at lag 1. A single null early in x used to slide every
    # later x one step back, pairing y_t with x_t and destroying the relation.
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    y = np.r_[0.0, 0.8 * x[:-1]] + rng.normal(scale=0.1, size=n)
    clean = granger_causality_test(pl.Series("x", x), pl.Series("y", y), max_lag=1)[0]
    xs = x.tolist()
    xs[10] = None
    gapped = granger_causality_test(pl.Series("x", xs), pl.Series("y", y), max_lag=1)[0]
    assert gapped.is_significant
    assert gapped.f_statistic > 0.5 * clean.f_statistic


def test_nan_values_are_dropped_pairwise():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = np.r_[0.0, 0.8 * x[:-1]] + rng.normal(scale=0.1, size=200)
    x[50] = np.nan
    results = granger_causality_test(pl.Series("x", x), pl.Series("y", y), max_lag=1)
    assert np.isfinite(results[0].f_statistic)
    assert results[0].is_significant
