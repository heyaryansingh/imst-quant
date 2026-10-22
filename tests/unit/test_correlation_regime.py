"""Tests for correlation regime detection."""

import numpy as np
import polars as pl

from imst_quant.utils.correlation_regime import (
    CorrelationRegime,
    correlation_stability,
    detect_correlation_regime,
    rolling_correlation_matrix,
)


def _correlated(n=300, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    noise = np.sqrt(1 - rho**2) / rho
    return pl.DataFrame(
        {name: common + rng.normal(0, 0.01 * noise, n) for name in ("A", "B", "C")}
    )


def test_highly_correlated_assets_are_crisis_regime():
    result = detect_correlation_regime(_correlated())
    assert result.regime == CorrelationRegime.CRISIS
    assert result.mean_correlation > 0.7
    assert result.eigenvalue_concentration > 0.8


def test_independent_assets_are_decorrelated():
    rng = np.random.default_rng(1)
    df = pl.DataFrame({k: rng.normal(0, 0.01, 300) for k in ("A", "B", "C")})
    assert detect_correlation_regime(df).regime == CorrelationRegime.DECORRELATED


def test_non_finite_rows_do_not_fake_decorrelation():
    df = _correlated()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 290)
        .then(float("nan"))
        .otherwise(pl.col("A"))
        .alias("A"),
        pl.when(pl.int_range(pl.len()) == 295)
        .then(None)
        .otherwise(pl.col("B"))
        .alias("B"),
    )
    result = detect_correlation_regime(df)
    assert result.regime == CorrelationRegime.CRISIS
    assert np.isfinite(result.mean_correlation)
    assert all(np.isfinite(m["mean_corr"]) for m in rolling_correlation_matrix(df))
    assert correlation_stability(df).is_stable
