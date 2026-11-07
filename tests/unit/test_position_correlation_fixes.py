import numpy as np
import polars as pl

from imst_quant.utils.position_correlation import calculate_position_correlation_matrix


def _df():
    rng = np.random.default_rng(0)
    a = rng.normal(size=30)
    b = a + rng.normal(size=30) * 0.1
    a[3] = np.nan
    return pl.DataFrame({"A": a, "B": b})


def test_pearson_ignores_nan_rows():
    m, _ = calculate_position_correlation_matrix(_df(), min_periods=20)
    assert np.isfinite(m).all()


def test_spearman_two_assets_returns_matrix():
    m, _ = calculate_position_correlation_matrix(_df(), method="spearman", min_periods=20)
    assert m.shape == (2, 2)
