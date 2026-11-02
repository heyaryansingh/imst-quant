import numpy as np
import polars as pl

from imst_quant.utils.mean_reversion import _to_numpy, estimate_half_life


def test_polars_nan_and_inf_dropped_like_numpy():
    raw = [1.0, float("nan"), 2.0, None, float("inf"), 3.0]
    expected = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(_to_numpy(pl.Series(raw)), expected)
    assert np.array_equal(_to_numpy(np.array(raw, dtype=float)), expected)
    assert np.array_equal(_to_numpy(raw), expected)


def test_half_life_finite_with_polars_nan():
    rng = np.random.default_rng(0)
    x = [0.0]
    for _ in range(300):
        x.append(0.8 * x[-1] + rng.normal())
    x[50] = float("nan")
    hl = estimate_half_life(pl.Series(x))
    assert np.isfinite(hl)
