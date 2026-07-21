"""Fractional differentiation for stationarity with memory preservation.

Integer differencing (returns) makes prices stationary but destroys the
long-memory structure predictive models rely on. Fractional differentiation
(Lopez de Prado, "Advances in Financial Machine Learning", 2018, ch. 5)
applies the binomial expansion of (1 - B)^d for a real order d in (0, 1),
producing a series that is stationary while retaining maximal correlation
with the original level series.

This module implements the fixed-width window (FFD) variant, which uses a
truncated weight vector so every output point is computed from the same
number of lags.
"""

import numpy as np
from typing import Any, Dict, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def ffd_weights(d: float, threshold: float = 1e-4, max_width: int = 10000) -> np.ndarray:
    """Compute truncated fractional differentiation weights.

    Weights follow the recursion w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k,
    truncated once |w_k| drops below threshold.

    Args:
        d: Differentiation order (0 = identity, 1 = first difference).
        threshold: Truncation cutoff on absolute weight (default 1e-4).
        max_width: Hard cap on window width (default 10000).

    Returns:
        Weight array [w_0, w_1, ..., w_{K-1}], newest observation first.

    Raises:
        ValueError: If d < 0 or threshold <= 0.
    """
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    weights = [1.0]
    k = 1
    while k < max_width:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.asarray(weights)


def frac_diff_ffd(
    series: ArrayLike,
    d: float,
    threshold: float = 1e-4,
) -> Dict[str, Any]:
    """Fractionally differentiate a series with a fixed-width window.

    Args:
        series: Level series (e.g. log prices).
        d: Differentiation order in [0, 1] typically.
        threshold: Weight truncation cutoff (default 1e-4).

    Returns:
        Dict with:
            values: Differentiated series (length n - width + 1).
            start_index: Index into the input where output begins.
            width: Window width used.
            weights: Weight vector applied.

    Raises:
        ValueError: If the series is shorter than the weight window.
    """
    arr = np.asarray(series, dtype=float)
    weights = ffd_weights(d, threshold)
    width = weights.size
    if arr.size < width:
        raise ValueError(
            f"Series length {arr.size} shorter than FFD window {width}; "
            "raise threshold or supply more data"
        )

    # output[i] = sum_k weights[k] * arr[i - k]; np.convolve flips the
    # kernel internally, so full-mode index i is exactly this sum.
    values = np.convolve(arr, weights, mode="full")[width - 1 : arr.size]

    return {
        "values": values,
        "start_index": width - 1,
        "width": int(width),
        "weights": weights,
    }


def memory_vs_stationarity(
    series: ArrayLike,
    d_grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
    threshold: float = 1e-4,
) -> Dict[str, Any]:
    """Profile the memory/stationarity trade-off across differentiation orders.

    For each d in the grid, computes the correlation between the
    differentiated series and the original levels (memory retained) and the
    lag-1 autocorrelation of the differentiated series (persistence). Lower d
    keeps more memory; higher d looks more like white noise.

    Args:
        series: Level series (e.g. log prices).
        d_grid: Differentiation orders to evaluate.
        threshold: Weight truncation cutoff.

    Returns:
        Dict with per-order entries {d, correlation_with_original,
        lag1_autocorr, std, width} and the input length.

    Raises:
        ValueError: If the series is too short for the largest window.
    """
    arr = np.asarray(series, dtype=float)
    profiles = []
    for d in d_grid:
        result = frac_diff_ffd(arr, d, threshold)
        values = result["values"]
        aligned = arr[result["start_index"]:]
        if values.size < 3 or np.std(values) == 0 or np.std(aligned) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(values, aligned)[0, 1])
        if values.size < 3 or np.std(values) == 0:
            lag1 = float("nan")
        else:
            lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        profiles.append(
            {
                "d": float(d),
                "correlation_with_original": corr,
                "lag1_autocorr": lag1,
                "std": float(np.std(values)),
                "width": result["width"],
            }
        )

    return {"n_obs": int(arr.size), "profiles": profiles}
