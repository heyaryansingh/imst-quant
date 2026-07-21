"""Hurst exponent and variance ratio tests for trend/mean-reversion detection.

The Hurst exponent H characterizes the long-memory behavior of a time
series: H < 0.5 indicates mean reversion, H = 0.5 a random walk, and
H > 0.5 a trending (persistent) series. It is estimated here via rescaled
range (R/S) analysis (Hurst, 1951) and the aggregated variance method.

The Lo-MacKinlay (1988) variance ratio test complements the Hurst estimate
with a formal hypothesis test of the random-walk null: VR(q) significantly
below 1 suggests mean reversion, above 1 suggests momentum.
"""

import numpy as np
from typing import Any, Dict, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def _validate_returns(returns: ArrayLike, min_obs: int) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_obs:
        raise ValueError(f"Need at least {min_obs} finite observations, got {arr.size}")
    return arr


def rescaled_range_hurst(
    returns: ArrayLike,
    min_window: int = 8,
    max_window: int = None,
) -> Dict[str, Any]:
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    Splits the series into non-overlapping windows of increasing size,
    computes the mean rescaled range R/S per size, and fits
    log(R/S) = H * log(n) + c by least squares.

    Args:
        returns: Return series (not prices).
        min_window: Smallest window size to include (default 8).
        max_window: Largest window size (default: len(returns) // 2).

    Returns:
        Dict with hurst, intercept, r_squared, window_sizes, and
        rs_values (mean R/S per window size).

    Raises:
        ValueError: If fewer than 4 * min_window observations.
    """
    arr = _validate_returns(returns, min_obs=4 * min_window)
    n = arr.size
    if max_window is None:
        max_window = n // 2
    max_window = min(max_window, n // 2)

    sizes = np.unique(
        np.floor(np.logspace(np.log10(min_window), np.log10(max_window), 12)).astype(int)
    )
    sizes = sizes[sizes >= min_window]

    log_sizes = []
    log_rs = []
    rs_values = []
    for size in sizes:
        n_chunks = n // size
        rs_chunk = []
        for i in range(n_chunks):
            chunk = arr[i * size : (i + 1) * size]
            dev = np.cumsum(chunk - chunk.mean())
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            mean_rs = float(np.mean(rs_chunk))
            rs_values.append(mean_rs)
            log_sizes.append(np.log(size))
            log_rs.append(np.log(mean_rs))

    if len(log_sizes) < 3:
        raise ValueError("Not enough valid window sizes for R/S regression")

    slope, intercept = np.polyfit(log_sizes, log_rs, 1)
    fitted = slope * np.array(log_sizes) + intercept
    resid = np.array(log_rs) - fitted
    ss_tot = float(np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2))
    r_squared = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0

    return {
        "hurst": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "window_sizes": [int(s) for s in sizes[: len(rs_values)]],
        "rs_values": rs_values,
    }


def aggregated_variance_hurst(
    returns: ArrayLike,
    min_window: int = 2,
    max_window: int = None,
) -> Dict[str, Any]:
    """Estimate the Hurst exponent via the aggregated variance method.

    For a self-similar process, Var(aggregated mean over m points) scales
    as m^(2H - 2). Fits log(variance) against log(m) and recovers H from
    the slope: H = 1 + slope / 2.

    Args:
        returns: Return series.
        min_window: Smallest aggregation level (default 2).
        max_window: Largest aggregation level (default: len(returns) // 10).

    Returns:
        Dict with hurst, slope, and r_squared of the log-log fit.

    Raises:
        ValueError: If fewer than 40 observations.
    """
    arr = _validate_returns(returns, min_obs=40)
    n = arr.size
    if max_window is None:
        max_window = max(n // 10, min_window + 2)
    max_window = min(max_window, n // 4)

    sizes = np.unique(
        np.floor(np.logspace(np.log10(min_window), np.log10(max_window), 10)).astype(int)
    )
    sizes = sizes[sizes >= min_window]

    log_m = []
    log_var = []
    for m in sizes:
        n_chunks = n // m
        if n_chunks < 4:
            continue
        means = arr[: n_chunks * m].reshape(n_chunks, m).mean(axis=1)
        v = means.var(ddof=1)
        if v > 0:
            log_m.append(np.log(m))
            log_var.append(np.log(v))

    if len(log_m) < 3:
        raise ValueError("Not enough aggregation levels for variance regression")

    slope, _ = np.polyfit(log_m, log_var, 1)
    fitted = np.polyval(np.polyfit(log_m, log_var, 1), log_m)
    resid = np.array(log_var) - fitted
    ss_tot = float(np.sum((np.array(log_var) - np.mean(log_var)) ** 2))
    r_squared = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0

    return {
        "hurst": float(1.0 + slope / 2.0),
        "slope": float(slope),
        "r_squared": r_squared,
    }


def variance_ratio_test(returns: ArrayLike, lag: int = 2) -> Dict[str, Any]:
    """Lo-MacKinlay variance ratio test of the random-walk hypothesis.

    Computes VR(q) = Var(q-period returns) / (q * Var(1-period returns))
    with the heteroskedasticity-robust z-statistic of Lo & MacKinlay (1988).

    Args:
        returns: Return series.
        lag: Aggregation period q (default 2). Must be >= 2.

    Returns:
        Dict with variance_ratio, z_score, p_value (two-sided, normal
        approximation), lag, and interpretation ("mean_reverting",
        "random_walk", or "trending").

    Raises:
        ValueError: If lag < 2 or the series is too short.
    """
    if lag < 2:
        raise ValueError("lag must be at least 2")
    arr = _validate_returns(returns, min_obs=lag * 10)
    n = arr.size

    mu = arr.mean()
    var_1 = float(np.sum((arr - mu) ** 2)) / (n - 1)
    if var_1 <= 0:
        raise ValueError("Return series has zero variance")

    # Overlapping q-period sums with small-sample bias correction per
    # Lo-MacKinlay; the divisor m already contains the factor q, so var_q
    # is a per-period variance directly comparable to var_1.
    q_sums = np.convolve(arr, np.ones(lag), mode="valid")
    m = lag * (n - lag + 1) * (1 - lag / n)
    var_q = float(np.sum((q_sums - lag * mu) ** 2)) / m
    vr = var_q / var_1

    # Heteroskedasticity-robust asymptotic variance
    demeaned_sq = (arr - mu) ** 2
    theta = 0.0
    denom = float(np.sum(demeaned_sq)) ** 2
    for j in range(1, lag):
        delta = n * float(np.sum(demeaned_sq[j:] * demeaned_sq[:-j])) / denom
        theta += (2 * (lag - j) / lag) ** 2 * delta

    z = np.sqrt(n) * (vr - 1.0) / np.sqrt(theta) if theta > 0 else 0.0
    # Two-sided p-value via error function (avoids scipy dependency here)
    from math import erf, sqrt

    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

    if p_value < 0.05:
        interpretation = "mean_reverting" if vr < 1.0 else "trending"
    else:
        interpretation = "random_walk"

    return {
        "variance_ratio": float(vr),
        "z_score": float(z),
        "p_value": float(p_value),
        "lag": lag,
        "interpretation": interpretation,
    }


def classify_regime(hurst: float, tolerance: float = 0.05) -> str:
    """Classify series behavior from a Hurst exponent estimate.

    Args:
        hurst: Estimated Hurst exponent.
        tolerance: Half-width of the band around 0.5 treated as a random
            walk (default 0.05).

    Returns:
        "mean_reverting" (H < 0.5 - tolerance), "trending"
        (H > 0.5 + tolerance), or "random_walk" otherwise.
    """
    if hurst < 0.5 - tolerance:
        return "mean_reverting"
    if hurst > 0.5 + tolerance:
        return "trending"
    return "random_walk"


def analyze_hurst(
    returns: ArrayLike,
    variance_ratio_lags: Sequence[int] = (2, 5, 10),
) -> Dict[str, Any]:
    """Full long-memory analysis: R/S Hurst, aggregated variance, VR tests.

    Args:
        returns: Return series.
        variance_ratio_lags: Lags for the variance ratio tests.

    Returns:
        Dict with n_observations, hurst_rs, hurst_aggvar, regime
        (classification from the R/S estimate), and variance_ratios
        (list of variance_ratio_test results, skipping lags the series
        is too short for).
    """
    arr = _validate_returns(returns, min_obs=40)

    rs = rescaled_range_hurst(arr)
    aggvar = aggregated_variance_hurst(arr)

    vr_results = []
    for lag in variance_ratio_lags:
        if arr.size >= lag * 10:
            vr_results.append(variance_ratio_test(arr, lag=lag))

    return {
        "n_observations": int(arr.size),
        "hurst_rs": rs["hurst"],
        "hurst_rs_r_squared": rs["r_squared"],
        "hurst_aggvar": aggvar["hurst"],
        "regime": classify_regime(rs["hurst"]),
        "variance_ratios": vr_results,
    }
