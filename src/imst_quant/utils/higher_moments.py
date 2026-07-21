"""Higher-moment systematic risk: downside beta, coskewness, cokurtosis.

Ordinary beta assumes co-movement is symmetric. In practice assets that
fall harder in down markets deserve a risk premium, which these measures
capture:

- Downside/upside beta (Bawa & Lindenberg, 1977): beta computed only over
  periods when the benchmark is below/above a threshold.
- Coskewness: whether the asset amplifies benchmark variance shocks
  (negative coskewness = loses exactly when volatility spikes = bad).
- Cokurtosis: sensitivity to benchmark tail events.
"""

import numpy as np
from typing import Any, Dict, Sequence

Vector = Sequence[float]


def _validate_pair(asset: Vector, benchmark: Vector, min_obs: int = 20) -> tuple:
    a = np.asarray(asset, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("asset and benchmark must be 1D arrays")
    if a.size != b.size:
        raise ValueError(f"Length mismatch: asset {a.size} vs benchmark {b.size}")
    if a.size < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {a.size}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("inputs contain non-finite values")
    return a, b


def _conditional_beta(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 10:
        raise ValueError("Fewer than 10 observations in the conditioning regime")
    bs = b[mask]
    var = bs.var(ddof=1)
    # Tolerance, not == 0: a constant np.full array has var ~1e-36 from summation noise
    if var < 1e-18:
        raise ValueError("Benchmark has zero variance in the conditioning regime")
    cov = np.cov(a[mask], bs, ddof=1)[0, 1]
    return float(cov / var)


def downside_beta(asset: Vector, benchmark: Vector, threshold: float = 0.0) -> float:
    """Beta over periods when the benchmark return is below ``threshold``.

    Args:
        asset: 1D asset return series.
        benchmark: 1D benchmark return series, same length.
        threshold: Benchmark return defining "down" periods (default 0).

    Returns:
        Downside beta. Values above the unconditional beta mean the asset
        is disproportionately exposed to benchmark losses.

    Raises:
        ValueError: If inputs are malformed or too few down periods exist.
    """
    a, b = _validate_pair(asset, benchmark)
    return _conditional_beta(a, b, b < threshold)


def upside_beta(asset: Vector, benchmark: Vector, threshold: float = 0.0) -> float:
    """Beta over periods when the benchmark return is above ``threshold``.

    Args:
        asset: 1D asset return series.
        benchmark: 1D benchmark return series, same length.
        threshold: Benchmark return defining "up" periods (default 0).

    Returns:
        Upside beta.

    Raises:
        ValueError: If inputs are malformed or too few up periods exist.
    """
    a, b = _validate_pair(asset, benchmark)
    return _conditional_beta(a, b, b > threshold)


def coskewness(asset: Vector, benchmark: Vector) -> float:
    """Standardized coskewness E[(ra-ua)(rb-ub)^2] / (sa * sb^2).

    Args:
        asset: 1D asset return series.
        benchmark: 1D benchmark return series, same length.

    Returns:
        Coskewness. Negative values mean the asset tends to fall when
        benchmark volatility spikes — undesirable tail behavior.

    Raises:
        ValueError: If inputs are malformed or either series is constant.
    """
    a, b = _validate_pair(asset, benchmark)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if sa < 1e-12 or sb < 1e-12:
        raise ValueError("inputs must not be constant")
    da, db = a - a.mean(), b - b.mean()
    return float((da * db**2).mean() / (sa * sb**2))


def cokurtosis(asset: Vector, benchmark: Vector) -> float:
    """Standardized cokurtosis E[(ra-ua)(rb-ub)^3] / (sa * sb^3).

    Args:
        asset: 1D asset return series.
        benchmark: 1D benchmark return series, same length.

    Returns:
        Cokurtosis — sensitivity of the asset to benchmark tail moves.

    Raises:
        ValueError: If inputs are malformed or either series is constant.
    """
    a, b = _validate_pair(asset, benchmark)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if sa < 1e-12 or sb < 1e-12:
        raise ValueError("inputs must not be constant")
    da, db = a - a.mean(), b - b.mean()
    return float((da * db**3).mean() / (sa * sb**3))


def analyze_higher_moments(
    asset: Vector,
    benchmark: Vector,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Full higher-moment risk profile of an asset against a benchmark.

    Args:
        asset: 1D asset return series.
        benchmark: 1D benchmark return series, same length.
        threshold: Benchmark return separating up and down regimes.

    Returns:
        Dict with beta (unconditional), downside_beta, upside_beta,
        beta_asymmetry (downside - upside; positive = crash-heavy),
        coskewness, cokurtosis, n_observations, n_down_periods,
        n_up_periods, and assessment ("defensive" when beta_asymmetry
        < -0.1, "symmetric" within +/-0.1, else "crash_exposed").

    Raises:
        ValueError: If inputs are malformed or a regime has too few periods.
    """
    a, b = _validate_pair(asset, benchmark)
    var_b = b.var(ddof=1)
    if var_b < 1e-18:
        raise ValueError("benchmark has zero variance")
    beta = float(np.cov(a, b, ddof=1)[0, 1] / var_b)
    d_beta = downside_beta(a, b, threshold=threshold)
    u_beta = upside_beta(a, b, threshold=threshold)
    asymmetry = d_beta - u_beta
    if asymmetry < -0.1:
        assessment = "defensive"
    elif asymmetry <= 0.1:
        assessment = "symmetric"
    else:
        assessment = "crash_exposed"
    return {
        "beta": beta,
        "downside_beta": d_beta,
        "upside_beta": u_beta,
        "beta_asymmetry": float(asymmetry),
        "coskewness": coskewness(a, b),
        "cokurtosis": cokurtosis(a, b),
        "n_observations": int(a.size),
        "n_down_periods": int((b < threshold).sum()),
        "n_up_periods": int((b > threshold).sum()),
        "assessment": assessment,
    }
