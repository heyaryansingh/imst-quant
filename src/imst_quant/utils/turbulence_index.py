"""Financial turbulence index and absorption ratio for systemic risk monitoring.

The turbulence index (Kritzman & Li, 2010) measures how statistically
unusual a day's cross-asset return vector is, via the Mahalanobis distance
from the historical mean under the historical covariance. Elevated
turbulence has historically coincided with poor risk-asset performance and
degraded diversification.

The absorption ratio (Kritzman et al., 2011) measures the fraction of total
asset variance explained by the top principal components. A high or rising
absorption ratio means markets are tightly coupled and shocks propagate
broadly — a fragility indicator.
"""

import numpy as np
from typing import Any, Dict, Optional, Sequence

Matrix = Sequence[Sequence[float]]


def _validate_matrix(returns: Matrix, min_rows: int) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (periods, assets)")
    if arr.shape[0] < min_rows:
        raise ValueError(f"Need at least {min_rows} periods, got {arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError("Need at least 2 assets")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values")
    return arr


def turbulence_index(
    returns: Matrix,
    lookback: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the Mahalanobis turbulence index for each period.

    For each period t, distance d_t = (r_t - mu)' Sigma^-1 (r_t - mu),
    where mu and Sigma are estimated over the lookback window (or the full
    sample when lookback is None). Under multivariate normality d_t is
    approximately chi-squared with n_assets degrees of freedom, so
    dividing by n_assets normalizes the typical value to ~1.

    Args:
        returns: (periods, assets) matrix of returns.
        lookback: Trailing window for mean/covariance estimation. When
            None, uses the full sample (in-sample turbulence). When set,
            the first ``lookback`` periods have no score (NaN).

    Returns:
        Dict with turbulence (per-period normalized distances, list with
        NaN during warmup), n_assets, mean_turbulence, max_turbulence,
        and current_turbulence (last period).

    Raises:
        ValueError: If input is malformed or too short.
    """
    arr = _validate_matrix(returns, min_rows=10)
    n_periods, n_assets = arr.shape

    scores = np.full(n_periods, np.nan)

    def _mahalanobis(window: np.ndarray, obs: np.ndarray) -> float:
        mu = window.mean(axis=0)
        cov = np.cov(window, rowvar=False)
        # Ridge for numerical stability on near-singular covariances
        cov += np.eye(n_assets) * 1e-10 * np.trace(cov)
        diff = obs - mu
        return float(diff @ np.linalg.solve(cov, diff))

    if lookback is None:
        for t in range(n_periods):
            scores[t] = _mahalanobis(arr, arr[t]) / n_assets
    else:
        if lookback < max(10, n_assets + 2):
            raise ValueError(f"lookback must be at least {max(10, n_assets + 2)}")
        for t in range(lookback, n_periods):
            scores[t] = _mahalanobis(arr[t - lookback : t], arr[t]) / n_assets

    valid = scores[np.isfinite(scores)]
    return {
        "turbulence": scores.tolist(),
        "n_assets": int(n_assets),
        "mean_turbulence": float(valid.mean()) if valid.size else float("nan"),
        "max_turbulence": float(valid.max()) if valid.size else float("nan"),
        "current_turbulence": float(scores[-1]),
    }


def turbulence_regimes(
    turbulence: Sequence[float],
    quantile: float = 0.9,
) -> Dict[str, Any]:
    """Flag turbulent periods by quantile threshold.

    Args:
        turbulence: Per-period turbulence scores (NaN entries ignored).
        quantile: Threshold quantile above which a period is flagged
            turbulent (default 0.9).

    Returns:
        Dict with threshold, is_turbulent (bool per period, False for
        NaN), n_turbulent, pct_turbulent, and currently_turbulent.

    Raises:
        ValueError: If quantile is outside (0, 1) or no finite scores.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(turbulence, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("turbulence contains no finite values")

    threshold = float(np.quantile(finite, quantile))
    flags = np.where(np.isfinite(arr), arr > threshold, False)
    n_turb = int(flags.sum())

    return {
        "threshold": threshold,
        "is_turbulent": flags.tolist(),
        "n_turbulent": n_turb,
        "pct_turbulent": float(n_turb) / finite.size,
        "currently_turbulent": bool(flags[-1]),
    }


def absorption_ratio(
    returns: Matrix,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Fraction of total variance absorbed by the top principal components.

    Args:
        returns: (periods, assets) matrix of returns.
        n_components: Number of eigenvectors in the numerator. Default:
            ceil(n_assets / 5) per Kritzman et al. (2011).

    Returns:
        Dict with absorption_ratio, n_components, n_assets, and
        eigenvalue_share (variance share of each component, descending).

    Raises:
        ValueError: If input is malformed or n_components out of range.
    """
    arr = _validate_matrix(returns, min_rows=10)
    n_assets = arr.shape[1]

    if n_components is None:
        n_components = int(np.ceil(n_assets / 5))
    if not 1 <= n_components <= n_assets:
        raise ValueError(f"n_components must be in [1, {n_assets}]")

    cov = np.cov(arr, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("Covariance matrix has non-positive total variance")

    share = eigenvalues / total
    return {
        "absorption_ratio": float(share[:n_components].sum()),
        "n_components": int(n_components),
        "n_assets": int(n_assets),
        "eigenvalue_share": share.tolist(),
    }


def rolling_absorption_ratio(
    returns: Matrix,
    window: int = 60,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Rolling absorption ratio with a shift signal.

    The standardized shift (delta between short-run and long-run AR) is a
    common early-warning signal: a sharp rise in absorption indicates
    increasing market fragility.

    Args:
        returns: (periods, assets) matrix of returns.
        window: Rolling estimation window (default 60).
        n_components: Passed through to absorption_ratio.

    Returns:
        Dict with absorption_ratios (NaN during warmup), window,
        current_ratio, and trend ("rising", "falling", or "stable" based
        on the last window-half change exceeding 0.02).

    Raises:
        ValueError: If the series is shorter than window + 5.
    """
    arr = _validate_matrix(returns, min_rows=window + 5)
    n_periods = arr.shape[0]

    ratios = np.full(n_periods, np.nan)
    for t in range(window, n_periods):
        ratios[t] = absorption_ratio(arr[t - window : t], n_components)["absorption_ratio"]

    valid = ratios[np.isfinite(ratios)]
    half = max(len(valid) // 2, 1)
    delta = float(valid[-1] - np.mean(valid[-half:])) if valid.size else 0.0
    if delta > 0.02:
        trend = "rising"
    elif delta < -0.02:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "absorption_ratios": ratios.tolist(),
        "window": int(window),
        "current_ratio": float(ratios[-1]),
        "trend": trend,
    }
