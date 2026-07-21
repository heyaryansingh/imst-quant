"""Change-point detection for return and volatility regime breaks.

Two complementary detectors:

- CUSUM mean-shift detection (Page, 1954): a two-sided cumulative sum on
  standardized returns that raises an alarm when the drift-adjusted sum
  exceeds a threshold. Good for catching persistent shifts in mean return.

- ICSS variance breaks (Inclan & Tiao, 1994): the centered cumulative sum
  of squares statistic D_k, applied with binary segmentation, locates
  points where return variance changes. Good for splitting a history into
  homogeneous volatility regimes.
"""

import numpy as np
from typing import Any, Dict, List, Sequence

Vector = Sequence[float]

# Asymptotic 95% critical value for sqrt(T/2) * max|D_k| (Inclan & Tiao, 1994)
_ICSS_CRITICAL_95 = 1.358


def _validate_returns(returns: Vector, min_obs: int) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if arr.size < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values")
    return arr


def cusum_mean_shift(
    returns: Vector,
    threshold: float = 8.0,
    drift: float = 0.5,
) -> Dict[str, Any]:
    """Detect mean shifts in a return series with a two-sided CUSUM.

    Returns are standardized by the full-sample mean and std, then the
    upper and lower cumulative sums

        S+_t = max(0, S+_{t-1} + z_t - drift)
        S-_t = max(0, S-_{t-1} - z_t - drift)

    raise an alarm whenever either exceeds ``threshold``. Both sums reset
    to zero after an alarm so successive shifts are each reported once.

    Args:
        returns: 1D return series.
        threshold: Alarm level in standard deviations of cumulated drift.
            Larger values mean fewer, more confident alarms.
        drift: Allowance subtracted each step; shifts smaller than about
            2 * drift standard deviations go undetected by design.

    Returns:
        Dict with alarms (list of {index, direction}), n_alarms,
        n_observations, threshold, and drift.

    Raises:
        ValueError: If input is malformed or parameters are non-positive.
    """
    arr = _validate_returns(returns, min_obs=20)
    if threshold <= 0 or drift <= 0:
        raise ValueError("threshold and drift must be positive")

    std = arr.std(ddof=1)
    if std < 1e-12:
        raise ValueError("returns has zero variance")
    z = (arr - arr.mean()) / std

    alarms: List[Dict[str, Any]] = []
    s_pos = 0.0
    s_neg = 0.0
    for t, zt in enumerate(z):
        s_pos = max(0.0, s_pos + zt - drift)
        s_neg = max(0.0, s_neg - zt - drift)
        if s_pos > threshold or s_neg > threshold:
            alarms.append(
                {"index": t, "direction": "up" if s_pos > threshold else "down"}
            )
            s_pos = 0.0
            s_neg = 0.0

    return {
        "alarms": alarms,
        "n_alarms": len(alarms),
        "n_observations": int(arr.size),
        "threshold": threshold,
        "drift": drift,
    }


def _max_dk(sq: np.ndarray) -> tuple:
    """Return (k*, sqrt(T/2)*|D_k*|) for the centered cumsum-of-squares."""
    total = sq.sum()
    if total <= 0:
        return 0, 0.0
    t = sq.size
    ck = np.cumsum(sq)
    k = np.arange(1, t + 1)
    dk = ck / total - k / t
    idx = int(np.argmax(np.abs(dk[:-1]))) if t > 1 else 0
    stat = float(np.sqrt(t / 2.0) * abs(dk[idx]))
    return idx, stat


def icss_variance_breaks(
    returns: Vector,
    min_segment: int = 30,
    critical_value: float = _ICSS_CRITICAL_95,
) -> Dict[str, Any]:
    """Locate variance change points via ICSS with binary segmentation.

    Demeaned squared returns are scanned for the maximizer of the centered
    cumulative-sum-of-squares statistic D_k; when sqrt(T/2)*|D_k| exceeds
    the critical value the segment is split and both halves are scanned
    recursively, until no segment shows a significant break or segments
    fall below ``min_segment`` observations.

    Args:
        returns: 1D return series.
        min_segment: Smallest segment length that will be scanned or split.
        critical_value: Rejection level for sqrt(T/2)*max|D_k|; the default
            1.358 is the asymptotic 95% value.

    Returns:
        Dict with break_indices (sorted global indices), n_breaks, and
        segments (list of {start, end, volatility} with end exclusive,
        volatility = per-period std of that segment).

    Raises:
        ValueError: If input is malformed or min_segment is too small.
    """
    arr = _validate_returns(returns, min_obs=2 * 30)
    if min_segment < 10:
        raise ValueError("min_segment must be at least 10")

    sq = (arr - arr.mean()) ** 2
    breaks: List[int] = []

    def _scan(start: int, end: int) -> None:
        if end - start < 2 * min_segment:
            return
        idx, stat = _max_dk(sq[start:end])
        cp = start + idx + 1  # first index of the new regime
        if stat > critical_value and min_segment <= cp - start and end - cp >= min_segment:
            breaks.append(cp)
            _scan(start, cp)
            _scan(cp, end)

    _scan(0, arr.size)
    breaks.sort()

    edges = [0] + breaks + [int(arr.size)]
    segments = [
        {
            "start": s,
            "end": e,
            "volatility": float(arr[s:e].std(ddof=1)),
        }
        for s, e in zip(edges[:-1], edges[1:])
    ]

    return {
        "break_indices": breaks,
        "n_breaks": len(breaks),
        "segments": segments,
    }


def analyze_change_points(
    returns: Vector,
    cusum_threshold: float = 8.0,
    cusum_drift: float = 0.5,
    min_segment: int = 30,
) -> Dict[str, Any]:
    """Run both detectors and summarize regime stability.

    Args:
        returns: 1D return series.
        cusum_threshold: Alarm level for the mean-shift CUSUM.
        cusum_drift: Drift allowance for the mean-shift CUSUM.
        min_segment: Minimum segment length for variance segmentation.

    Returns:
        Dict with mean_shift (cusum_mean_shift output), variance
        (icss_variance_breaks output), n_observations, and stability
        ("stable" when neither detector fires, else "unstable").
    """
    mean_shift = cusum_mean_shift(returns, threshold=cusum_threshold, drift=cusum_drift)
    variance = icss_variance_breaks(returns, min_segment=min_segment)
    unstable = mean_shift["n_alarms"] > 0 or variance["n_breaks"] > 0
    return {
        "mean_shift": mean_shift,
        "variance": variance,
        "n_observations": mean_shift["n_observations"],
        "stability": "unstable" if unstable else "stable",
    }
