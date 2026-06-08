"""Signal decay and staleness detection.

This module analyzes how quickly a trading signal's predictive power
degrades over time (its "half-life"). Signals eventually lose
effectiveness due to alpha decay, crowding, or regime changes.
Detecting staleness early allows timely strategy rotation.

Functions:
    measure_signal_decay: Measure how signal IC decays over forward horizons
    detect_signal_staleness: Detect if a signal has become stale
    rolling_signal_ic: Track signal information coefficient over time
    signal_half_life: Estimate the half-life of a signal's predictive power
    decay_report: Comprehensive signal decay analysis report

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.signal_decay import measure_signal_decay
    >>> signals = pl.Series("signal", [1, -1, 1, 1, -1, 1, -1])
    >>> returns = pl.Series("returns", [0.02, -0.01, 0.015, 0.01, -0.02, 0.005, -0.01])
    >>> decay = measure_signal_decay(signals, returns, horizons=[1, 5, 10, 20])
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class DecayCurve:
    """Decay curve for a signal's predictive power."""

    horizons: List[int]
    ic_values: List[float]
    ic_pvalues: List[float]
    half_life: Optional[float]
    is_significant_at: List[bool]
    decay_rate: float


@dataclass
class StalenessResult:
    """Result of signal staleness detection."""

    is_stale: bool
    current_ic: float
    historical_mean_ic: float
    ic_zscore: float
    consecutive_insignificant: int
    recent_hit_rate: float
    recommendation: str


def _to_numpy(series: Union[pl.Series, np.ndarray]) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(series, pl.Series):
        return series.drop_nulls().to_numpy().astype(np.float64)
    return np.asarray(series, dtype=np.float64)


def measure_signal_decay(
    signals: Union[pl.Series, np.ndarray],
    returns: Union[pl.Series, np.ndarray],
    horizons: Optional[List[int]] = None,
    significance_level: float = 0.05,
) -> DecayCurve:
    """Measure how a signal's information coefficient decays over forward horizons.

    For each horizon h, computes the rank correlation (Spearman IC) between
    the signal at time t and the cumulative return from t to t+h. A decaying
    IC curve shows the signal losing predictive power over longer horizons.

    Args:
        signals: Trading signal values (e.g., -1, 0, 1 or continuous scores).
        returns: Corresponding period returns (aligned with signals).
        horizons: List of forward horizons to test (default: [1, 2, 5, 10, 20]).
        significance_level: Threshold for IC significance (default: 0.05).

    Returns:
        DecayCurve with IC values at each horizon and estimated half-life.

    Example:
        >>> signals = pl.Series([1, -1, 1, 1, -1, 1, -1, 1, -1, 1] * 10)
        >>> returns = pl.Series(np.random.randn(100) * 0.02)
        >>> curve = measure_signal_decay(signals, returns)
    """
    sig = _to_numpy(signals)
    ret = _to_numpy(returns)

    n = min(len(sig), len(ret))
    sig = sig[:n]
    ret = ret[:n]

    if horizons is None:
        horizons = [1, 2, 5, 10, 20]

    ic_values = []
    ic_pvalues = []
    is_significant = []

    for h in horizons:
        if h >= n:
            ic_values.append(0.0)
            ic_pvalues.append(1.0)
            is_significant.append(False)
            continue

        # Forward cumulative returns
        fwd_returns = np.array([
            np.sum(ret[i + 1 : i + 1 + h]) if i + 1 + h <= n else np.nan
            for i in range(n)
        ])

        valid = ~np.isnan(fwd_returns)
        if np.sum(valid) < 10:
            ic_values.append(0.0)
            ic_pvalues.append(1.0)
            is_significant.append(False)
            continue

        # Spearman rank IC
        corr, pval = stats.spearmanr(sig[valid], fwd_returns[valid])
        ic_values.append(float(corr) if not np.isnan(corr) else 0.0)
        ic_pvalues.append(float(pval) if not np.isnan(pval) else 1.0)
        is_significant.append(bool(pval < significance_level))

    # Estimate half-life from IC decay
    half_life = signal_half_life(horizons, ic_values)

    # Decay rate: slope of log(|IC|) vs horizon
    valid_ics = [(h, abs(ic)) for h, ic in zip(horizons, ic_values) if abs(ic) > 0.001]
    if len(valid_ics) >= 2:
        h_arr = np.array([v[0] for v in valid_ics])
        ic_arr = np.array([v[1] for v in valid_ics])
        log_ic = np.log(ic_arr)
        slope, _, _, _, _ = stats.linregress(h_arr, log_ic)
        decay_rate = float(-slope)
    else:
        decay_rate = 0.0

    return DecayCurve(
        horizons=horizons,
        ic_values=ic_values,
        ic_pvalues=ic_pvalues,
        half_life=half_life,
        is_significant_at=is_significant,
        decay_rate=decay_rate,
    )


def signal_half_life(
    horizons: List[int],
    ic_values: List[float],
) -> Optional[float]:
    """Estimate the half-life of signal predictive power from IC decay curve.

    Fits an exponential decay model IC(h) = IC(0) * exp(-lambda * h)
    and computes the half-life as ln(2) / lambda.

    Args:
        horizons: Forward horizons tested.
        ic_values: Information coefficient at each horizon.

    Returns:
        Estimated half-life in periods, or None if estimation fails.

    Example:
        >>> hl = signal_half_life([1, 5, 10, 20], [0.1, 0.07, 0.04, 0.02])
    """
    valid = [(h, abs(ic)) for h, ic in zip(horizons, ic_values) if abs(ic) > 0.001]

    if len(valid) < 2:
        return None

    h_arr = np.array([v[0] for v in valid])
    ic_arr = np.array([v[1] for v in valid])

    # Fit log-linear model: log(IC) = a - lambda * h
    log_ic = np.log(ic_arr)
    slope, intercept, r_value, p_value, std_err = stats.linregress(h_arr, log_ic)

    if slope >= 0:
        # IC is not decaying
        return None

    lam = -slope
    half_life = np.log(2) / lam

    return float(half_life)


def rolling_signal_ic(
    signals: Union[pl.Series, np.ndarray],
    returns: Union[pl.Series, np.ndarray],
    window: int = 60,
    horizon: int = 1,
) -> pl.DataFrame:
    """Compute rolling Information Coefficient of a signal over time.

    Tracks how the signal's predictive power varies through time,
    useful for detecting regime-dependent alpha and staleness.

    Args:
        signals: Signal values (time series).
        returns: Return values (aligned with signals).
        window: Rolling window size (default: 60).
        horizon: Forward return horizon (default: 1).

    Returns:
        DataFrame with columns: index, ic, ic_abs, is_significant.

    Example:
        >>> sigs = pl.Series(np.random.choice([-1, 0, 1], size=200))
        >>> rets = pl.Series(np.random.randn(200) * 0.02)
        >>> ic_df = rolling_signal_ic(sigs, rets, window=60)
    """
    sig = _to_numpy(signals)
    ret = _to_numpy(returns)
    n = min(len(sig), len(ret))
    sig = sig[:n]
    ret = ret[:n]

    # Forward returns
    fwd_ret = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd_ret[i] = np.sum(ret[i + 1 : i + 1 + horizon])

    indices = []
    ics = []
    ic_abs = []
    significance = []

    for end in range(window, n):
        start = end - window
        s_win = sig[start:end]
        r_win = fwd_ret[start:end]

        valid = ~np.isnan(r_win)
        if np.sum(valid) < 10:
            indices.append(end)
            ics.append(0.0)
            ic_abs.append(0.0)
            significance.append(False)
            continue

        corr, pval = stats.spearmanr(s_win[valid], r_win[valid])
        ic_val = float(corr) if not np.isnan(corr) else 0.0
        p_val = float(pval) if not np.isnan(pval) else 1.0

        indices.append(end)
        ics.append(ic_val)
        ic_abs.append(abs(ic_val))
        significance.append(p_val < 0.05)

    return pl.DataFrame({
        "index": indices,
        "ic": ics,
        "ic_abs": ic_abs,
        "is_significant": significance,
    })


def detect_signal_staleness(
    signals: Union[pl.Series, np.ndarray],
    returns: Union[pl.Series, np.ndarray],
    recent_window: int = 60,
    baseline_window: int = 252,
    staleness_zscore: float = -1.5,
    min_hit_rate: float = 0.48,
) -> StalenessResult:
    """Detect if a trading signal has become stale (lost predictive power).

    Compares the signal's recent IC to its historical baseline using a
    z-score test. Also checks hit rate (fraction of correct direction
    predictions) as a secondary indicator.

    A signal is flagged as stale if:
    - Recent IC z-score is below staleness_zscore, OR
    - Recent hit rate is below min_hit_rate, OR
    - Multiple consecutive windows show insignificant IC

    Args:
        signals: Signal values (full history).
        returns: Return values (aligned with signals).
        recent_window: Window for recent IC (default: 60).
        baseline_window: Window for historical baseline IC (default: 252).
        staleness_zscore: Z-score threshold for staleness (default: -1.5).
        min_hit_rate: Minimum acceptable hit rate (default: 0.48).

    Returns:
        StalenessResult with staleness assessment and recommendation.

    Example:
        >>> sigs = pl.Series(np.random.choice([-1, 1], size=300))
        >>> rets = pl.Series(np.random.randn(300) * 0.02)
        >>> result = detect_signal_staleness(sigs, rets)
        >>> print(f"Stale: {result.is_stale}, IC z-score: {result.ic_zscore:.2f}")
    """
    sig = _to_numpy(signals)
    ret = _to_numpy(returns)
    n = min(len(sig), len(ret))
    sig = sig[:n]
    ret = ret[:n]

    if n < recent_window + 10:
        return StalenessResult(
            is_stale=False,
            current_ic=0.0,
            historical_mean_ic=0.0,
            ic_zscore=0.0,
            consecutive_insignificant=0,
            recent_hit_rate=0.5,
            recommendation="Insufficient data for staleness detection.",
        )

    # Rolling IC for full history
    ic_df = rolling_signal_ic(sig, ret, window=recent_window, horizon=1)

    if ic_df.height == 0:
        return StalenessResult(
            is_stale=False,
            current_ic=0.0,
            historical_mean_ic=0.0,
            ic_zscore=0.0,
            consecutive_insignificant=0,
            recent_hit_rate=0.5,
            recommendation="Unable to compute rolling IC.",
        )

    ic_series = ic_df["ic"].to_numpy()

    # Current IC (most recent window)
    current_ic = float(ic_series[-1]) if len(ic_series) > 0 else 0.0

    # Historical baseline
    baseline_end = max(0, len(ic_series) - 1)
    baseline_start = max(0, baseline_end - baseline_window)
    baseline_ics = ic_series[baseline_start:baseline_end]

    if len(baseline_ics) > 5:
        hist_mean = float(np.mean(baseline_ics))
        hist_std = float(np.std(baseline_ics, ddof=1))
        ic_zscore = (current_ic - hist_mean) / hist_std if hist_std > 0 else 0.0
    else:
        hist_mean = current_ic
        hist_std = 0.0
        ic_zscore = 0.0

    # Consecutive insignificant windows
    sig_flags = ic_df["is_significant"].to_list()
    consec_insig = 0
    for flag in reversed(sig_flags):
        if not flag:
            consec_insig += 1
        else:
            break

    # Hit rate: fraction of correct direction predictions
    recent_sig = sig[n - recent_window : n]
    recent_ret = ret[n - recent_window : n]
    valid_mask = recent_sig != 0
    if np.sum(valid_mask) > 0:
        hits = np.sum(np.sign(recent_sig[valid_mask]) == np.sign(recent_ret[valid_mask]))
        hit_rate = float(hits / np.sum(valid_mask))
    else:
        hit_rate = 0.5

    # Staleness decision
    is_stale = (
        ic_zscore < staleness_zscore
        or hit_rate < min_hit_rate
        or consec_insig >= 5
    )

    # Recommendation
    if is_stale:
        reasons = []
        if ic_zscore < staleness_zscore:
            reasons.append(f"IC dropped {ic_zscore:.1f} std below historical mean")
        if hit_rate < min_hit_rate:
            reasons.append(f"hit rate ({hit_rate:.1%}) below threshold ({min_hit_rate:.1%})")
        if consec_insig >= 5:
            reasons.append(f"{consec_insig} consecutive windows with insignificant IC")
        recommendation = f"Signal appears stale: {'; '.join(reasons)}. Consider reducing weight or rotating to alternative signals."
    else:
        recommendation = "Signal is performing within historical norms."

    return StalenessResult(
        is_stale=is_stale,
        current_ic=current_ic,
        historical_mean_ic=hist_mean,
        ic_zscore=ic_zscore,
        consecutive_insignificant=consec_insig,
        recent_hit_rate=hit_rate,
        recommendation=recommendation,
    )


def decay_report(
    signals: Union[pl.Series, np.ndarray],
    returns: Union[pl.Series, np.ndarray],
    horizons: Optional[List[int]] = None,
    window: int = 60,
) -> Dict:
    """Generate comprehensive signal decay analysis report.

    Combines decay curve analysis, staleness detection, and rolling IC
    into a single report.

    Args:
        signals: Signal values.
        returns: Return values (aligned).
        horizons: Forward horizons to test (default: [1, 2, 5, 10, 20]).
        window: Rolling window for IC and staleness (default: 60).

    Returns:
        Dictionary with:
        - decay_curve: DecayCurve analysis
        - staleness: StalenessResult
        - rolling_ic_summary: Summary stats of rolling IC
        - recommendation: Overall recommendation

    Example:
        >>> report = decay_report(signals, returns)
        >>> print(report["recommendation"])
    """
    curve = measure_signal_decay(signals, returns, horizons)
    staleness = detect_signal_staleness(signals, returns, recent_window=window)
    ic_df = rolling_signal_ic(signals, returns, window=window)

    ic_arr = ic_df["ic"].to_numpy() if ic_df.height > 0 else np.array([0.0])

    rolling_summary = {
        "mean_ic": float(np.mean(ic_arr)),
        "std_ic": float(np.std(ic_arr)),
        "min_ic": float(np.min(ic_arr)),
        "max_ic": float(np.max(ic_arr)),
        "pct_significant": float(np.mean(ic_df["is_significant"].to_numpy())) if ic_df.height > 0 else 0.0,
    }

    # Overall recommendation
    if staleness.is_stale and curve.half_life is not None and curve.half_life < 5:
        recommendation = "CRITICAL: Signal has short half-life and is currently stale. Immediate rotation recommended."
    elif staleness.is_stale:
        recommendation = "WARNING: Signal shows staleness. Monitor closely and prepare alternative signals."
    elif curve.half_life is not None and curve.half_life < 3:
        recommendation = "CAUTION: Signal has very short predictive half-life. Only suitable for short-horizon strategies."
    else:
        recommendation = "Signal performance is within acceptable bounds."

    return {
        "decay_curve": curve,
        "staleness": staleness,
        "rolling_ic_summary": rolling_summary,
        "recommendation": recommendation,
    }
