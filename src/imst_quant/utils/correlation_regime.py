"""Correlation regime change detection and monitoring.

This module detects structural breaks in asset correlation matrices over
time. During market crises, correlations tend to spike toward 1.0
("correlation breakdown"), destroying diversification benefits exactly
when they are most needed. This module provides tools to detect, measure,
and alert on such regime changes.

Functions:
    rolling_correlation_matrix: Compute rolling pairwise correlations
    detect_correlation_regime: Classify current correlation regime
    correlation_divergence: Measure divergence from baseline correlation
    correlation_stability: Assess stability of the correlation structure
    eigenvalue_concentration: Track risk concentration via PCA eigenvalues

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.correlation_regime import detect_correlation_regime
    >>> returns = pl.DataFrame({
    ...     "AAPL": [0.01, -0.02, 0.015],
    ...     "MSFT": [0.012, -0.018, 0.013],
    ...     "GOOG": [0.008, -0.025, 0.011],
    ... })
    >>> regime = detect_correlation_regime(returns, window=60)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


class CorrelationRegime(str, Enum):
    """Identified correlation regime states."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"
    DECORRELATED = "decorrelated"


@dataclass
class CorrelationRegimeResult:
    """Result of correlation regime detection."""

    regime: CorrelationRegime
    mean_correlation: float
    median_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_dispersion: float
    eigenvalue_concentration: float
    divergence_from_baseline: float
    assets_analyzed: int
    window_size: int


@dataclass
class CorrelationStability:
    """Stability metrics for the correlation structure."""

    frobenius_distance: float
    max_pairwise_change: float
    mean_absolute_change: float
    unstable_pairs: List[Tuple[str, str, float]]
    is_stable: bool


def rolling_correlation_matrix(
    returns: pl.DataFrame,
    window: int = 60,
    step: int = 1,
) -> List[Dict]:
    """Compute rolling pairwise correlation matrices.

    Calculates correlation matrices over a rolling window, returning
    a time series of correlation snapshots for regime analysis.

    Args:
        returns: DataFrame with one column per asset, each row is a return.
        window: Rolling window size in periods (default: 60 trading days).
        step: Step size between windows (default: 1).

    Returns:
        List of dicts, each containing:
        - index: Window end index
        - matrix: Correlation matrix as 2D numpy array
        - mean_corr: Mean off-diagonal correlation
        - assets: List of asset names

    Example:
        >>> df = pl.DataFrame({"A": [0.01]*100, "B": [0.02]*100})
        >>> matrices = rolling_correlation_matrix(df, window=20)
    """
    assets = returns.columns
    n_assets = len(assets)
    arr = returns.to_numpy().astype(np.float64)
    n_rows = arr.shape[0]

    results = []
    for end in range(window, n_rows + 1, step):
        start = end - window
        window_data = arr[start:end]

        # Compute correlation matrix
        corr = np.corrcoef(window_data, rowvar=False)
        # Handle NaN from constant columns
        corr = np.nan_to_num(corr, nan=0.0)

        # Mean off-diagonal correlation
        mask = ~np.eye(n_assets, dtype=bool)
        mean_corr = float(np.mean(np.abs(corr[mask]))) if n_assets > 1 else 0.0

        results.append(
            {
                "index": end - 1,
                "matrix": corr,
                "mean_corr": mean_corr,
                "assets": assets,
            }
        )

    return results


def detect_correlation_regime(
    returns: pl.DataFrame,
    window: int = 60,
    baseline_window: int = 252,
    crisis_threshold: float = 0.7,
    elevated_threshold: float = 0.5,
    decorrelated_threshold: float = 0.15,
) -> CorrelationRegimeResult:
    """Detect the current correlation regime from recent returns.

    Classifies the current regime by comparing the recent rolling
    correlation against historical baseline thresholds.

    Regimes:
    - CRISIS: Mean |correlation| >= crisis_threshold (diversification failure)
    - ELEVATED: Mean |correlation| >= elevated_threshold
    - NORMAL: Mean |correlation| between decorrelated and elevated thresholds
    - DECORRELATED: Mean |correlation| < decorrelated_threshold

    Args:
        returns: DataFrame with one column per asset.
        window: Recent window for current correlation (default: 60).
        baseline_window: Longer window for baseline comparison (default: 252).
        crisis_threshold: Threshold for crisis regime (default: 0.7).
        elevated_threshold: Threshold for elevated regime (default: 0.5).
        decorrelated_threshold: Threshold for decorrelated regime (default: 0.15).

    Returns:
        CorrelationRegimeResult with regime classification and metrics.

    Example:
        >>> df = pl.DataFrame({"A": np.random.randn(300).tolist(),
        ...                    "B": np.random.randn(300).tolist()})
        >>> result = detect_correlation_regime(df)
        >>> print(f"Regime: {result.regime.value}")
    """
    assets = returns.columns
    n_assets = len(assets)
    arr = returns.to_numpy().astype(np.float64)
    n_rows = arr.shape[0]

    # Use available data, but at least window size
    recent_start = max(0, n_rows - window)
    recent_data = arr[recent_start:]

    if recent_data.shape[0] < 3 or n_assets < 2:
        return CorrelationRegimeResult(
            regime=CorrelationRegime.NORMAL,
            mean_correlation=0.0,
            median_correlation=0.0,
            max_correlation=0.0,
            min_correlation=0.0,
            correlation_dispersion=0.0,
            eigenvalue_concentration=0.0,
            divergence_from_baseline=0.0,
            assets_analyzed=n_assets,
            window_size=window,
        )

    # Current correlation matrix
    corr = np.corrcoef(recent_data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    mask = ~np.eye(n_assets, dtype=bool)
    off_diag = np.abs(corr[mask])

    mean_corr = float(np.mean(off_diag))
    median_corr = float(np.median(off_diag))
    max_corr = float(np.max(off_diag))
    min_corr = float(np.min(off_diag))
    dispersion = float(np.std(off_diag))

    # Eigenvalue concentration (how much risk is in first PC)
    eigen_conc = eigenvalue_concentration(recent_data)

    # Baseline divergence
    baseline_start = max(0, n_rows - baseline_window)
    baseline_data = arr[baseline_start:recent_start] if recent_start > baseline_start else arr[:recent_start]
    if baseline_data.shape[0] >= 3:
        baseline_corr = np.corrcoef(baseline_data, rowvar=False)
        baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)
        divergence = float(np.sqrt(np.mean((corr - baseline_corr) ** 2)))
    else:
        divergence = 0.0

    # Classify regime
    if mean_corr >= crisis_threshold:
        regime = CorrelationRegime.CRISIS
    elif mean_corr >= elevated_threshold:
        regime = CorrelationRegime.ELEVATED
    elif mean_corr < decorrelated_threshold:
        regime = CorrelationRegime.DECORRELATED
    else:
        regime = CorrelationRegime.NORMAL

    return CorrelationRegimeResult(
        regime=regime,
        mean_correlation=mean_corr,
        median_correlation=median_corr,
        max_correlation=max_corr,
        min_correlation=min_corr,
        correlation_dispersion=dispersion,
        eigenvalue_concentration=eigen_conc,
        divergence_from_baseline=divergence,
        assets_analyzed=n_assets,
        window_size=window,
    )


def correlation_divergence(
    current_corr: np.ndarray,
    baseline_corr: np.ndarray,
) -> Dict[str, float]:
    """Measure divergence between two correlation matrices.

    Computes multiple distance metrics between a current and baseline
    correlation matrix to quantify structural changes.

    Args:
        current_corr: Current correlation matrix (n x n).
        baseline_corr: Baseline correlation matrix (n x n).

    Returns:
        Dictionary with distance metrics:
        - frobenius: Frobenius norm of the difference
        - max_change: Maximum absolute element-wise change
        - mean_change: Mean absolute element-wise change
        - spectral: Largest eigenvalue of the difference matrix

    Example:
        >>> curr = np.array([[1, 0.8], [0.8, 1]])
        >>> base = np.array([[1, 0.3], [0.3, 1]])
        >>> div = correlation_divergence(curr, base)
        >>> print(f"Frobenius distance: {div['frobenius']:.4f}")
    """
    diff = current_corr - baseline_corr

    frobenius = float(np.linalg.norm(diff, "fro"))
    max_change = float(np.max(np.abs(diff)))
    mean_change = float(np.mean(np.abs(diff)))

    eigenvalues = np.linalg.eigvalsh(diff)
    spectral = float(np.max(np.abs(eigenvalues)))

    return {
        "frobenius": frobenius,
        "max_change": max_change,
        "mean_change": mean_change,
        "spectral": spectral,
    }


def correlation_stability(
    returns: pl.DataFrame,
    window: int = 60,
    change_threshold: float = 0.3,
) -> CorrelationStability:
    """Assess the stability of the correlation structure over two consecutive windows.

    Compares the most recent correlation window to the immediately
    preceding window to detect sudden structural shifts.

    Args:
        returns: DataFrame with one column per asset.
        window: Window size for each correlation estimate (default: 60).
        change_threshold: Threshold for flagging unstable pairs (default: 0.3).

    Returns:
        CorrelationStability with distance metrics and flagged pairs.

    Example:
        >>> df = pl.DataFrame({"A": np.random.randn(150).tolist(),
        ...                    "B": np.random.randn(150).tolist()})
        >>> stability = correlation_stability(df, window=60)
        >>> print(f"Stable: {stability.is_stable}")
    """
    assets = returns.columns
    arr = returns.to_numpy().astype(np.float64)
    n_rows = arr.shape[0]
    n_assets = len(assets)

    if n_rows < 2 * window or n_assets < 2:
        return CorrelationStability(
            frobenius_distance=0.0,
            max_pairwise_change=0.0,
            mean_absolute_change=0.0,
            unstable_pairs=[],
            is_stable=True,
        )

    # Recent window
    recent = arr[n_rows - window :]
    corr_recent = np.corrcoef(recent, rowvar=False)
    corr_recent = np.nan_to_num(corr_recent, nan=0.0)

    # Previous window
    prev = arr[n_rows - 2 * window : n_rows - window]
    corr_prev = np.corrcoef(prev, rowvar=False)
    corr_prev = np.nan_to_num(corr_prev, nan=0.0)

    div = correlation_divergence(corr_recent, corr_prev)

    # Find unstable pairs
    diff = np.abs(corr_recent - corr_prev)
    unstable = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if diff[i, j] >= change_threshold:
                unstable.append((assets[i], assets[j], float(diff[i, j])))

    unstable.sort(key=lambda x: x[2], reverse=True)

    is_stable = len(unstable) == 0

    return CorrelationStability(
        frobenius_distance=div["frobenius"],
        max_pairwise_change=div["max_change"],
        mean_absolute_change=div["mean_change"],
        unstable_pairs=unstable,
        is_stable=is_stable,
    )


def eigenvalue_concentration(
    data: np.ndarray,
) -> float:
    """Calculate eigenvalue concentration ratio (first PC explained variance).

    A high concentration means portfolio risk is dominated by a single
    factor (typically market beta), indicating poor diversification.
    During crises, this ratio typically spikes.

    Args:
        data: 2D numpy array of returns (rows=time, cols=assets).

    Returns:
        Fraction of total variance explained by the first principal component
        (0 to 1). Higher means more concentrated risk.

    Example:
        >>> data = np.random.randn(100, 5)
        >>> conc = eigenvalue_concentration(data)
        >>> print(f"PC1 explains {conc:.1%} of variance")
    """
    if data.shape[0] < 3 or data.shape[1] < 2:
        return 0.0

    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = eigenvalues[::-1]  # Sort descending

    total = np.sum(np.maximum(eigenvalues, 0))
    if total == 0:
        return 0.0

    return float(np.maximum(eigenvalues[0], 0) / total)
