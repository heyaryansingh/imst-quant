"""Position correlation analysis utilities.

This module provides utilities for analyzing correlations between portfolio
positions to identify concentration risks, diversification quality, and
potential hedging opportunities.

Functions:
    calculate_position_correlation_matrix: Build correlation matrix for positions
    identify_clustered_positions: Find groups of highly correlated positions
    calculate_effective_positions: Measure true diversification accounting for correlation
    analyze_correlation_stability: Measure how stable correlations are over time
    detect_correlation_regime_changes: Identify periods of correlation regime shifts

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.position_correlation import (
    ...     calculate_position_correlation_matrix,
    ...     identify_clustered_positions
    ... )
    >>> returns_df = pl.DataFrame({...})  # Multi-asset returns
    >>> corr_matrix = calculate_position_correlation_matrix(returns_df)
    >>> clusters = identify_clustered_positions(corr_matrix)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class CorrelationCluster:
    """Represents a cluster of correlated positions.

    Attributes:
        assets: List of asset identifiers in this cluster
        avg_correlation: Average pairwise correlation within cluster
        size: Number of assets in cluster
        correlation_matrix: Subset correlation matrix for cluster
    """

    assets: List[str]
    avg_correlation: float
    size: int
    correlation_matrix: np.ndarray


def calculate_position_correlation_matrix(
    returns: pl.DataFrame,
    asset_columns: Optional[List[str]] = None,
    method: str = "pearson",
    min_periods: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """Calculate correlation matrix for portfolio positions.

    Computes pairwise correlations between asset returns to assess
    diversification and identify concentration risks.

    Args:
        returns: DataFrame with returns for each asset (columns = assets)
        asset_columns: List of column names to include. If None, uses all numeric columns.
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
        min_periods: Minimum number of observations required for valid correlation

    Returns:
        Tuple of (correlation_matrix, asset_names)
        - correlation_matrix: NxN correlation matrix
        - asset_names: List of asset names corresponding to matrix rows/cols

    Example:
        >>> df = pl.DataFrame({
        ...     "AAPL": [0.01, 0.02, -0.01],
        ...     "MSFT": [0.015, 0.018, -0.008],
        ...     "BTC": [-0.02, 0.05, 0.03]
        ... })
        >>> corr_matrix, names = calculate_position_correlation_matrix(df)
    """
    if asset_columns is None:
        # Use all numeric columns
        asset_columns = [
            col for col in returns.columns if returns[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    if len(asset_columns) < 2:
        raise ValueError("Need at least 2 assets to calculate correlations")

    # Extract return columns and convert to numpy
    returns_subset = returns.select(asset_columns).to_numpy()

    # Check minimum periods
    valid_rows = ~np.isnan(returns_subset).any(axis=1)
    if valid_rows.sum() < min_periods:
        raise ValueError(
            f"Insufficient data: {valid_rows.sum()} periods available, "
            f"{min_periods} required"
        )

    # Calculate correlation matrix
    if method == "pearson":
        corr_matrix = np.corrcoef(returns_subset, rowvar=False)
    elif method == "spearman":
        from scipy.stats import spearmanr

        corr_matrix, _ = spearmanr(returns_subset, axis=0, nan_policy="omit")
    elif method == "kendall":
        # Kendall's tau (more expensive)
        n_assets = len(asset_columns)
        corr_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                from scipy.stats import kendalltau

                tau, _ = kendalltau(
                    returns_subset[:, i],
                    returns_subset[:, j],
                    nan_policy="omit",
                )
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr_matrix, asset_columns


def identify_clustered_positions(
    correlation_matrix: np.ndarray,
    asset_names: List[str],
    threshold: float = 0.7,
) -> List[CorrelationCluster]:
    """Identify clusters of highly correlated positions.

    Groups positions that move together, indicating concentration risk
    or potential redundancy in the portfolio.

    Args:
        correlation_matrix: NxN correlation matrix
        asset_names: List of asset names corresponding to matrix
        threshold: Minimum correlation to consider assets in same cluster

    Returns:
        List of CorrelationCluster objects, sorted by average correlation

    Example:
        >>> corr = np.array([[1.0, 0.85, 0.2], [0.85, 1.0, 0.15], [0.2, 0.15, 1.0]])
        >>> names = ["AAPL", "MSFT", "BTC"]
        >>> clusters = identify_clustered_positions(corr, names, threshold=0.7)
    """
    n_assets = len(asset_names)
    if correlation_matrix.shape != (n_assets, n_assets):
        raise ValueError("Correlation matrix shape doesn't match number of assets")

    # Track which assets have been assigned to clusters
    assigned = set()
    clusters = []

    for i in range(n_assets):
        if i in assigned:
            continue

        # Find all assets correlated with asset i above threshold
        cluster_members = [i]
        assigned.add(i)

        for j in range(i + 1, n_assets):
            if j in assigned:
                continue

            if abs(correlation_matrix[i, j]) >= threshold:
                cluster_members.append(j)
                assigned.add(j)

        # Only create cluster if it has more than 1 member
        if len(cluster_members) > 1:
            cluster_assets = [asset_names[idx] for idx in cluster_members]
            cluster_corr_matrix = correlation_matrix[np.ix_(cluster_members, cluster_members)]

            # Calculate average pairwise correlation (excluding diagonal)
            n = len(cluster_members)
            avg_corr = (cluster_corr_matrix.sum() - n) / (n * (n - 1))

            cluster = CorrelationCluster(
                assets=cluster_assets,
                avg_correlation=float(avg_corr),
                size=len(cluster_members),
                correlation_matrix=cluster_corr_matrix,
            )
            clusters.append(cluster)

    # Sort by average correlation (highest first)
    clusters.sort(key=lambda c: c.avg_correlation, reverse=True)

    return clusters


def calculate_effective_positions(
    correlation_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate effective number of positions accounting for correlations.

    Measures true diversification by adjusting for correlation structure.
    Equal to number of positions if all uncorrelated, lower if correlated.

    Args:
        correlation_matrix: NxN correlation matrix
        weights: Optional position weights (default: equal-weighted)

    Returns:
        Effective number of positions (1 to N)

    Formula:
        Effective N = 1 / sum(w_i^2 * sum(w_j * corr_ij))

    Example:
        >>> corr = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.15], [0.2, 0.15, 1.0]])
        >>> effective_n = calculate_effective_positions(corr)
    """
    n = correlation_matrix.shape[0]

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights)
        if len(weights) != n:
            raise ValueError("Weights must match correlation matrix size")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()  # Normalize

    # Calculate portfolio variance contribution
    variance_contrib = np.zeros(n)
    for i in range(n):
        variance_contrib[i] = weights[i] ** 2 + sum(
            weights[i] * weights[j] * correlation_matrix[i, j]
            for j in range(n)
            if j != i
        )

    # Effective number of positions
    effective_n = 1.0 / variance_contrib.sum()

    return float(effective_n)


def analyze_correlation_stability(
    returns: pl.DataFrame,
    asset_columns: List[str],
    window_size: int = 60,
    step_size: int = 20,
) -> Dict[str, float]:
    """Analyze stability of correlations over time.

    Measures how much correlations change across rolling windows,
    indicating regime stability.

    Args:
        returns: DataFrame with asset returns
        asset_columns: List of assets to analyze
        window_size: Size of rolling window for correlation calculation
        step_size: Step size between windows

    Returns:
        Dictionary with:
        - avg_correlation_stability: Average correlation across time
        - correlation_volatility: Std dev of rolling correlations
        - max_correlation_change: Largest correlation shift observed
        - stability_score: 0-1 score (1 = perfectly stable)

    Example:
        >>> df = pl.DataFrame({...})  # Time series of returns
        >>> stability = analyze_correlation_stability(df, ["AAPL", "MSFT"])
    """
    n_rows = returns.height

    if n_rows < window_size:
        raise ValueError(f"Need at least {window_size} periods")

    # Calculate rolling correlations
    window_starts = range(0, n_rows - window_size + 1, step_size)
    correlations_over_time = []

    for start in window_starts:
        window_data = returns[start : start + window_size]
        try:
            corr_matrix, _ = calculate_position_correlation_matrix(
                window_data,
                asset_columns=asset_columns,
                min_periods=window_size // 2,
            )
            # Extract upper triangle (unique correlations)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = float(np.mean(upper_tri))
            correlations_over_time.append(avg_corr)
        except ValueError:
            # Skip windows with insufficient data
            continue

    if len(correlations_over_time) < 2:
        return {
            "avg_correlation_stability": 0.0,
            "correlation_volatility": 0.0,
            "max_correlation_change": 0.0,
            "stability_score": 0.0,
        }

    corr_series = np.array(correlations_over_time)

    # Calculate metrics
    avg_stability = float(np.mean(corr_series))
    corr_volatility = float(np.std(corr_series))

    # Max change between consecutive windows
    changes = np.abs(np.diff(corr_series))
    max_change = float(np.max(changes)) if len(changes) > 0 else 0.0

    # Stability score: inverse of coefficient of variation
    cv = corr_volatility / abs(avg_stability) if avg_stability != 0 else float("inf")
    stability_score = 1.0 / (1.0 + cv)

    return {
        "avg_correlation_stability": avg_stability,
        "correlation_volatility": corr_volatility,
        "max_correlation_change": max_change,
        "stability_score": stability_score,
    }


def detect_correlation_regime_changes(
    returns: pl.DataFrame,
    asset_columns: List[str],
    window_size: int = 60,
    threshold: float = 0.3,
) -> List[Dict[str, any]]:
    """Detect periods where correlation regime significantly changes.

    Identifies timestamps where portfolio correlations shift dramatically,
    potentially indicating market stress or regime transitions.

    Args:
        returns: DataFrame with asset returns and date column
        asset_columns: List of assets to analyze
        window_size: Size of rolling window
        threshold: Minimum correlation change to flag as regime shift

    Returns:
        List of regime change events with:
        - period_start: Start index of new regime
        - correlation_before: Average correlation before shift
        - correlation_after: Average correlation after shift
        - magnitude: Size of correlation change

    Example:
        >>> df = pl.DataFrame({...})
        >>> regime_changes = detect_correlation_regime_changes(
        ...     df, ["AAPL", "MSFT", "GOOGL"]
        ... )
    """
    n_rows = returns.height

    if n_rows < window_size * 2:
        raise ValueError(f"Need at least {window_size * 2} periods")

    regime_changes = []

    # Rolling correlation analysis
    for i in range(window_size, n_rows - window_size, window_size // 2):
        window_before = returns[i - window_size : i]
        window_after = returns[i : i + window_size]

        try:
            corr_before, _ = calculate_position_correlation_matrix(
                window_before,
                asset_columns=asset_columns,
                min_periods=window_size // 2,
            )
            corr_after, _ = calculate_position_correlation_matrix(
                window_after,
                asset_columns=asset_columns,
                min_periods=window_size // 2,
            )

            # Average correlation (upper triangle)
            avg_before = float(
                np.mean(corr_before[np.triu_indices_from(corr_before, k=1)])
            )
            avg_after = float(
                np.mean(corr_after[np.triu_indices_from(corr_after, k=1)])
            )

            magnitude = abs(avg_after - avg_before)

            if magnitude >= threshold:
                regime_changes.append(
                    {
                        "period_start": i,
                        "correlation_before": avg_before,
                        "correlation_after": avg_after,
                        "magnitude": magnitude,
                        "direction": "increase" if avg_after > avg_before else "decrease",
                    }
                )

        except ValueError:
            # Skip if insufficient data
            continue

    return regime_changes


def correlation_heatmap_data(
    correlation_matrix: np.ndarray,
    asset_names: List[str],
) -> pl.DataFrame:
    """Convert correlation matrix to long-format DataFrame for visualization.

    Args:
        correlation_matrix: NxN correlation matrix
        asset_names: List of asset names

    Returns:
        DataFrame with columns: asset_1, asset_2, correlation

    Example:
        >>> corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        >>> names = ["AAPL", "MSFT"]
        >>> heatmap_df = correlation_heatmap_data(corr, names)
    """
    n = len(asset_names)
    data = []

    for i in range(n):
        for j in range(n):
            data.append(
                {
                    "asset_1": asset_names[i],
                    "asset_2": asset_names[j],
                    "correlation": float(correlation_matrix[i, j]),
                }
            )

    return pl.DataFrame(data)
