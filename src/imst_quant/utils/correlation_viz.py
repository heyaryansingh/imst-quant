"""
Correlation matrix visualization and analysis utilities.

Provides methods to compute, visualize, and analyze correlation structures
in portfolio returns.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def calculate_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate rolling pairwise correlations.

    Args:
        returns: DataFrame of asset returns
        window: Rolling window size in periods
        min_periods: Minimum periods required (default: window)

    Returns:
        DataFrame with rolling correlation for each asset pair
    """
    if min_periods is None:
        min_periods = window

    assets = returns.columns
    n_assets = len(assets)

    # Initialize result DataFrame
    dates = returns.index[window - 1:]
    pairs = [f"{assets[i]}_{assets[j]}"
             for i in range(n_assets)
             for j in range(i + 1, n_assets)]

    rolling_corr = pd.DataFrame(index=dates, columns=pairs)

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if j <= i:
                continue

            pair_name = f"{asset1}_{asset2}"
            rolling_corr[pair_name] = returns[asset1].rolling(
                window=window,
                min_periods=min_periods
            ).corr(returns[asset2])

    return rolling_corr.astype(float)


def hierarchical_cluster_correlation(
    corr_matrix: pd.DataFrame,
    method: str = 'ward',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering on correlation matrix.

    Clusters assets based on correlation structure to identify groups
    of similarly behaving assets.

    Args:
        corr_matrix: Correlation matrix
        method: Linkage method ('ward', 'single', 'complete', 'average')

    Returns:
        Tuple of (linkage_matrix, reordered_indices)
    """
    # Convert correlation to distance (1 - correlation)
    distance = 1 - corr_matrix.abs()

    # Convert to condensed distance matrix
    condensed_dist = squareform(distance, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(condensed_dist, method=method)

    # Get optimal leaf ordering
    reordered_indices = hierarchy.leaves_list(linkage_matrix)

    return linkage_matrix, reordered_indices


def identify_correlation_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> List[List[str]]:
    """
    Identify clusters of highly correlated assets.

    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold for cluster membership

    Returns:
        List of asset clusters (each cluster is a list of asset names)
    """
    clusters = []
    assigned = set()
    assets = corr_matrix.columns.tolist()

    for asset in assets:
        if asset in assigned:
            continue

        # Find all assets correlated above threshold
        correlated = corr_matrix[asset][corr_matrix[asset] >= threshold].index.tolist()
        correlated = [a for a in correlated if a not in assigned]

        if len(correlated) > 1:  # At least 2 assets in cluster
            clusters.append(correlated)
            assigned.update(correlated)

    return clusters


def calculate_correlation_stability(
    returns: pd.DataFrame,
    window: int = 60,
    lookback: int = 250,
) -> pd.DataFrame:
    """
    Calculate correlation stability metrics.

    Measures how stable pairwise correlations are over time.

    Args:
        returns: DataFrame of asset returns
        window: Window for correlation calculation
        lookback: Lookback period for stability analysis

    Returns:
        DataFrame with stability metrics for each asset pair
    """
    rolling_corr = calculate_rolling_correlation(returns, window=window)

    stability_metrics = []

    for pair in rolling_corr.columns:
        recent_corr = rolling_corr[pair].dropna()[-lookback:]

        if len(recent_corr) < 2:
            continue

        metrics = {
            'pair': pair,
            'mean_corr': recent_corr.mean(),
            'std_corr': recent_corr.std(),
            'min_corr': recent_corr.min(),
            'max_corr': recent_corr.max(),
            'range': recent_corr.max() - recent_corr.min(),
        }

        stability_metrics.append(metrics)

    return pd.DataFrame(stability_metrics)


def detect_correlation_regime_changes(
    rolling_corr: pd.DataFrame,
    threshold: float = 0.3,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Detect significant changes in correlation regimes.

    Identifies dates when correlation changes exceed threshold.

    Args:
        rolling_corr: Rolling correlation DataFrame
        threshold: Threshold for significant change

    Returns:
        Dictionary mapping asset pairs to dates of regime changes
    """
    regime_changes = {}

    for pair in rolling_corr.columns:
        corr_series = rolling_corr[pair].dropna()

        # Calculate changes
        changes = corr_series.diff().abs()

        # Identify significant changes
        significant_changes = changes[changes > threshold]

        if len(significant_changes) > 0:
            regime_changes[pair] = significant_changes.index.tolist()

    return regime_changes


def calculate_average_correlation(
    corr_matrix: pd.DataFrame,
    exclude_diagonal: bool = True,
) -> float:
    """
    Calculate average pairwise correlation.

    Args:
        corr_matrix: Correlation matrix
        exclude_diagonal: Whether to exclude diagonal (self-correlation)

    Returns:
        Average correlation
    """
    if exclude_diagonal:
        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(corr_matrix.values, k=1)
        mask = upper_triangle != 0
        correlations = upper_triangle[mask]
    else:
        correlations = corr_matrix.values.flatten()

    return float(np.mean(correlations))


def calculate_correlation_dispersion(
    corr_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate dispersion metrics for correlation matrix.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Dictionary with dispersion metrics
    """
    # Get upper triangle (excluding diagonal)
    upper_triangle = np.triu(corr_matrix.values, k=1)
    mask = upper_triangle != 0
    correlations = upper_triangle[mask]

    return {
        'mean': float(np.mean(correlations)),
        'median': float(np.median(correlations)),
        'std': float(np.std(correlations)),
        'min': float(np.min(correlations)),
        'max': float(np.max(correlations)),
        'q25': float(np.percentile(correlations, 25)),
        'q75': float(np.percentile(correlations, 75)),
        'iqr': float(np.percentile(correlations, 75) - np.percentile(correlations, 25)),
    }


def identify_correlation_outliers(
    corr_matrix: pd.DataFrame,
    n_std: float = 2.0,
) -> List[Tuple[str, str, float]]:
    """
    Identify asset pairs with unusually high or low correlations.

    Args:
        corr_matrix: Correlation matrix
        n_std: Number of standard deviations from mean to consider outlier

    Returns:
        List of (asset1, asset2, correlation) tuples for outliers
    """
    # Calculate mean and std of upper triangle
    upper_triangle = np.triu(corr_matrix.values, k=1)
    mask = upper_triangle != 0
    correlations = upper_triangle[mask]

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    outliers = []
    assets = corr_matrix.columns

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if j <= i:
                continue

            corr = corr_matrix.iloc[i, j]

            # Check if outlier
            if abs(corr - mean_corr) > n_std * std_corr:
                outliers.append((asset1, asset2, corr))

    return outliers


def calculate_tail_correlation(
    returns: pd.DataFrame,
    quantile: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate tail correlation (correlation during extreme events).

    Measures correlation specifically during tail events (extreme losses).

    Args:
        returns: DataFrame of asset returns
        quantile: Quantile threshold for tail events (default 0.05 = 5%)

    Returns:
        Tail correlation matrix
    """
    n_assets = len(returns.columns)
    tail_corr = np.zeros((n_assets, n_assets))

    for i, asset1 in enumerate(returns.columns):
        # Identify tail events for asset1
        threshold = returns[asset1].quantile(quantile)
        tail_mask = returns[asset1] <= threshold

        for j, asset2 in enumerate(returns.columns):
            if i == j:
                tail_corr[i, j] = 1.0
            else:
                # Calculate correlation during tail events
                tail_returns1 = returns.loc[tail_mask, asset1]
                tail_returns2 = returns.loc[tail_mask, asset2]

                if len(tail_returns1) > 2:
                    tail_corr[i, j] = tail_returns1.corr(tail_returns2)
                else:
                    tail_corr[i, j] = np.nan

    return pd.DataFrame(tail_corr, index=returns.columns, columns=returns.columns)


def calculate_dynamic_correlation_score(
    returns: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 60,
) -> pd.DataFrame:
    """
    Calculate dynamic correlation score comparing short vs long term correlations.

    A high score indicates correlations are currently higher than their long-term average.

    Args:
        returns: DataFrame of asset returns
        short_window: Short-term window
        long_window: Long-term window

    Returns:
        DataFrame with dynamic correlation scores
    """
    short_corr = calculate_rolling_correlation(returns, window=short_window)
    long_corr = calculate_rolling_correlation(returns, window=long_window)

    # Align indices
    common_index = short_corr.index.intersection(long_corr.index)
    short_corr = short_corr.loc[common_index]
    long_corr = long_corr.loc[common_index]

    # Calculate score (short-term vs long-term)
    dynamic_score = (short_corr - long_corr) / (long_corr.abs() + 1e-10)

    return dynamic_score


def eigen_decomposition_analysis(
    corr_matrix: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Perform eigenvalue decomposition of correlation matrix.

    Useful for understanding market structure and systemic risk.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Dictionary with eigenvalues, eigenvectors, and explained variance
    """
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix.values)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate explained variance
    total_variance = eigenvalues.sum()
    explained_variance = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance)

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
    }
