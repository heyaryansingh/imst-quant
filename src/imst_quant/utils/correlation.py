"""Correlation analysis utilities for asset and feature relationships.

This module provides tools for analyzing correlations between assets,
features, and market factors to support portfolio construction and
risk management decisions.

Functions:
    calculate_asset_correlation: Compute pairwise asset correlations
    calculate_rolling_correlation: Time-varying correlation analysis
    correlation_heatmap_data: Prepare data for correlation visualization
    identify_correlation_regimes: Detect correlation regime changes
    feature_importance_correlation: Analyze feature-target correlations

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.correlation import calculate_asset_correlation
    >>> returns = pl.read_parquet("data/gold/features.parquet")
    >>> corr_matrix = calculate_asset_correlation(returns, return_col="return_1d")
    >>> print(corr_matrix)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def calculate_asset_correlation(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    asset_col: str = "ticker",
    date_col: str = "date",
    method: str = "pearson",
    min_observations: int = 30,
) -> pl.DataFrame:
    """Calculate pairwise correlation matrix between assets.

    Computes correlation coefficients between all asset pairs based on
    their return time series. Useful for portfolio diversification analysis.

    Args:
        df: DataFrame with columns for date, asset, and returns.
        return_col: Column name containing return values.
        asset_col: Column name for asset identifiers.
        date_col: Column name for dates.
        method: Correlation method ('pearson' or 'spearman'). Defaults to 'pearson'.
        min_observations: Minimum overlapping observations required.

    Returns:
        DataFrame with correlation matrix (assets as both rows and columns).

    Example:
        >>> df = pl.DataFrame({
        ...     "date": ["2024-01-01"] * 2 + ["2024-01-02"] * 2,
        ...     "ticker": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
        ...     "return_1d": [0.01, 0.02, -0.01, -0.015]
        ... })
        >>> corr = calculate_asset_correlation(df)
    """
    # Pivot to wide format (dates as rows, assets as columns)
    pivot_df = df.pivot(
        values=return_col,
        index=date_col,
        on=asset_col,
    )

    # Get asset columns (exclude date)
    assets = [c for c in pivot_df.columns if c != date_col]

    # Calculate correlation matrix
    corr_data: Dict[str, List[float]] = {"asset": assets}

    for asset1 in assets:
        correlations = []
        for asset2 in assets:
            if asset1 == asset2:
                correlations.append(1.0)
            else:
                # Get overlapping non-null values
                mask = pivot_df[asset1].is_not_null() & pivot_df[asset2].is_not_null()
                filtered = pivot_df.filter(mask)

                if filtered.height < min_observations:
                    correlations.append(float("nan"))
                else:
                    s1 = filtered[asset1]
                    s2 = filtered[asset2]

                    if method == "spearman":
                        # Rank-based correlation
                        r1 = s1.rank()
                        r2 = s2.rank()
                        corr = _pearson_corr(r1, r2)
                    else:
                        corr = _pearson_corr(s1, s2)

                    correlations.append(corr)

        corr_data[asset1] = correlations

    return pl.DataFrame(corr_data)


def _pearson_corr(s1: pl.Series, s2: pl.Series) -> float:
    """Calculate Pearson correlation between two series."""
    n = s1.len()
    if n == 0:
        return float("nan")

    mean1 = s1.mean()
    mean2 = s2.mean()

    if mean1 is None or mean2 is None:
        return float("nan")

    std1 = s1.std()
    std2 = s2.std()

    if std1 is None or std2 is None or std1 == 0 or std2 == 0:
        return float("nan")

    # Covariance
    diff1 = s1 - mean1
    diff2 = s2 - mean2
    cov = (diff1 * diff2).sum() / (n - 1)

    return float(cov / (std1 * std2))


def calculate_rolling_correlation(
    df: pl.DataFrame,
    asset1: str,
    asset2: str,
    return_col: str = "return_1d",
    asset_col: str = "ticker",
    date_col: str = "date",
    window: int = 60,
) -> pl.DataFrame:
    """Calculate rolling correlation between two assets over time.

    Computes time-varying correlation using a rolling window, useful for
    identifying correlation regime changes.

    Args:
        df: DataFrame with time series return data.
        asset1: First asset identifier.
        asset2: Second asset identifier.
        return_col: Column name for returns.
        asset_col: Column name for asset identifiers.
        date_col: Column name for dates.
        window: Rolling window size in periods. Defaults to 60.

    Returns:
        DataFrame with date and rolling_correlation columns.

    Example:
        >>> corr_ts = calculate_rolling_correlation(df, "AAPL", "GOOGL", window=30)
        >>> print(corr_ts.tail())
    """
    # Pivot data
    pivot_df = df.pivot(
        values=return_col,
        index=date_col,
        on=asset_col,
    ).sort(date_col)

    if asset1 not in pivot_df.columns or asset2 not in pivot_df.columns:
        return pl.DataFrame({date_col: [], "rolling_correlation": []})

    # Calculate rolling correlation
    s1 = pivot_df[asset1]
    s2 = pivot_df[asset2]

    correlations = []
    dates = pivot_df[date_col].to_list()

    for i in range(len(dates)):
        if i < window - 1:
            correlations.append(None)
        else:
            window_s1 = s1.slice(i - window + 1, window)
            window_s2 = s2.slice(i - window + 1, window)

            # Remove null pairs
            valid_mask = window_s1.is_not_null() & window_s2.is_not_null()
            valid_s1 = window_s1.filter(valid_mask)
            valid_s2 = window_s2.filter(valid_mask)

            if valid_s1.len() < window // 2:
                correlations.append(None)
            else:
                correlations.append(_pearson_corr(valid_s1, valid_s2))

    return pl.DataFrame({
        date_col: dates,
        "rolling_correlation": correlations,
    })


def correlation_heatmap_data(
    corr_matrix: pl.DataFrame,
    asset_col: str = "asset",
) -> Dict[str, Any]:
    """Prepare correlation matrix for heatmap visualization.

    Converts the correlation DataFrame into a format suitable for
    plotting libraries (matplotlib, plotly, etc.).

    Args:
        corr_matrix: Correlation DataFrame from calculate_asset_correlation().
        asset_col: Column name containing asset identifiers.

    Returns:
        Dictionary with:
        - labels: Asset names for axes
        - values: 2D list of correlation values
        - min_corr: Minimum correlation value
        - max_corr: Maximum correlation value (excluding diagonal)

    Example:
        >>> heatmap = correlation_heatmap_data(corr_matrix)
        >>> print(heatmap["labels"])
    """
    assets = corr_matrix[asset_col].to_list()
    values = []

    min_corr = 1.0
    max_off_diag = -1.0

    for i, asset in enumerate(assets):
        row = corr_matrix[asset].to_list()
        values.append(row)

        for j, val in enumerate(row):
            if not np.isnan(val):
                if i != j:  # Off-diagonal
                    min_corr = min(min_corr, val)
                    max_off_diag = max(max_off_diag, val)

    return {
        "labels": assets,
        "values": values,
        "min_corr": min_corr,
        "max_corr": max_off_diag,
        "shape": (len(assets), len(assets)),
    }


def identify_correlation_regimes(
    rolling_corr: pl.DataFrame,
    date_col: str = "date",
    corr_col: str = "rolling_correlation",
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
) -> pl.DataFrame:
    """Identify correlation regime changes over time.

    Classifies periods into high, medium, and low correlation regimes
    and detects regime transitions.

    Args:
        rolling_corr: DataFrame with rolling correlation time series.
        date_col: Column name for dates.
        corr_col: Column name for correlation values.
        high_threshold: Threshold for high correlation regime. Defaults to 0.7.
        low_threshold: Threshold for low correlation regime. Defaults to 0.3.

    Returns:
        DataFrame with date, correlation, regime, and regime_change columns.

    Example:
        >>> regimes = identify_correlation_regimes(rolling_corr)
        >>> print(regimes.filter(pl.col("regime_change")))
    """
    df = rolling_corr.with_columns(
        pl.when(pl.col(corr_col).abs() >= high_threshold)
        .then(pl.lit("high"))
        .when(pl.col(corr_col).abs() <= low_threshold)
        .then(pl.lit("low"))
        .otherwise(pl.lit("medium"))
        .alias("regime")
    )

    # Detect regime changes
    df = df.with_columns(
        (pl.col("regime") != pl.col("regime").shift(1)).alias("regime_change")
    )

    return df


def feature_importance_correlation(
    df: pl.DataFrame,
    target_col: str = "return_1d",
    feature_cols: Optional[List[str]] = None,
    method: str = "pearson",
) -> pl.DataFrame:
    """Analyze correlations between features and target variable.

    Calculates correlation between each feature and the target,
    useful for feature selection and understanding predictive relationships.

    Args:
        df: DataFrame with features and target column.
        target_col: Column name for the target variable.
        feature_cols: List of feature columns. If None, uses all numeric columns.
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        DataFrame with feature names, correlations, and absolute correlations
        sorted by importance (highest absolute correlation first).

    Example:
        >>> importance = feature_importance_correlation(df, target_col="return_1d")
        >>> print(importance.head(10))  # Top 10 correlated features
    """
    if feature_cols is None:
        # Auto-detect numeric columns, excluding target
        feature_cols = [
            c for c in df.columns
            if c != target_col and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    target = df[target_col].drop_nulls()
    results: List[Dict[str, Any]] = []

    for col in feature_cols:
        feature = df[col]

        # Get overlapping non-null values
        mask = feature.is_not_null() & df[target_col].is_not_null()
        filtered_df = df.filter(mask)

        if filtered_df.height < 30:
            continue

        f_series = filtered_df[col]
        t_series = filtered_df[target_col]

        if method == "spearman":
            corr = _pearson_corr(f_series.rank(), t_series.rank())
        else:
            corr = _pearson_corr(f_series, t_series)

        if not np.isnan(corr):
            results.append({
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
            })

    return pl.DataFrame(results).sort("abs_correlation", descending=True)


def calculate_correlation_summary(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    asset_col: str = "ticker",
    date_col: str = "date",
) -> Dict[str, Any]:
    """Generate comprehensive correlation analysis summary.

    Combines multiple correlation analyses into a single summary report.

    Args:
        df: DataFrame with return data.
        return_col: Column name for returns.
        asset_col: Column name for asset identifiers.
        date_col: Column name for dates.

    Returns:
        Dictionary with:
        - avg_correlation: Average pairwise correlation
        - min_correlation: Minimum correlation pair
        - max_correlation: Maximum correlation pair (excluding self)
        - correlation_matrix: Full correlation matrix
        - highly_correlated: Pairs with correlation > 0.7
        - negatively_correlated: Pairs with correlation < -0.3

    Example:
        >>> summary = calculate_correlation_summary(df)
        >>> print(f"Average correlation: {summary['avg_correlation']:.3f}")
    """
    corr_matrix = calculate_asset_correlation(
        df, return_col=return_col, asset_col=asset_col, date_col=date_col
    )

    assets = corr_matrix["asset"].to_list()
    pairs: List[Tuple[str, str, float]] = []

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:  # Upper triangle only
                corr = corr_matrix[asset1][j]
                if not np.isnan(corr):
                    pairs.append((asset1, asset2, corr))

    if not pairs:
        return {
            "avg_correlation": float("nan"),
            "min_correlation": None,
            "max_correlation": None,
            "correlation_matrix": corr_matrix,
            "highly_correlated": [],
            "negatively_correlated": [],
        }

    correlations = [p[2] for p in pairs]
    avg_corr = sum(correlations) / len(correlations)

    min_pair = min(pairs, key=lambda x: x[2])
    max_pair = max(pairs, key=lambda x: x[2])

    highly_correlated = [(p[0], p[1], p[2]) for p in pairs if p[2] > 0.7]
    negatively_correlated = [(p[0], p[1], p[2]) for p in pairs if p[2] < -0.3]

    return {
        "avg_correlation": avg_corr,
        "min_correlation": {"pair": (min_pair[0], min_pair[1]), "value": min_pair[2]},
        "max_correlation": {"pair": (max_pair[0], max_pair[1]), "value": max_pair[2]},
        "correlation_matrix": corr_matrix,
        "highly_correlated": highly_correlated,
        "negatively_correlated": negatively_correlated,
        "total_pairs": len(pairs),
    }
