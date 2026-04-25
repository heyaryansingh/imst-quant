"""Data preprocessing utilities for feature engineering.

Common transformations, normalization, and windowing functions for time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import structlog

logger = structlog.get_logger()


def normalize_returns(
    returns: pd.Series,
    method: str = "z-score",
    window: Optional[int] = None,
) -> pd.Series:
    """Normalize returns using various methods.

    Args:
        returns: Series of returns
        method: Normalization method ('z-score', 'min-max', 'robust')
        window: Rolling window size (None for full history)

    Returns:
        Normalized returns series
    """
    if method == "z-score":
        if window:
            mean = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            return (returns - mean) / std
        else:
            return (returns - returns.mean()) / returns.std()

    elif method == "min-max":
        if window:
            min_val = returns.rolling(window).min()
            max_val = returns.rolling(window).max()
            return (returns - min_val) / (max_val - min_val)
        else:
            return (returns - returns.min()) / (returns.max() - returns.min())

    elif method == "robust":
        # Using median and IQR (less sensitive to outliers)
        if window:
            median = returns.rolling(window).median()
            q75 = returns.rolling(window).quantile(0.75)
            q25 = returns.rolling(window).quantile(0.25)
            iqr = q75 - q25
            return (returns - median) / iqr
        else:
            median = returns.median()
            q75, q25 = returns.quantile(0.75), returns.quantile(0.25)
            return (returns - median) / (q75 - q25)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_time_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 20,
    step_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for time series forecasting.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        window_size: Number of timesteps in each window
        step_size: Step between windows (1 = no skip)

    Returns:
        Tuple of (X, y) arrays with shape (samples, window_size, features)
    """
    X_windows = []
    y_windows = []

    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i : i + window_size]
        target = df.iloc[i + window_size]

        X_windows.append(window[feature_cols].values)
        y_windows.append(target.get("target", target.get("returns", 0)))

    return np.array(X_windows), np.array(y_windows)


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "forward_fill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Handle missing values in time series data.

    Args:
        df: DataFrame with potential missing values
        method: Method to handle NaNs ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if method == "forward_fill":
        return df.ffill(limit=limit)

    elif method == "backward_fill":
        return df.bfill(limit=limit)

    elif method == "interpolate":
        return df.interpolate(method="linear", limit=limit)

    elif method == "drop":
        return df.dropna()

    else:
        raise ValueError(f"Unknown missing value method: {method}")


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Remove or clip outliers from specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: Method to detect outliers ('iqr', 'z-score', 'percentile')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()

    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == "z-score":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]

        elif method == "percentile":
            lower_pct = (100 - threshold) / 2
            upper_pct = 100 - lower_pct
            lower_bound = df[col].quantile(lower_pct / 100)
            upper_bound = df[col].quantile(upper_pct / 100)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def add_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Add lagged features for time series modeling.

    Args:
        df: Input DataFrame with time series data
        columns: Columns to create lags for
        lags: List of lag periods (e.g., [1, 2, 5, 10])

    Returns:
        DataFrame with additional lagged feature columns
    """
    df = df.copy()

    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def compute_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    stats: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """Compute rolling statistics for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to compute rolling stats for
        windows: Window sizes (e.g., [5, 10, 20])
        stats: Statistics to compute ('mean', 'std', 'min', 'max', 'median')

    Returns:
        DataFrame with additional rolling statistic columns
    """
    df = df.copy()

    for col in columns:
        for window in windows:
            for stat in stats:
                col_name = f"{col}_roll{window}_{stat}"

                if stat == "mean":
                    df[col_name] = df[col].rolling(window).mean()
                elif stat == "std":
                    df[col_name] = df[col].rolling(window).std()
                elif stat == "min":
                    df[col_name] = df[col].rolling(window).min()
                elif stat == "max":
                    df[col_name] = df[col].rolling(window).max()
                elif stat == "median":
                    df[col_name] = df[col].rolling(window).median()

    return df


def align_multifrequency_data(
    high_freq_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    method: str = "forward_fill",
) -> pd.DataFrame:
    """Align data from different frequencies (e.g., minute and daily).

    Args:
        high_freq_df: DataFrame with higher frequency data (e.g., minute bars)
        low_freq_df: DataFrame with lower frequency data (e.g., daily)
        method: How to fill low-frequency values ('forward_fill', 'backward_fill')

    Returns:
        DataFrame with both frequencies aligned to high frequency index
    """
    # Merge on index, then fill low-frequency values
    merged = high_freq_df.join(low_freq_df, how="left", rsuffix="_low_freq")

    if method == "forward_fill":
        merged = merged.ffill()
    elif method == "backward_fill":
        merged = merged.bfill()

    return merged


def split_train_val_test(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data into train/validation/test sets.

    Args:
        df: Input DataFrame
        train_pct: Fraction for training (default 70%)
        val_pct: Fraction for validation (default 15%)
        test_pct: Fraction for testing (default 15%)
        shuffle: Whether to shuffle (not recommended for time series)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, \
        "Percentages must sum to 1.0"

    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df
