"""Advanced data windowing utilities for time series analysis.

This module provides flexible windowing operations for creating features,
backtesting, and walk-forward analysis with support for expanding, rolling,
and custom window strategies.

Functions:
    create_rolling_features: Generate rolling window features (mean, std, etc.)
    create_expanding_windows: Create expanding window features for walk-forward
    split_train_test_windows: Split data into train/test with multiple windows
    apply_embargo: Apply embargo periods to prevent lookahead bias
    create_time_based_folds: Create time-based cross-validation folds

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.data_windowing import create_rolling_features
    >>> df = pl.DataFrame({
    ...     "close": [100, 101, 102, 103, 104],
    ...     "volume": [1000, 1100, 1050, 1200, 1150],
    ... })
    >>> df = create_rolling_features(df, ["close", "volume"], window=3)
"""

from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple

import polars as pl


def create_rolling_features(
    df: pl.DataFrame,
    columns: List[str],
    window: int = 20,
    features: Optional[List[str]] = None,
    suffix: Optional[str] = None,
) -> pl.DataFrame:
    """Create rolling window features for specified columns.

    Args:
        df: Input DataFrame.
        columns: List of column names to create features for.
        window: Rolling window size (default: 20).
        features: List of features to create. Options:
            - "mean": Rolling mean
            - "std": Rolling standard deviation
            - "min": Rolling minimum
            - "max": Rolling maximum
            - "median": Rolling median
            - "sum": Rolling sum
            - "skew": Rolling skewness
            - "kurt": Rolling kurtosis
            Default: ["mean", "std"]
        suffix: Suffix for new column names (default: f"_roll{window}")

    Returns:
        DataFrame with added rolling feature columns.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 101, 103, 105]})
        >>> result = create_rolling_features(df, ["price"], window=3)
        >>> print(result.columns)
    """
    if features is None:
        features = ["mean", "std"]

    if suffix is None:
        suffix = f"_roll{window}"

    result_df = df.clone()

    for col in columns:
        for feat in features:
            new_col_name = f"{col}{suffix}_{feat}"

            if feat == "mean":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_mean(window).alias(new_col_name)
                )
            elif feat == "std":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_std(window).alias(new_col_name)
                )
            elif feat == "min":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_min(window).alias(new_col_name)
                )
            elif feat == "max":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_max(window).alias(new_col_name)
                )
            elif feat == "median":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_median(window).alias(new_col_name)
                )
            elif feat == "sum":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_sum(window).alias(new_col_name)
                )
            elif feat == "skew":
                result_df = result_df.with_columns(
                    pl.col(col).rolling_skew(window).alias(new_col_name)
                )
            elif feat == "kurt":
                # Polars doesn't have rolling kurtosis, use manual calculation
                mean_col = pl.col(col).rolling_mean(window)
                std_col = pl.col(col).rolling_std(window)
                result_df = result_df.with_columns(
                    (
                        (pl.col(col) - mean_col).pow(4).rolling_mean(window)
                        / std_col.pow(4)
                        - 3
                    )
                    .fill_null(0)
                    .alias(new_col_name)
                )

    return result_df


def create_expanding_windows(
    df: pl.DataFrame,
    columns: List[str],
    min_periods: int = 20,
    features: Optional[List[str]] = None,
    suffix: str = "_expanding",
) -> pl.DataFrame:
    """Create expanding window features (useful for walk-forward analysis).

    Args:
        df: Input DataFrame.
        columns: List of column names to create features for.
        min_periods: Minimum number of observations required (default: 20).
        features: List of features to create (default: ["mean", "std"]).
        suffix: Suffix for new column names (default: "_expanding").

    Returns:
        DataFrame with added expanding window feature columns.

    Example:
        >>> df = pl.DataFrame({"returns": [0.01, -0.02, 0.03, 0.01, -0.01]})
        >>> result = create_expanding_windows(df, ["returns"], min_periods=2)
    """
    if features is None:
        features = ["mean", "std"]

    result_df = df.clone()

    for col in columns:
        for feat in features:
            new_col_name = f"{col}{suffix}_{feat}"

            if feat == "mean":
                result_df = result_df.with_columns(
                    pl.col(col).cum_sum() / pl.int_range(1, pl.len() + 1).alias(new_col_name)
                )
            elif feat == "std":
                # Expanding std using Welford's online algorithm
                cum_mean = pl.col(col).cum_sum() / pl.int_range(1, pl.len() + 1)
                result_df = result_df.with_columns(
                    pl.col(col)
                    .rolling_std(window_size=pl.len(), min_periods=min_periods)
                    .alias(new_col_name)
                )
            elif feat == "min":
                result_df = result_df.with_columns(pl.col(col).cum_min().alias(new_col_name))
            elif feat == "max":
                result_df = result_df.with_columns(pl.col(col).cum_max().alias(new_col_name))
            elif feat == "sum":
                result_df = result_df.with_columns(pl.col(col).cum_sum().alias(new_col_name))

            # Apply min_periods masking
            if feat in ["mean", "std"]:
                result_df = result_df.with_columns(
                    pl.when(pl.int_range(0, pl.len()) < min_periods)
                    .then(None)
                    .otherwise(pl.col(new_col_name))
                    .alias(new_col_name)
                )

    return result_df


def split_train_test_windows(
    df: pl.DataFrame,
    train_size: float = 0.7,
    test_size: Optional[float] = None,
    n_splits: int = 1,
    gap: int = 0,
    datetime_col: Optional[str] = None,
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """Split data into train/test windows for time series cross-validation.

    Args:
        df: Input DataFrame (must be sorted by time).
        train_size: Fraction of data for training (default: 0.7).
        test_size: Fraction of data for testing (default: 1 - train_size).
        n_splits: Number of train/test splits (default: 1).
        gap: Number of samples to skip between train and test (default: 0).
        datetime_col: Optional datetime column for time-based splitting.

    Returns:
        List of (train_df, test_df) tuples.

    Example:
        >>> df = pl.DataFrame({"value": list(range(100))})
        >>> splits = split_train_test_windows(df, train_size=0.7, n_splits=3)
        >>> len(splits)
        3
    """
    if test_size is None:
        test_size = 1.0 - train_size

    total_size = len(df)
    splits = []

    if n_splits == 1:
        # Single train/test split
        train_end = int(total_size * train_size)
        test_start = train_end + gap
        test_end = min(test_start + int(total_size * test_size), total_size)

        train_df = df.slice(0, train_end)
        test_df = df.slice(test_start, test_end - test_start)
        splits.append((train_df, test_df))

    else:
        # Multiple expanding window splits
        step_size = (total_size - int(total_size * train_size)) // n_splits

        for i in range(n_splits):
            train_end = int(total_size * train_size) + i * step_size
            test_start = train_end + gap
            test_end = min(test_start + step_size, total_size)

            if test_end > test_start:
                train_df = df.slice(0, train_end)
                test_df = df.slice(test_start, test_end - test_start)
                splits.append((train_df, test_df))

    return splits


def apply_embargo(
    df: pl.DataFrame,
    datetime_col: str,
    embargo_period: timedelta,
    event_col: Optional[str] = None,
) -> pl.DataFrame:
    """Apply embargo periods to prevent lookahead bias in backtesting.

    Removes data points within the embargo period after significant events
    (e.g., trades, signals) to prevent information leakage.

    Args:
        df: Input DataFrame with datetime index.
        datetime_col: Name of datetime column.
        embargo_period: Length of embargo period after each event.
        event_col: Column marking events (1 = event, 0 = no event).
            If None, embargo is applied to all rows.

    Returns:
        DataFrame with embargoed rows removed.

    Example:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "date": [datetime(2024, 1, i) for i in range(1, 11)],
        ...     "signal": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ... })
        >>> result = apply_embargo(df, "date", timedelta(days=2), "signal")
    """
    if event_col is None:
        # Apply embargo to all rows
        event_mask = pl.lit(True)
    else:
        event_mask = pl.col(event_col) == 1

    # For each event, mark rows within embargo period
    embargo_mask = pl.lit(False)

    # Get event rows
    event_rows = df.filter(event_mask)

    for event_date in event_rows[datetime_col].to_list():
        embargo_end = event_date + embargo_period
        embargo_mask = embargo_mask | (
            (pl.col(datetime_col) > event_date) & (pl.col(datetime_col) <= embargo_end)
        )

    # Filter out embargoed rows
    return df.filter(~embargo_mask)


def create_time_based_folds(
    df: pl.DataFrame,
    datetime_col: str,
    n_folds: int = 5,
    gap: Optional[timedelta] = None,
    expanding: bool = True,
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """Create time-based cross-validation folds for time series.

    Args:
        df: Input DataFrame with datetime column.
        datetime_col: Name of datetime column.
        n_folds: Number of CV folds (default: 5).
        gap: Optional gap period between train and validation sets.
        expanding: If True, use expanding window (train grows).
            If False, use rolling window (train size fixed).

    Returns:
        List of (train_df, val_df) tuples for each fold.

    Example:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "date": [datetime(2024, 1, i) for i in range(1, 101)],
        ...     "value": list(range(100)),
        ... })
        >>> folds = create_time_based_folds(df, "date", n_folds=3)
        >>> len(folds)
        3
    """
    # Sort by datetime
    df = df.sort(datetime_col)

    total_rows = len(df)
    fold_size = total_rows // (n_folds + 1)

    folds = []

    for i in range(n_folds):
        if expanding:
            # Expanding window: train size grows
            train_end_idx = (i + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = (i + 2) * fold_size

            train_df = df.slice(0, train_end_idx)
        else:
            # Rolling window: train size fixed
            train_start_idx = max(0, (i + 1) * fold_size - fold_size)
            train_end_idx = (i + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = (i + 2) * fold_size

            train_df = df.slice(train_start_idx, train_end_idx - train_start_idx)

        # Apply gap if specified
        if gap is not None:
            train_end_date = train_df[datetime_col].max()
            val_df = df.filter(pl.col(datetime_col) >= train_end_date + gap)
            val_df = val_df.slice(0, val_end_idx - val_start_idx)
        else:
            val_df = df.slice(val_start_idx, val_end_idx - val_start_idx)

        if len(val_df) > 0:
            folds.append((train_df, val_df))

    return folds


def create_lag_features(
    df: pl.DataFrame,
    columns: List[str],
    lags: List[int],
    fill_strategy: Literal["null", "forward", "backward", "zero"] = "null",
) -> pl.DataFrame:
    """Create lagged features for time series prediction.

    Args:
        df: Input DataFrame.
        columns: List of column names to create lags for.
        lags: List of lag periods (e.g., [1, 2, 3, 5, 10]).
        fill_strategy: Strategy for filling missing values at start:
            - "null": Leave as null
            - "forward": Forward fill
            - "backward": Backward fill
            - "zero": Fill with zero

    Returns:
        DataFrame with added lag feature columns.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 101, 103, 105]})
        >>> result = create_lag_features(df, ["price"], lags=[1, 2])
        >>> result.columns
        ['price', 'price_lag1', 'price_lag2']
    """
    result_df = df.clone()

    for col in columns:
        for lag in lags:
            lag_col_name = f"{col}_lag{lag}"
            result_df = result_df.with_columns(pl.col(col).shift(lag).alias(lag_col_name))

            # Apply fill strategy
            if fill_strategy == "forward":
                result_df = result_df.with_columns(
                    pl.col(lag_col_name).fill_null(strategy="forward")
                )
            elif fill_strategy == "backward":
                result_df = result_df.with_columns(
                    pl.col(lag_col_name).fill_null(strategy="backward")
                )
            elif fill_strategy == "zero":
                result_df = result_df.with_columns(pl.col(lag_col_name).fill_null(0))
            # "null" strategy: do nothing

    return result_df


def create_diff_features(
    df: pl.DataFrame,
    columns: List[str],
    periods: List[int] = [1],
    pct_change: bool = False,
) -> pl.DataFrame:
    """Create difference or percentage change features.

    Args:
        df: Input DataFrame.
        columns: List of column names to create diffs for.
        periods: List of difference periods (default: [1]).
        pct_change: If True, create percentage change instead of absolute diff.

    Returns:
        DataFrame with added difference/pct_change columns.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 101, 103, 105]})
        >>> result = create_diff_features(df, ["price"], periods=[1, 2])
        >>> result.columns
        ['price', 'price_diff1', 'price_diff2']
    """
    result_df = df.clone()

    for col in columns:
        for period in periods:
            if pct_change:
                diff_col_name = f"{col}_pct{period}"
                result_df = result_df.with_columns(
                    ((pl.col(col) - pl.col(col).shift(period)) / pl.col(col).shift(period))
                    .alias(diff_col_name)
                )
            else:
                diff_col_name = f"{col}_diff{period}"
                result_df = result_df.with_columns(
                    (pl.col(col) - pl.col(col).shift(period)).alias(diff_col_name)
                )

    return result_df
