"""Data cleaning and preprocessing utilities for trading data.

This module provides functions for cleaning, validating, and preprocessing
market data, social media data, and trade records.

Functions:
    remove_outliers: Detect and remove statistical outliers
    fill_missing_data: Handle missing values with various strategies
    normalize_timestamps: Standardize timestamp formats and timezones
    validate_price_data: Check for data quality issues in OHLCV data
    resample_data: Resample time series to different frequencies
"""

from typing import Dict, List, Optional, Tuple, Literal
import polars as pl
from datetime import datetime, timezone


def remove_outliers(
    df: pl.DataFrame,
    column: str,
    method: Literal["iqr", "zscore", "percentile"] = "iqr",
    threshold: float = 3.0,
) -> pl.DataFrame:
    """Remove statistical outliers from a column.

    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        method: Detection method:
            - 'iqr': Interquartile range method (default)
            - 'zscore': Z-score method
            - 'percentile': Remove extreme percentiles
        threshold: Threshold for outlier detection:
            - For IQR: multiple of IQR (default: 3.0)
            - For Z-score: number of standard deviations (default: 3.0)
            - For percentile: percentile to clip (default: 3.0 means 3rd/97th)

    Returns:
        DataFrame with outliers removed
    """
    if df.is_empty() or column not in df.columns:
        return df

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return df.filter(
            (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
        )

    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()

        return df.filter(
            (pl.col(column) >= mean - threshold * std) &
            (pl.col(column) <= mean + threshold * std)
        )

    elif method == "percentile":
        lower_pct = threshold
        upper_pct = 100 - threshold

        lower_bound = df[column].quantile(lower_pct / 100)
        upper_bound = df[column].quantile(upper_pct / 100)

        return df.filter(
            (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
        )

    return df


def fill_missing_data(
    df: pl.DataFrame,
    column: str,
    method: Literal["forward", "backward", "linear", "mean", "median"] = "forward",
) -> pl.DataFrame:
    """Fill missing values in a column using various strategies.

    Args:
        df: Input DataFrame
        column: Column name to fill
        method: Filling strategy:
            - 'forward': Forward fill (use last valid value)
            - 'backward': Backward fill (use next valid value)
            - 'linear': Linear interpolation
            - 'mean': Fill with column mean
            - 'median': Fill with column median

    Returns:
        DataFrame with missing values filled
    """
    if df.is_empty() or column not in df.columns:
        return df

    if method == "forward":
        return df.with_columns(pl.col(column).forward_fill().alias(column))

    elif method == "backward":
        return df.with_columns(pl.col(column).backward_fill().alias(column))

    elif method == "linear":
        return df.with_columns(pl.col(column).interpolate().alias(column))

    elif method == "mean":
        mean_value = df[column].mean()
        return df.with_columns(pl.col(column).fill_null(mean_value).alias(column))

    elif method == "median":
        median_value = df[column].median()
        return df.with_columns(pl.col(column).fill_null(median_value).alias(column))

    return df


def normalize_timestamps(
    df: pl.DataFrame,
    timestamp_col: str,
    target_tz: str = "UTC",
) -> pl.DataFrame:
    """Normalize timestamps to a standard timezone.

    Args:
        df: Input DataFrame
        timestamp_col: Column name containing timestamps
        target_tz: Target timezone (default: 'UTC')

    Returns:
        DataFrame with normalized timestamps
    """
    if df.is_empty() or timestamp_col not in df.columns:
        return df

    # Convert to datetime if needed
    if df[timestamp_col].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col(timestamp_col).str.to_datetime().alias(timestamp_col)
        )

    # Convert to target timezone
    df = df.with_columns(
        pl.col(timestamp_col).dt.convert_time_zone(target_tz).alias(timestamp_col)
    )

    return df


def validate_price_data(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """Validate OHLCV price data for common quality issues.

    Checks for:
    - High < Low violations
    - Open/Close outside High/Low range
    - Negative prices
    - Zero or negative volume
    - Duplicate timestamps

    Args:
        df: Input DataFrame with OHLCV data
        open_col: Column name for opening price
        high_col: Column name for high price
        low_col: Column name for low price
        close_col: Column name for closing price
        volume_col: Column name for volume

    Returns:
        Tuple of (cleaned DataFrame, issues dict with counts)
    """
    if df.is_empty():
        return df, {}

    issues = {
        "high_low_violations": 0,
        "open_out_of_range": 0,
        "close_out_of_range": 0,
        "negative_prices": 0,
        "invalid_volume": 0,
    }

    # Check high < low
    high_low_valid = df.filter(pl.col(high_col) >= pl.col(low_col))
    issues["high_low_violations"] = len(df) - len(high_low_valid)
    df = high_low_valid

    # Check open in range
    open_valid = df.filter(
        (pl.col(open_col) >= pl.col(low_col)) &
        (pl.col(open_col) <= pl.col(high_col))
    )
    issues["open_out_of_range"] = len(df) - len(open_valid)
    df = open_valid

    # Check close in range
    close_valid = df.filter(
        (pl.col(close_col) >= pl.col(low_col)) &
        (pl.col(close_col) <= pl.col(high_col))
    )
    issues["close_out_of_range"] = len(df) - len(close_valid)
    df = close_valid

    # Check for negative prices
    price_cols = [open_col, high_col, low_col, close_col]
    for col in price_cols:
        before = len(df)
        df = df.filter(pl.col(col) > 0)
        issues["negative_prices"] += before - len(df)

    # Check volume
    if volume_col in df.columns:
        before = len(df)
        df = df.filter(pl.col(volume_col) > 0)
        issues["invalid_volume"] = before - len(df)

    return df, issues


def resample_data(
    df: pl.DataFrame,
    timestamp_col: str,
    frequency: str,
    agg_funcs: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Resample time series data to a different frequency.

    Args:
        df: Input DataFrame
        timestamp_col: Column name for timestamps
        frequency: Target frequency (e.g., '1h', '1d', '1w')
        agg_funcs: Dictionary mapping column names to aggregation functions
            (e.g., {'close': 'last', 'volume': 'sum'})
            If None, uses default OHLCV aggregations

    Returns:
        Resampled DataFrame
    """
    if df.is_empty() or timestamp_col not in df.columns:
        return df

    # Default aggregations for OHLCV data
    if agg_funcs is None:
        agg_funcs = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

    # Sort by timestamp
    df = df.sort(timestamp_col)

    # Group by frequency and aggregate
    aggregations = []
    for col, func in agg_funcs.items():
        if col not in df.columns:
            continue

        if func == "first":
            aggregations.append(pl.col(col).first().alias(col))
        elif func == "last":
            aggregations.append(pl.col(col).last().alias(col))
        elif func == "max":
            aggregations.append(pl.col(col).max().alias(col))
        elif func == "min":
            aggregations.append(pl.col(col).min().alias(col))
        elif func == "sum":
            aggregations.append(pl.col(col).sum().alias(col))
        elif func == "mean":
            aggregations.append(pl.col(col).mean().alias(col))

    resampled = (
        df
        .group_by_dynamic(timestamp_col, every=frequency)
        .agg(aggregations)
    )

    return resampled


def detect_gaps(
    df: pl.DataFrame,
    timestamp_col: str,
    expected_frequency: str = "1d",
) -> pl.DataFrame:
    """Detect gaps in time series data.

    Args:
        df: Input DataFrame sorted by timestamp
        timestamp_col: Column name for timestamps
        expected_frequency: Expected frequency between records (e.g., '1d', '1h')

    Returns:
        DataFrame with detected gaps showing start_time, end_time, and duration
    """
    if df.is_empty() or timestamp_col not in df.columns:
        return pl.DataFrame()

    # Sort by timestamp
    df_sorted = df.sort(timestamp_col)

    # Calculate time differences
    df_with_diff = df_sorted.with_columns(
        (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1)).alias("time_diff")
    )

    # Parse expected frequency to timedelta
    freq_map = {
        "1m": 60,
        "5m": 300,
        "1h": 3600,
        "1d": 86400,
        "1w": 604800,
    }

    expected_seconds = freq_map.get(expected_frequency, 86400)

    # Find gaps (where time_diff > expected)
    gaps = df_with_diff.filter(
        pl.col("time_diff").dt.total_seconds() > expected_seconds * 1.5
    )

    if gaps.is_empty():
        return pl.DataFrame()

    # Create gap report
    gap_report = gaps.select([
        pl.col(timestamp_col).shift(1).alias("gap_start"),
        pl.col(timestamp_col).alias("gap_end"),
        pl.col("time_diff").alias("gap_duration"),
    ])

    return gap_report


def remove_duplicates(
    df: pl.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> Tuple[pl.DataFrame, int]:
    """Remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates (None = all columns)
        keep: Which duplicate to keep ('first' or 'last')

    Returns:
        Tuple of (deduplicated DataFrame, number of duplicates removed)
    """
    if df.is_empty():
        return df, 0

    original_len = len(df)

    if subset is None:
        deduplicated = df.unique(keep=keep)
    else:
        deduplicated = df.unique(subset=subset, keep=keep)

    duplicates_removed = original_len - len(deduplicated)

    return deduplicated, duplicates_removed
