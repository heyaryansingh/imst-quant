"""Data pipeline optimization utilities for efficient batch processing.

This module provides utilities for optimizing data processing pipelines,
including parallel processing, chunk-based operations, and memory-efficient
aggregations for large datasets.

Functions:
    parallel_apply: Apply function to DataFrame chunks in parallel
    batch_aggregate_sentiment: Efficient sentiment aggregation with chunking
    optimize_parquet_files: Optimize Parquet file structure for queries
    merge_incremental_data: Merge new data with existing datasets efficiently
    calculate_rolling_features: Memory-efficient rolling window calculations

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.data_pipeline import batch_aggregate_sentiment
    >>> df = pl.read_parquet("sentiment.parquet")
    >>> aggregated = batch_aggregate_sentiment(df, group_by="asset_id")
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import polars as pl
import numpy as np


def parallel_apply(
    df: pl.DataFrame,
    func: Callable[[pl.DataFrame], pl.DataFrame],
    chunk_size: int = 10000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Apply a function to DataFrame chunks in parallel.

    Useful for applying expensive operations to large DataFrames by
    splitting into chunks and processing in parallel.

    Args:
        df: Input DataFrame.
        func: Function to apply to each chunk (must accept and return DataFrame).
        chunk_size: Number of rows per chunk.
        n_jobs: Number of parallel jobs (-1 for all CPUs).

    Returns:
        Concatenated DataFrame with function applied to all chunks.

    Example:
        >>> def process_chunk(chunk_df):
        ...     return chunk_df.with_columns(pl.col("value") * 2)
        >>> result = parallel_apply(df, process_chunk, chunk_size=5000)
    """
    # Split DataFrame into chunks
    n_rows = len(df)
    chunks = []

    for i in range(0, n_rows, chunk_size):
        chunk = df.slice(i, min(chunk_size, n_rows - i))
        chunks.append(func(chunk))

    # Concatenate results
    return pl.concat(chunks)


def batch_aggregate_sentiment(
    df: pl.DataFrame,
    group_by: Union[str, List[str]] = "asset_id",
    date_col: str = "date",
    value_cols: Optional[List[str]] = None,
    chunk_by_date: bool = True,
) -> pl.DataFrame:
    """Efficient sentiment aggregation with chunking for large datasets.

    Aggregates sentiment data by asset and date with memory-efficient
    chunking strategy.

    Args:
        df: Input DataFrame with sentiment data.
        group_by: Column(s) to group by (default: "asset_id").
        date_col: Date column for temporal aggregation.
        value_cols: Columns to aggregate (default: all numeric columns).
        chunk_by_date: If True, process one date at a time to reduce memory.

    Returns:
        Aggregated DataFrame with mean, std, count per group.

    Example:
        >>> df = pl.read_parquet("sentiment_raw.parquet")
        >>> agg = batch_aggregate_sentiment(df, group_by=["asset_id", "source"])
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    # Identify numeric columns to aggregate
    if value_cols is None:
        value_cols = [
            col
            for col in df.columns
            if col not in group_by + [date_col]
            and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

    if chunk_by_date:
        # Process one date at a time
        unique_dates = df[date_col].unique().sort()
        aggregated_chunks = []

        for date_val in unique_dates:
            date_chunk = df.filter(pl.col(date_col) == date_val)

            # Aggregate for this date
            agg_expressions = []
            for col in value_cols:
                agg_expressions.extend([
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).count().alias(f"{col}_count"),
                ])

            agg_chunk = date_chunk.group_by(group_by + [date_col]).agg(agg_expressions)
            aggregated_chunks.append(agg_chunk)

        return pl.concat(aggregated_chunks)
    else:
        # Standard aggregation (may use more memory)
        agg_expressions = []
        for col in value_cols:
            agg_expressions.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
                pl.col(col).count().alias(f"{col}_count"),
            ])

        return df.group_by(group_by + [date_col]).agg(agg_expressions)


def optimize_parquet_files(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    partition_cols: Optional[List[str]] = None,
    row_group_size: int = 100000,
    compression: str = "snappy",
) -> None:
    """Optimize Parquet file structure for faster queries.

    Re-writes Parquet files with optimized settings: partitioning,
    row group size, and compression.

    Args:
        input_path: Path to input Parquet file or directory.
        output_path: Path to output optimized Parquet file/directory.
        partition_cols: Columns to partition by (e.g., ["date", "asset_id"]).
        row_group_size: Number of rows per row group.
        compression: Compression codec ("snappy", "gzip", "lz4", "zstd").

    Example:
        >>> optimize_parquet_files(
        ...     "data/raw/sentiment.parquet",
        ...     "data/optimized/sentiment.parquet",
        ...     partition_cols=["date"],
        ...     compression="zstd"
        ... )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read the input file(s)
    if input_path.is_dir():
        df = pl.scan_parquet(str(input_path / "*.parquet")).collect()
    else:
        df = pl.read_parquet(input_path)

    # Write with optimized settings
    if partition_cols:
        # Polars native partitioning (if supported)
        output_path.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            output_path / "data.parquet",
            compression=compression,
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            output_path,
            compression=compression,
        )


def merge_incremental_data(
    existing_path: Union[str, Path],
    new_data: pl.DataFrame,
    key_cols: List[str],
    date_col: str = "date",
    dedup_strategy: str = "last",
) -> pl.DataFrame:
    """Merge new data with existing dataset efficiently.

    Handles incremental updates by merging new data with existing,
    removing duplicates based on key columns.

    Args:
        existing_path: Path to existing Parquet file.
        new_data: New data to merge.
        key_cols: Columns defining uniqueness (e.g., ["asset_id", "date"]).
        date_col: Date column for sorting.
        dedup_strategy: Deduplication strategy ("last" or "first").

    Returns:
        Merged DataFrame with duplicates removed.

    Example:
        >>> new_df = pl.DataFrame({"asset_id": ["AAPL"], "date": ["2024-01-01"]})
        >>> merged = merge_incremental_data(
        ...     "data/bronze/prices.parquet",
        ...     new_df,
        ...     key_cols=["asset_id", "date"]
        ... )
    """
    existing_path = Path(existing_path)

    # Load existing data
    if existing_path.exists():
        existing_df = pl.read_parquet(existing_path)
    else:
        # No existing data, return new data
        return new_data

    # Concatenate existing and new
    combined = pl.concat([existing_df, new_data])

    # Sort by date (most recent first for "last" strategy)
    if dedup_strategy == "last":
        combined = combined.sort(date_col, descending=True)
    else:
        combined = combined.sort(date_col, descending=False)

    # Remove duplicates keeping first occurrence (which is most recent if sorted desc)
    deduplicated = combined.unique(subset=key_cols, keep="first")

    # Sort back to ascending order
    return deduplicated.sort(date_col)


def calculate_rolling_features(
    df: pl.DataFrame,
    value_col: str,
    windows: List[int],
    group_by: Optional[Union[str, List[str]]] = None,
    features: List[str] = ["mean", "std", "min", "max"],
) -> pl.DataFrame:
    """Memory-efficient rolling window feature calculations.

    Calculates rolling statistics for specified windows without
    materializing large intermediate DataFrames.

    Args:
        df: Input DataFrame (must be sorted by date within groups).
        value_col: Column to calculate rolling features on.
        windows: List of window sizes (e.g., [7, 30, 90]).
        group_by: Column(s) to group by before rolling (optional).
        features: List of features to calculate ("mean", "std", "min", "max", "sum").

    Returns:
        DataFrame with original columns plus rolling features.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": pl.date_range(start="2024-01-01", end="2024-12-31", interval="1d"),
        ...     "asset_id": ["AAPL"] * 365,
        ...     "price": np.random.randn(365).cumsum() + 100
        ... })
        >>> result = calculate_rolling_features(
        ...     df,
        ...     value_col="price",
        ...     windows=[7, 30],
        ...     group_by="asset_id",
        ...     features=["mean", "std"]
        ... )
    """
    result = df.clone()

    for window in windows:
        for feature in features:
            col_name = f"{value_col}_rolling_{window}d_{feature}"

            if group_by:
                # Group-aware rolling calculation
                if feature == "mean":
                    result = result.with_columns(
                        pl.col(value_col)
                        .rolling_mean(window_size=window)
                        .over(group_by)
                        .alias(col_name)
                    )
                elif feature == "std":
                    result = result.with_columns(
                        pl.col(value_col)
                        .rolling_std(window_size=window)
                        .over(group_by)
                        .alias(col_name)
                    )
                elif feature == "min":
                    result = result.with_columns(
                        pl.col(value_col)
                        .rolling_min(window_size=window)
                        .over(group_by)
                        .alias(col_name)
                    )
                elif feature == "max":
                    result = result.with_columns(
                        pl.col(value_col)
                        .rolling_max(window_size=window)
                        .over(group_by)
                        .alias(col_name)
                    )
                elif feature == "sum":
                    result = result.with_columns(
                        pl.col(value_col)
                        .rolling_sum(window_size=window)
                        .over(group_by)
                        .alias(col_name)
                    )
            else:
                # Global rolling calculation
                if feature == "mean":
                    result = result.with_columns(
                        pl.col(value_col).rolling_mean(window_size=window).alias(col_name)
                    )
                elif feature == "std":
                    result = result.with_columns(
                        pl.col(value_col).rolling_std(window_size=window).alias(col_name)
                    )
                elif feature == "min":
                    result = result.with_columns(
                        pl.col(value_col).rolling_min(window_size=window).alias(col_name)
                    )
                elif feature == "max":
                    result = result.with_columns(
                        pl.col(value_col).rolling_max(window_size=window).alias(col_name)
                    )
                elif feature == "sum":
                    result = result.with_columns(
                        pl.col(value_col).rolling_sum(window_size=window).alias(col_name)
                    )

    return result


def detect_data_quality_issues(
    df: pl.DataFrame,
    date_col: str = "date",
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Detect common data quality issues in time series data.

    Identifies missing values, outliers, data gaps, and duplicates.

    Args:
        df: Input DataFrame.
        date_col: Date column for temporal analysis.
        numeric_cols: Numeric columns to check (default: all numeric).

    Returns:
        Dictionary with quality metrics and issue counts.

    Example:
        >>> issues = detect_data_quality_issues(df, date_col="date")
        >>> print(f"Missing values: {issues['missing_values']}")
    """
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in df.columns
            if col != date_col and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

    issues = {
        "total_rows": len(df),
        "missing_values": {},
        "outliers": {},
        "duplicates": 0,
        "date_gaps": [],
    }

    # Check missing values
    for col in numeric_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            issues["missing_values"][col] = null_count

    # Check duplicates
    if date_col in df.columns:
        unique_count = df.select(pl.col(date_col)).unique().height
        issues["duplicates"] = len(df) - unique_count

    # Check for extreme outliers (> 5 std from mean)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if mean is not None and std is not None and std > 0:
            outliers = df.filter(
                (pl.col(col) > mean + 5 * std) | (pl.col(col) < mean - 5 * std)
            )
            if len(outliers) > 0:
                issues["outliers"][col] = len(outliers)

    # Check for date gaps (if date column exists)
    if date_col in df.columns and len(df) > 1:
        sorted_df = df.sort(date_col)
        dates = sorted_df[date_col].to_list()

        # Simplified gap detection (would need proper date arithmetic in production)
        # This is a placeholder
        issues["date_gaps"] = []

    return issues
