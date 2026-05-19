"""Time-based sentiment aggregation for financial signal generation.

This module implements 3-hour bucketing and daily sentiment aggregation
as described in the paper replication methodology. Sentiment scores are
aggregated using influence-weighted averaging.

Constants:
    BUCKET_HOURS: Size of each time bucket in hours (3).
    BUCKETS_PER_DAY: Number of buckets per day (8).

Functions:
    assign_time_bucket: Map timestamp to 3-hour bucket index.
    aggregate_daily_sentiment: Compute influence-weighted daily sentiment.

Example:
    >>> from datetime import date
    >>> daily_sentiment = aggregate_daily_sentiment(posts_df, date(2026, 4, 7))
    >>> print(daily_sentiment)  # {'AAPL': 0.45, 'TSLA': -0.12}
"""

from datetime import date, datetime
from typing import Dict, Optional

import polars as pl

BUCKET_HOURS: int = 3
BUCKETS_PER_DAY: int = 24 // BUCKET_HOURS  # 8


def assign_time_bucket(timestamp: datetime) -> int:
    """Assign a timestamp to its corresponding 3-hour bucket.

    Divides a 24-hour day into 8 buckets of 3 hours each:
    - Bucket 0: 00:00-02:59
    - Bucket 1: 03:00-05:59
    - ...
    - Bucket 7: 21:00-23:59

    Args:
        timestamp: The datetime to assign to a bucket.

    Returns:
        Integer bucket index from 0 to 7.

    Example:
        >>> from datetime import datetime
        >>> assign_time_bucket(datetime(2026, 4, 7, 14, 30))
        4
    """
    return timestamp.hour // BUCKET_HOURS


def aggregate_daily_sentiment(
    posts_df: pl.DataFrame,
    target_date: date,
    influence_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Aggregate sentiment scores to daily level per asset with influence weighting.

    Computes a weighted average of sentiment polarities using author influence
    scores as weights. Authors without influence scores default to weight 1.0.

    Formula:
        sentiment_index(asset, day) = sum(polarity_i * influence_i) / sum(influence_i)

    Args:
        posts_df: Polars DataFrame with columns: date, asset_id, author_id, polarity.
        target_date: The date to aggregate sentiment for.
        influence_scores: Optional dict mapping author_id to influence weight.

    Returns:
        Dict mapping asset_id to aggregated sentiment score for the day.

    Example:
        >>> scores = aggregate_daily_sentiment(df, date(2026, 4, 7), {"user1": 2.5})
        >>> scores.get("AAPL", 0.0)
        0.32
    """
    influence_scores = influence_scores or {}
    date_str = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    day_posts = posts_df.filter(pl.col("date") == date_str)

    if len(day_posts) == 0:
        return {}

    results = {}
    for asset in day_posts["asset_id"].unique().to_list():
        asset_posts = day_posts.filter(pl.col("asset_id") == asset)
        weighted_sum = 0.0
        weight_sum = 0.0
        for row in asset_posts.iter_rows(named=True):
            author_id = str(row.get("author_id", ""))
            polarity = float(row.get("polarity", 0.0))
            influence = influence_scores.get(author_id, 1.0)
            weighted_sum += polarity * influence
            weight_sum += influence
        if weight_sum > 0:
            results[asset] = weighted_sum / weight_sum
        else:
            results[asset] = 0.0
    return results


def calculate_sentiment_momentum(
    sentiment_df: pl.DataFrame,
    asset_col: str = "asset_id",
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    window: int = 5
) -> pl.DataFrame:
    """Calculate sentiment momentum (rate of change) for each asset.

    Args:
        sentiment_df: DataFrame with daily sentiment scores.
        asset_col: Column name for asset identifier.
        sentiment_col: Column name for sentiment score.
        date_col: Column name for date.
        window: Lookback window for momentum calculation.

    Returns:
        DataFrame with sentiment_momentum column.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        ...     "asset_id": ["AAPL", "AAPL", "AAPL"],
        ...     "sentiment": [0.2, 0.3, 0.5]
        ... })
        >>> result = calculate_sentiment_momentum(df, window=2)
    """
    df = sentiment_df.sort([asset_col, date_col])

    df = df.with_columns([
        (pl.col(sentiment_col) - pl.col(sentiment_col).shift(window))
        .over(asset_col)
        .alias("sentiment_momentum")
    ])

    return df


def calculate_sentiment_volatility(
    sentiment_df: pl.DataFrame,
    asset_col: str = "asset_id",
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    window: int = 20
) -> pl.DataFrame:
    """Calculate rolling sentiment volatility for each asset.

    Measures stability/variability in sentiment over time.

    Args:
        sentiment_df: DataFrame with daily sentiment scores.
        asset_col: Column name for asset identifier.
        sentiment_col: Column name for sentiment score.
        date_col: Column name for date.
        window: Rolling window for volatility calculation.

    Returns:
        DataFrame with sentiment_volatility column.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": ["2026-01-01"] * 3,
        ...     "asset_id": ["AAPL", "GOOGL", "TSLA"],
        ...     "sentiment": [0.2, -0.1, 0.4]
        ... })
        >>> result = calculate_sentiment_volatility(df)
    """
    df = sentiment_df.sort([asset_col, date_col])

    df = df.with_columns([
        pl.col(sentiment_col)
        .rolling_std(window_size=window)
        .over(asset_col)
        .alias("sentiment_volatility")
    ])

    return df


def calculate_sentiment_z_score(
    sentiment_df: pl.DataFrame,
    asset_col: str = "asset_id",
    sentiment_col: str = "sentiment",
    window: int = 60
) -> pl.DataFrame:
    """Calculate z-score normalized sentiment for each asset.

    Normalizes sentiment relative to its historical distribution,
    useful for identifying extreme sentiment levels.

    Args:
        sentiment_df: DataFrame with sentiment scores.
        asset_col: Column name for asset identifier.
        sentiment_col: Column name for sentiment score.
        window: Rolling window for mean/std calculation.

    Returns:
        DataFrame with sentiment_z_score column.

    Example:
        >>> df = pl.DataFrame({
        ...     "asset_id": ["AAPL"] * 5,
        ...     "sentiment": [0.1, 0.3, -0.2, 0.5, 0.2]
        ... })
        >>> result = calculate_sentiment_z_score(df, window=3)
    """
    df = sentiment_df.with_columns([
        pl.col(sentiment_col)
        .rolling_mean(window_size=window)
        .over(asset_col)
        .alias("_rolling_mean"),
        pl.col(sentiment_col)
        .rolling_std(window_size=window)
        .over(asset_col)
        .alias("_rolling_std")
    ])

    df = df.with_columns([
        ((pl.col(sentiment_col) - pl.col("_rolling_mean")) / pl.col("_rolling_std"))
        .alias("sentiment_z_score")
    ]).drop(["_rolling_mean", "_rolling_std"])

    return df


def aggregate_sentiment_by_bucket(
    posts_df: pl.DataFrame,
    timestamp_col: str = "created_utc",
    sentiment_col: str = "polarity",
    asset_col: str = "asset_id"
) -> pl.DataFrame:
    """Aggregate sentiment scores into 3-hour time buckets.

    Args:
        posts_df: DataFrame with post-level sentiment data.
        timestamp_col: Column name for timestamp.
        sentiment_col: Column name for sentiment score.
        asset_col: Column name for asset identifier.

    Returns:
        DataFrame with bucket_id, asset_id, and aggregated sentiment.

    Example:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "created_utc": [datetime(2026, 1, 1, 2), datetime(2026, 1, 1, 5)],
        ...     "asset_id": ["AAPL", "AAPL"],
        ...     "polarity": [0.5, -0.2]
        ... })
        >>> result = aggregate_sentiment_by_bucket(df)
    """
    # Extract hour and calculate bucket
    df = posts_df.with_columns([
        pl.col(timestamp_col).dt.hour().alias("hour")
    ]).with_columns([
        (pl.col("hour") // BUCKET_HOURS).alias("bucket_id")
    ])

    # Aggregate by bucket and asset
    aggregated = df.group_by([asset_col, "bucket_id"]).agg([
        pl.col(sentiment_col).mean().alias("avg_sentiment"),
        pl.col(sentiment_col).count().alias("post_count"),
        pl.col(sentiment_col).std().alias("sentiment_std")
    ])

    return aggregated


def calculate_sentiment_divergence(
    sentiment_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    asset_col: str = "asset_id",
    date_col: str = "date",
    sentiment_col: str = "sentiment",
    return_col: str = "return_1d"
) -> pl.DataFrame:
    """Calculate divergence between sentiment and price returns.

    Identifies periods where sentiment and price movements disagree,
    which can signal reversals or momentum exhaustion.

    Args:
        sentiment_df: DataFrame with sentiment scores.
        returns_df: DataFrame with price returns.
        asset_col: Column name for asset identifier.
        date_col: Column name for date.
        sentiment_col: Column name for sentiment.
        return_col: Column name for returns.

    Returns:
        DataFrame with sentiment_return_divergence column.

    Example:
        >>> sent = pl.DataFrame({
        ...     "date": ["2026-01-01", "2026-01-02"],
        ...     "asset_id": ["AAPL", "AAPL"],
        ...     "sentiment": [0.5, 0.6]
        ... })
        >>> rets = pl.DataFrame({
        ...     "date": ["2026-01-01", "2026-01-02"],
        ...     "asset_id": ["AAPL", "AAPL"],
        ...     "return_1d": [-0.02, -0.01]
        ... })
        >>> result = calculate_sentiment_divergence(sent, rets)
    """
    # Join sentiment and returns
    merged = sentiment_df.join(
        returns_df.select([date_col, asset_col, return_col]),
        on=[date_col, asset_col],
        how="inner"
    )

    # Calculate divergence: positive when sentiment and returns disagree
    merged = merged.with_columns([
        (pl.col(sentiment_col).sign() != pl.col(return_col).sign())
        .cast(pl.Int8)
        .alias("sentiment_return_divergence")
    ])

    return merged
