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
