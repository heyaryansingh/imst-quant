"""3-hour bucketing and daily sentiment aggregation (paper replication)."""

from datetime import date, datetime
from typing import Dict

import polars as pl

BUCKET_HOURS = 3
BUCKETS_PER_DAY = 24 // BUCKET_HOURS  # 8


def assign_time_bucket(timestamp: datetime) -> int:
    """Assign post to 3-hour bucket within day (0-7)."""
    return timestamp.hour // BUCKET_HOURS


def aggregate_daily_sentiment(
    posts_df: pl.DataFrame,
    target_date: date,
    influence_scores: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Aggregate sentiment to daily level per asset.

    Formula: sentiment_index(asset, day) = sum(polarity_i * influence_i) / sum(influence_i)
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
