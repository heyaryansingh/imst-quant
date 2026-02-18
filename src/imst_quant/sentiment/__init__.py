"""Sentiment analysis: TextBlob baseline, aggregation."""

from .textblob import BaselineSentimentAnalyzer, SentimentScore
from .aggregation import (
    BUCKET_HOURS,
    BUCKETS_PER_DAY,
    assign_time_bucket,
    aggregate_daily_sentiment,
)

__all__ = [
    "BaselineSentimentAnalyzer",
    "SentimentScore",
    "BUCKET_HOURS",
    "BUCKETS_PER_DAY",
    "assign_time_bucket",
    "aggregate_daily_sentiment",
]
