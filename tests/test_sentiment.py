"""Unit tests for baseline sentiment (SENT-01 to SENT-04)."""

from datetime import date, datetime

import polars as pl
import pytest

from imst_quant.sentiment.aggregation import (
    BUCKET_HOURS,
    BUCKETS_PER_DAY,
    aggregate_daily_sentiment,
    assign_time_bucket,
)
from imst_quant.sentiment.textblob import BaselineSentimentAnalyzer, SentimentScore


def test_textblob_polarity_range():
    """SENT-01: TextBlob polarity in [-1, 1]."""
    analyzer = BaselineSentimentAnalyzer()
    s = analyzer.analyze("Apple is great")
    assert -1 <= s.polarity <= 1
    assert 0 <= s.subjectivity <= 1
    assert s.method == "textblob"


def test_textblob_empty():
    """BaselineSentimentAnalyzer handles empty text."""
    analyzer = BaselineSentimentAnalyzer()
    s = analyzer.analyze("")
    assert s.polarity == 0.0
    assert s.subjectivity == 0.0


def test_assign_time_bucket():
    """SENT-02: 3-hour buckets 0-7."""
    assert assign_time_bucket(datetime(2024, 1, 15, 0, 0)) == 0
    assert assign_time_bucket(datetime(2024, 1, 15, 2, 59)) == 0
    assert assign_time_bucket(datetime(2024, 1, 15, 3, 0)) == 1
    assert assign_time_bucket(datetime(2024, 1, 15, 6, 0)) == 2
    assert assign_time_bucket(datetime(2024, 1, 15, 21, 0)) == 7


def test_bucket_constants():
    """BUCKET_HOURS and BUCKETS_PER_DAY."""
    assert BUCKET_HOURS == 3
    assert BUCKETS_PER_DAY == 8


def test_aggregate_daily_sentiment_formula():
    """SENT-03, SENT-04: weighted mean formula."""
    posts_df = pl.DataFrame({
        "date": ["2024-01-15", "2024-01-15"],
        "asset_id": ["AAPL", "AAPL"],
        "author_id": ["a1", "a2"],
        "polarity": [0.5, -0.3],
    })
    # Equal influence (1.0): (0.5 - 0.3) / 2 = 0.1
    result = aggregate_daily_sentiment(posts_df, date(2024, 1, 15))
    assert "AAPL" in result
    assert abs(result["AAPL"] - 0.1) < 1e-6


def test_aggregate_daily_sentiment_influence_weighting():
    """SENT-03: influence weighting."""
    posts_df = pl.DataFrame({
        "date": ["2024-01-15", "2024-01-15"],
        "asset_id": ["AAPL", "AAPL"],
        "author_id": ["a1", "a2"],
        "polarity": [1.0, -1.0],
    })
    # a1 influence 2, a2 influence 1: (2*1 + 1*(-1)) / 3 = 1/3
    result = aggregate_daily_sentiment(
        posts_df, date(2024, 1, 15), influence_scores={"a1": 2.0, "a2": 1.0}
    )
    assert abs(result["AAPL"] - (1 / 3)) < 1e-6
