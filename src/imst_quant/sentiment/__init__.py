"""Sentiment analysis: TextBlob baseline, FinBERT, fusion, confidence, aggregation."""

from .textblob import BaselineSentimentAnalyzer, SentimentScore
from .aggregation import (
    BUCKET_HOURS,
    BUCKETS_PER_DAY,
    assign_time_bucket,
    aggregate_daily_sentiment,
)
from .finbert import FinBERTScore, FinBERTAnalyzer
from .advanced_scoring import (
    SentimentResult,
    AdvancedSentimentScorer,
    TemporalSentimentAnalyzer,
    compute_sentiment_divergence,
)
from .fusion import (
    SentimentSource,
    SentimentSignal,
    FusedSentiment,
    SentimentFusion,
    detect_sentiment_divergence,
    sentiment_regime_classifier,
)
from .confidence import (
    calculate_text_quality_score,
    calculate_sample_size_confidence,
    calculate_author_credibility_score,
    calculate_temporal_consistency,
    calculate_model_agreement,
    calculate_confidence_score,
    filter_by_confidence,
)

__all__ = [
    # TextBlob baseline
    "BaselineSentimentAnalyzer",
    "SentimentScore",
    # Aggregation
    "BUCKET_HOURS",
    "BUCKETS_PER_DAY",
    "assign_time_bucket",
    "aggregate_daily_sentiment",
    # FinBERT
    "FinBERTScore",
    "FinBERTAnalyzer",
    # Advanced scoring
    "SentimentResult",
    "AdvancedSentimentScorer",
    "TemporalSentimentAnalyzer",
    "compute_sentiment_divergence",
    # Fusion
    "SentimentSource",
    "SentimentSignal",
    "FusedSentiment",
    "SentimentFusion",
    "detect_sentiment_divergence",
    "sentiment_regime_classifier",
    # Confidence
    "calculate_text_quality_score",
    "calculate_sample_size_confidence",
    "calculate_author_credibility_score",
    "calculate_temporal_consistency",
    "calculate_model_agreement",
    "calculate_confidence_score",
    "filter_by_confidence",
]
