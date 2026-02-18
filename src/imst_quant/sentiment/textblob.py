"""Baseline sentiment via TextBlob (paper replication)."""

from typing import NamedTuple

from textblob import TextBlob


class SentimentScore(NamedTuple):
    """TextBlob sentiment output."""

    polarity: float  # [-1, 1]
    subjectivity: float  # [0, 1]
    method: str = "textblob"


class BaselineSentimentAnalyzer:
    """Paper replication: TextBlob polarity."""

    def analyze(self, text: str) -> SentimentScore:
        if not text or not isinstance(text, str):
            return SentimentScore(polarity=0.0, subjectivity=0.0, method="textblob")
        blob = TextBlob(text)
        return SentimentScore(
            polarity=float(blob.sentiment.polarity),
            subjectivity=float(blob.sentiment.subjectivity),
            method="textblob",
        )
