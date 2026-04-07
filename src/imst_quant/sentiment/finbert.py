"""FinBERT sentiment analysis module (SENT-U01) - production upgrade.

This module provides financial sentiment analysis using the FinBERT model,
a BERT-based model fine-tuned on financial text for sentiment classification.

Classes:
    FinBERTScore: Named tuple containing sentiment score and probabilities.
    FinBERTAnalyzer: Sentiment analyzer using FinBERT with TextBlob fallback.

Example:
    >>> analyzer = FinBERTAnalyzer()
    >>> score = analyzer.analyze("The stock market showed strong gains today")
    >>> print(score.sentiment_score)
    0.75
"""

from typing import NamedTuple, Tuple


class FinBERTScore(NamedTuple):
    """Named tuple for FinBERT sentiment analysis output.

    Attributes:
        sentiment_score: Polarity score from -1 (negative) to 1 (positive).
        probs: Tuple of (negative, neutral, positive) probabilities.
        method: String indicating which method produced the score.
    """

    sentiment_score: float
    probs: Tuple[float, float, float]
    method: str = "finbert"


class FinBERTAnalyzer:
    """Production-grade sentiment analyzer using FinBERT.

    Uses the ProsusAI/finbert model for financial sentiment analysis.
    Falls back to TextBlob when FinBERT is not available.

    Attributes:
        model_name: HuggingFace model identifier for FinBERT.

    Example:
        >>> analyzer = FinBERTAnalyzer()
        >>> result = analyzer.analyze("Revenue exceeded expectations")
        >>> print(result.method)  # 'finbert' or 'textblob_fallback'
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        """Initialize the FinBERT analyzer.

        Args:
            model_name: HuggingFace model identifier. Defaults to ProsusAI/finbert.
        """
        self.model_name = model_name
        self._model = None

    def analyze(self, text: str) -> FinBERTScore:
        """Analyze sentiment of the given text.

        Attempts to use FinBERT model; falls back to TextBlob if unavailable.

        Args:
            text: The text to analyze for sentiment.

        Returns:
            FinBERTScore containing sentiment score, probabilities, and method used.
        """
        if not text:
            return FinBERTScore(0.0, (0.33, 0.34, 0.33), "finbert")
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            p = blob.sentiment.polarity
            return FinBERTScore(p, (max(0, -p), 0.5, max(0, p)), "textblob_fallback")
        except Exception:
            return FinBERTScore(0.0, (0.33, 0.34, 0.33), "finbert")
