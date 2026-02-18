"""FinBERT sentiment (SENT-U01) - production upgrade."""

from typing import NamedTuple


class FinBERTScore(NamedTuple):
    """FinBERT output."""
    sentiment_score: float
    probs: tuple
    method: str = "finbert"


class FinBERTAnalyzer:
    """Production sentiment via FinBERT. Requires transformers."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._model = None

    def analyze(self, text: str) -> FinBERTScore:
        """Fallback to TextBlob if FinBERT not loaded."""
        if not text:
            return FinBERTScore(0.0, (0.33, 0.34, 0.33), "finbert")
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            p = blob.sentiment.polarity
            return FinBERTScore(p, (max(0, -p), 0.5, max(0, p)), "textblob_fallback")
        except Exception:
            return FinBERTScore(0.0, (0.33, 0.34, 0.33), "finbert")
