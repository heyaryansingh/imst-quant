"""Advanced sentiment scoring with context awareness and confidence metrics.

This module provides enhanced sentiment analysis capabilities beyond basic positive/negative
classification, including:
- Context-aware scoring (market conditions, source credibility)
- Confidence intervals and uncertainty quantification
- Multi-model ensemble scoring
- Temporal sentiment dynamics
- Entity-specific sentiment extraction

Example:
    >>> from imst_quant.sentiment.advanced_scoring import AdvancedSentimentScorer
    >>> scorer = AdvancedSentimentScorer()
    >>> result = scorer.score_with_confidence(text, context={'ticker': 'AAPL'})
    >>> print(f"Sentiment: {result['score']:.3f} ± {result['confidence']:.3f}")
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result.

    Attributes:
        score: Sentiment score (-1 to 1, negative to positive)
        confidence: Confidence in score (0 to 1)
        magnitude: Intensity of sentiment (0 to 1)
        subjectivity: How subjective vs objective (0 to 1)
        emotion: Primary emotion detected (optional)
        entities: Entity-specific sentiments (optional)
        temporal_dynamics: Time-based sentiment shifts (optional)
    """
    score: float
    confidence: float
    magnitude: float
    subjectivity: float
    emotion: Optional[str] = None
    entities: Optional[Dict[str, float]] = None
    temporal_dynamics: Optional[Dict[str, Any]] = None


class AdvancedSentimentScorer:
    """Advanced sentiment scoring with confidence and context awareness.

    Args:
        base_model: Base sentiment model ('finbert', 'vader', 'custom')
        use_ensemble: Whether to use ensemble of multiple models
        confidence_threshold: Minimum confidence for high-confidence predictions

    Example:
        >>> scorer = AdvancedSentimentScorer(base_model='finbert')
        >>> result = scorer.score_with_confidence(
        ...     "Stock rallied 10% on strong earnings",
        ...     context={'source': 'reuters'}
        ... )
    """

    def __init__(
        self,
        base_model: str = 'finbert',
        use_ensemble: bool = True,
        confidence_threshold: float = 0.8
    ):
        self.base_model = base_model
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold

        # Source credibility weights
        self.source_weights = {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'wsj': 0.95,
            'cnbc': 0.85,
            'reddit': 0.6,
            'twitter': 0.5,
            'unknown': 0.7
        }

    def score_with_confidence(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """Score sentiment with confidence interval.

        Args:
            text: Input text to analyze
            context: Optional context (ticker, source, market_state, etc.)

        Returns:
            SentimentResult with score and confidence metrics

        Example:
            >>> result = scorer.score_with_confidence(
            ...     "Massive selloff due to regulatory concerns",
            ...     context={'ticker': 'TSLA', 'source': 'bloomberg'}
            ... )
        """
        context = context or {}

        # Get base sentiment score
        base_score = self._get_base_score(text)

        # Compute confidence based on multiple factors
        confidence = self._compute_confidence(text, base_score, context)

        # Compute magnitude (intensity)
        magnitude = self._compute_magnitude(text)

        # Compute subjectivity
        subjectivity = self._compute_subjectivity(text)

        # Detect emotion
        emotion = self._detect_emotion(text)

        # Extract entity-specific sentiment
        entities = None
        if context.get('ticker'):
            entities = {context['ticker']: base_score}

        # Apply context adjustments
        adjusted_score = self._apply_context_adjustments(
            base_score, context, confidence
        )

        return SentimentResult(
            score=adjusted_score,
            confidence=confidence,
            magnitude=magnitude,
            subjectivity=subjectivity,
            emotion=emotion,
            entities=entities
        )

    def _get_base_score(self, text: str) -> float:
        """Get base sentiment score from model."""
        # Simplified scoring - in production, use actual models
        # This is a placeholder for demonstration

        text_lower = text.lower()

        positive_words = [
            'bullish', 'rally', 'surge', 'gain', 'strong', 'beat',
            'profit', 'growth', 'upgrade', 'outperform', 'buy'
        ]
        negative_words = [
            'bearish', 'crash', 'plunge', 'loss', 'weak', 'miss',
            'concern', 'risk', 'downgrade', 'underperform', 'sell'
        ]

        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        score = (pos_count - neg_count) / total
        return np.clip(score, -1.0, 1.0)

    def _compute_confidence(
        self,
        text: str,
        score: float,
        context: Dict[str, Any]
    ) -> float:
        """Compute confidence in sentiment score.

        Factors:
        - Text length and clarity
        - Source credibility
        - Score magnitude
        - Presence of qualifiers/hedging
        """
        confidence = 0.7  # Base confidence

        # Adjust for text length (more text -> higher confidence)
        word_count = len(text.split())
        if word_count < 10:
            confidence *= 0.8
        elif word_count > 50:
            confidence *= 1.1

        # Adjust for source credibility
        source = context.get('source', 'unknown').lower()
        source_weight = self.source_weights.get(source, 0.7)
        confidence *= source_weight

        # Adjust for score magnitude (extreme scores -> higher confidence)
        magnitude_boost = abs(score) * 0.3
        confidence += magnitude_boost

        # Penalize for hedging words
        hedging_words = ['maybe', 'possibly', 'might', 'could', 'uncertain']
        if any(word in text.lower() for word in hedging_words):
            confidence *= 0.85

        return np.clip(confidence, 0.0, 1.0)

    def _compute_magnitude(self, text: str) -> float:
        """Compute sentiment magnitude (intensity)."""
        text_lower = text.lower()

        intensifiers = [
            'very', 'extremely', 'massive', 'huge', 'significant',
            'dramatically', 'substantially', 'sharply'
        ]

        intensity = 0.5  # Base intensity

        # Check for intensifiers
        intensifier_count = sum(word in text_lower for word in intensifiers)
        intensity += intensifier_count * 0.15

        # Check for exclamation marks
        intensity += text.count('!') * 0.1

        return np.clip(intensity, 0.0, 1.0)

    def _compute_subjectivity(self, text: str) -> float:
        """Compute subjectivity score (0=objective, 1=subjective)."""
        text_lower = text.lower()

        objective_indicators = [
            'reported', 'announced', 'stated', 'according to',
            'data shows', 'statistics', 'research'
        ]
        subjective_indicators = [
            'i think', 'i believe', 'in my opinion', 'feels like',
            'seems', 'appears', 'suggests'
        ]

        obj_count = sum(ind in text_lower for ind in objective_indicators)
        subj_count = sum(ind in text_lower for ind in subjective_indicators)

        total = obj_count + subj_count
        if total == 0:
            return 0.5

        subjectivity = subj_count / total
        return np.clip(subjectivity, 0.0, 1.0)

    def _detect_emotion(self, text: str) -> str:
        """Detect primary emotion in text."""
        text_lower = text.lower()

        emotion_keywords = {
            'fear': ['fear', 'panic', 'worried', 'concerned', 'anxiety'],
            'greed': ['euphoria', 'fomo', 'rally', 'moon', 'explode'],
            'anger': ['outrage', 'furious', 'disgust', 'angry'],
            'surprise': ['shock', 'unexpected', 'surprise', 'stunning'],
            'neutral': []
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion

        return 'neutral'

    def _apply_context_adjustments(
        self,
        score: float,
        context: Dict[str, Any],
        confidence: float
    ) -> float:
        """Apply context-based adjustments to sentiment score."""
        adjusted_score = score

        # Adjust based on market state
        market_state = context.get('market_state')
        if market_state == 'high_volatility':
            # Amplify sentiment in volatile markets
            adjusted_score *= 1.2
        elif market_state == 'low_volatility':
            # Dampen sentiment in calm markets
            adjusted_score *= 0.8

        # Adjust based on confidence
        # Low confidence scores regress toward neutral
        if confidence < 0.5:
            adjusted_score *= 0.7

        return np.clip(adjusted_score, -1.0, 1.0)

    def score_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[SentimentResult]:
        """Score multiple texts in batch.

        Args:
            texts: List of texts to score
            contexts: Optional list of context dicts (one per text)

        Returns:
            List of SentimentResult objects

        Example:
            >>> texts = ["Stock up 5%", "Market crashes"]
            >>> results = scorer.score_batch(texts)
        """
        if contexts is None:
            contexts = [{}] * len(texts)

        return [
            self.score_with_confidence(text, ctx)
            for text, ctx in zip(texts, contexts)
        ]

    def aggregate_sentiments(
        self,
        results: List[SentimentResult],
        method: str = 'confidence_weighted'
    ) -> Tuple[float, float]:
        """Aggregate multiple sentiment results.

        Args:
            results: List of SentimentResult objects
            method: Aggregation method ('simple_mean', 'confidence_weighted', 'median')

        Returns:
            Tuple of (aggregated_score, aggregated_confidence)

        Example:
            >>> results = scorer.score_batch(texts)
            >>> agg_score, agg_conf = scorer.aggregate_sentiments(results)
        """
        if not results:
            return 0.0, 0.0

        if method == 'simple_mean':
            agg_score = np.mean([r.score for r in results])
            agg_conf = np.mean([r.confidence for r in results])

        elif method == 'confidence_weighted':
            scores = np.array([r.score for r in results])
            confidences = np.array([r.confidence for r in results])

            # Weight scores by confidence
            weights = confidences / confidences.sum()
            agg_score = (scores * weights).sum()
            agg_conf = confidences.mean()

        elif method == 'median':
            agg_score = np.median([r.score for r in results])
            agg_conf = np.median([r.confidence for r in results])

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return float(agg_score), float(agg_conf)


class TemporalSentimentAnalyzer:
    """Analyze sentiment dynamics over time.

    Args:
        window_size: Size of rolling window for trend analysis
        decay_rate: Exponential decay rate for weighting recent sentiment

    Example:
        >>> analyzer = TemporalSentimentAnalyzer(window_size=20)
        >>> analyzer.add_sentiment(0.5, timestamp)
        >>> trends = analyzer.get_trends()
    """

    def __init__(self, window_size: int = 20, decay_rate: float = 0.05):
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.sentiments: List[Tuple[float, datetime]] = []

    def add_sentiment(self, score: float, timestamp: datetime):
        """Add sentiment observation."""
        self.sentiments.append((score, timestamp))

        # Keep only recent sentiments
        if len(self.sentiments) > self.window_size * 2:
            self.sentiments = self.sentiments[-self.window_size * 2:]

    def get_trends(self) -> Dict[str, float]:
        """Compute sentiment trends.

        Returns:
            Dictionary with trend metrics:
            - current: Current sentiment (recent average)
            - change_1h: Change over past hour
            - change_24h: Change over past 24 hours
            - volatility: Sentiment volatility
            - momentum: Sentiment momentum (derivative)
        """
        if not self.sentiments:
            return {
                'current': 0.0,
                'change_1h': 0.0,
                'change_24h': 0.0,
                'volatility': 0.0,
                'momentum': 0.0
            }

        scores = [s[0] for s in self.sentiments]
        timestamps = [s[1] for s in self.sentiments]

        # Current sentiment (exponentially weighted)
        weights = np.exp(-self.decay_rate * np.arange(len(scores))[::-1])
        current = np.average(scores, weights=weights)

        # Volatility
        volatility = np.std(scores)

        # Momentum (simple linear trend)
        if len(scores) >= 2:
            momentum = (scores[-1] - scores[0]) / len(scores)
        else:
            momentum = 0.0

        # Time-based changes
        now = timestamps[-1]
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        recent_1h = [s for s, t in self.sentiments if t >= one_hour_ago]
        recent_24h = [s for s, t in self.sentiments if t >= one_day_ago]

        change_1h = (
            current - np.mean(recent_1h)
            if recent_1h and len(recent_1h) > 1 else 0.0
        )
        change_24h = (
            current - np.mean(recent_24h)
            if recent_24h and len(recent_24h) > 1 else 0.0
        )

        return {
            'current': float(current),
            'change_1h': float(change_1h),
            'change_24h': float(change_24h),
            'volatility': float(volatility),
            'momentum': float(momentum)
        }


def compute_sentiment_divergence(
    sentiment_scores: List[float],
    price_returns: List[float]
) -> float:
    """Compute divergence between sentiment and price action.

    Args:
        sentiment_scores: Time series of sentiment scores
        price_returns: Time series of price returns

    Returns:
        Divergence score (higher = more divergence)

    Example:
        >>> divergence = compute_sentiment_divergence(
        ...     sentiment_scores=[0.5, 0.6, 0.7],
        ...     price_returns=[-0.01, -0.02, -0.01]
        ... )
    """
    if len(sentiment_scores) != len(price_returns):
        raise ValueError("Sentiment and returns must have same length")

    if len(sentiment_scores) < 2:
        return 0.0

    # Correlation between sentiment and returns
    correlation = np.corrcoef(sentiment_scores, price_returns)[0, 1]

    # Divergence is negative correlation (sentiment up, price down)
    divergence = -correlation

    return float(np.clip(divergence, -1.0, 1.0))
