"""Advanced Sentiment Fusion - Multi-source sentiment aggregation with confidence weighting.

Combines sentiment from multiple sources (Reddit, news, social media, technical indicators)
using Bayesian fusion, time-decay weighting, and source credibility scoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum


class SentimentSource(Enum):
    """Sentiment data sources."""
    REDDIT = "reddit"
    NEWS = "news"
    TWITTER = "twitter"
    TECHNICAL = "technical"
    OPTIONS_FLOW = "options_flow"
    INSIDER_TRADING = "insider_trading"


@dataclass
class SentimentSignal:
    """Individual sentiment signal from a source."""
    source: SentimentSource
    timestamp: datetime
    symbol: str
    sentiment: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    volume: int  # Number of mentions/data points
    credibility: float = 1.0  # Source credibility multiplier


@dataclass
class FusedSentiment:
    """Fused sentiment result."""
    symbol: str
    sentiment: float
    confidence: float
    contributing_sources: List[SentimentSource]
    source_agreement: float  # 0-1, how much sources agree
    volume_weighted_sentiment: float
    time_weighted_sentiment: float
    bayesian_sentiment: float
    uncertainty: float


class SentimentFusion:
    """Multi-source sentiment fusion engine."""

    def __init__(
        self,
        time_decay_hours: float = 24.0,
        min_confidence: float = 0.3,
        credibility_weights: Optional[Dict[SentimentSource, float]] = None
    ):
        """Initialize sentiment fusion engine.

        Args:
            time_decay_hours: Hours for exponential time decay
            min_confidence: Minimum confidence threshold
            credibility_weights: Custom credibility weights per source
        """
        self.time_decay_hours = time_decay_hours
        self.min_confidence = min_confidence

        # Default credibility weights
        self.credibility_weights = credibility_weights or {
            SentimentSource.INSIDER_TRADING: 1.5,
            SentimentSource.OPTIONS_FLOW: 1.3,
            SentimentSource.NEWS: 1.2,
            SentimentSource.TECHNICAL: 1.1,
            SentimentSource.REDDIT: 0.9,
            SentimentSource.TWITTER: 0.8
        }

    def calculate_time_weight(
        self,
        signal_time: datetime,
        current_time: datetime
    ) -> float:
        """Calculate exponential time decay weight.

        Args:
            signal_time: When signal was generated
            current_time: Current time

        Returns:
            Time weight (0-1)
        """
        hours_ago = (current_time - signal_time).total_seconds() / 3600
        decay_rate = np.log(2) / self.time_decay_hours  # Half-life decay
        weight = np.exp(-decay_rate * hours_ago)
        return weight

    def calculate_source_agreement(self, signals: List[SentimentSignal]) -> float:
        """Calculate agreement between different sources.

        Args:
            signals: List of sentiment signals

        Returns:
            Agreement score (0-1)
        """
        if len(signals) < 2:
            return 1.0

        sentiments = np.array([s.sentiment for s in signals])

        # Calculate standard deviation of sentiments
        std = np.std(sentiments)

        # Convert to agreement score (lower std = higher agreement)
        # std of 0 = perfect agreement (1.0)
        # std of 2 = complete disagreement (0.0)
        agreement = 1.0 - min(std / 2.0, 1.0)

        return agreement

    def bayesian_fusion(
        self,
        signals: List[SentimentSignal],
        prior_sentiment: float = 0.0,
        prior_confidence: float = 0.5
    ) -> Tuple[float, float]:
        """Bayesian fusion of sentiment signals.

        Args:
            signals: List of sentiment signals
            prior_sentiment: Prior belief about sentiment
            prior_confidence: Confidence in prior

        Returns:
            Tuple of (posterior_sentiment, posterior_confidence)
        """
        # Start with prior
        posterior_mean = prior_sentiment
        posterior_precision = prior_confidence

        for signal in signals:
            # Signal precision (inverse variance)
            signal_precision = signal.confidence * signal.credibility

            # Bayesian update
            total_precision = posterior_precision + signal_precision

            posterior_mean = (
                (posterior_precision * posterior_mean + signal_precision * signal.sentiment)
                / total_precision
            )
            posterior_precision = total_precision

        # Convert precision back to confidence
        posterior_confidence = min(posterior_precision, 1.0)

        return posterior_mean, posterior_confidence

    def volume_weighted_fusion(self, signals: List[SentimentSignal]) -> float:
        """Calculate volume-weighted sentiment.

        Args:
            signals: List of sentiment signals

        Returns:
            Volume-weighted sentiment
        """
        if not signals:
            return 0.0

        total_volume = sum(s.volume for s in signals)
        if total_volume == 0:
            return np.mean([s.sentiment for s in signals])

        weighted_sentiment = sum(
            s.sentiment * s.volume for s in signals
        ) / total_volume

        return weighted_sentiment

    def time_weighted_fusion(
        self,
        signals: List[SentimentSignal],
        current_time: datetime
    ) -> float:
        """Calculate time-weighted sentiment.

        Args:
            signals: List of sentiment signals
            current_time: Current timestamp

        Returns:
            Time-weighted sentiment
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            time_weight = self.calculate_time_weight(signal.timestamp, current_time)
            weight = time_weight * signal.confidence * signal.credibility

            weighted_sum += signal.sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def calculate_uncertainty(self, signals: List[SentimentSignal]) -> float:
        """Calculate uncertainty in fused sentiment.

        Args:
            signals: List of sentiment signals

        Returns:
            Uncertainty measure (0-1)
        """
        if len(signals) < 2:
            return 0.5

        # Factors contributing to uncertainty:
        # 1. Disagreement between sources
        agreement = self.calculate_source_agreement(signals)

        # 2. Low average confidence
        avg_confidence = np.mean([s.confidence for s in signals])

        # 3. Few sources
        source_penalty = 1.0 / len(signals)

        uncertainty = (1.0 - agreement) * 0.5 + (1.0 - avg_confidence) * 0.3 + source_penalty * 0.2

        return min(uncertainty, 1.0)

    def fuse_sentiment(
        self,
        symbol: str,
        signals: List[SentimentSignal],
        current_time: Optional[datetime] = None
    ) -> FusedSentiment:
        """Fuse multiple sentiment signals into unified sentiment.

        Args:
            symbol: Stock symbol
            signals: List of sentiment signals
            current_time: Current timestamp (defaults to now)

        Returns:
            FusedSentiment object
        """
        if current_time is None:
            current_time = datetime.now()

        # Filter by confidence threshold
        valid_signals = [
            s for s in signals
            if s.confidence >= self.min_confidence and s.symbol == symbol
        ]

        if not valid_signals:
            return FusedSentiment(
                symbol=symbol,
                sentiment=0.0,
                confidence=0.0,
                contributing_sources=[],
                source_agreement=0.0,
                volume_weighted_sentiment=0.0,
                time_weighted_sentiment=0.0,
                bayesian_sentiment=0.0,
                uncertainty=1.0
            )

        # Apply credibility weights
        for signal in valid_signals:
            signal.credibility = self.credibility_weights.get(
                signal.source, 1.0
            )

        # Calculate different fusion methods
        volume_weighted = self.volume_weighted_fusion(valid_signals)
        time_weighted = self.time_weighted_fusion(valid_signals, current_time)
        bayesian_sent, bayesian_conf = self.bayesian_fusion(valid_signals)

        # Final fusion: weighted average of methods
        final_sentiment = (
            volume_weighted * 0.3 +
            time_weighted * 0.4 +
            bayesian_sent * 0.3
        )

        # Calculate metadata
        agreement = self.calculate_source_agreement(valid_signals)
        uncertainty = self.calculate_uncertainty(valid_signals)
        sources = list(set(s.source for s in valid_signals))

        # Final confidence combines Bayesian confidence with agreement
        final_confidence = bayesian_conf * agreement

        return FusedSentiment(
            symbol=symbol,
            sentiment=final_sentiment,
            confidence=final_confidence,
            contributing_sources=sources,
            source_agreement=agreement,
            volume_weighted_sentiment=volume_weighted,
            time_weighted_sentiment=time_weighted,
            bayesian_sentiment=bayesian_sent,
            uncertainty=uncertainty
        )

    def batch_fuse(
        self,
        signals_by_symbol: Dict[str, List[SentimentSignal]],
        current_time: Optional[datetime] = None
    ) -> Dict[str, FusedSentiment]:
        """Fuse sentiment for multiple symbols in batch.

        Args:
            signals_by_symbol: Dict mapping symbols to signal lists
            current_time: Current timestamp

        Returns:
            Dict mapping symbols to FusedSentiment
        """
        results = {}

        for symbol, signals in signals_by_symbol.items():
            results[symbol] = self.fuse_sentiment(symbol, signals, current_time)

        return results

    def create_fusion_report(
        self,
        fused_sentiments: Dict[str, FusedSentiment]
    ) -> pd.DataFrame:
        """Create detailed fusion report.

        Args:
            fused_sentiments: Dict of fused sentiments

        Returns:
            DataFrame with fusion details
        """
        data = []

        for symbol, fs in fused_sentiments.items():
            data.append({
                'symbol': symbol,
                'sentiment': fs.sentiment,
                'confidence': fs.confidence,
                'uncertainty': fs.uncertainty,
                'agreement': fs.source_agreement,
                'num_sources': len(fs.contributing_sources),
                'sources': ', '.join([s.value for s in fs.contributing_sources]),
                'volume_weighted': fs.volume_weighted_sentiment,
                'time_weighted': fs.time_weighted_sentiment,
                'bayesian': fs.bayesian_sentiment
            })

        df = pd.DataFrame(data)
        df = df.sort_values('confidence', ascending=False)

        return df


def detect_sentiment_divergence(
    fused_sentiment: FusedSentiment,
    threshold: float = 0.5
) -> bool:
    """Detect if there's significant divergence in sentiment sources.

    Args:
        fused_sentiment: FusedSentiment object
        threshold: Agreement threshold below which divergence is flagged

    Returns:
        True if divergence detected
    """
    return fused_sentiment.source_agreement < threshold


def sentiment_regime_classifier(sentiment: float, confidence: float) -> str:
    """Classify sentiment into regime.

    Args:
        sentiment: Fused sentiment value
        confidence: Confidence in sentiment

    Returns:
        Regime classification string
    """
    if confidence < 0.4:
        return "UNCERTAIN"

    if sentiment > 0.6:
        return "STRONG_BULLISH"
    elif sentiment > 0.2:
        return "BULLISH"
    elif sentiment < -0.6:
        return "STRONG_BEARISH"
    elif sentiment < -0.2:
        return "BEARISH"
    else:
        return "NEUTRAL"
