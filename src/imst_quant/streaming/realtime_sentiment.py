"""Real-time sentiment aggregation and streaming analytics.

This module provides streaming sentiment aggregation with windowed statistics,
momentum tracking, and anomaly detection for live trading signals.
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger()


@dataclass
class SentimentWindow:
    """Sliding window for real-time sentiment aggregation."""

    window_minutes: int = 60  # Window size in minutes
    values: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)

    def add(self, timestamp: datetime, value: float) -> None:
        """Add a new sentiment value to the window."""
        self.values.append(value)
        self.timestamps.append(timestamp)

        # Remove values outside the window
        cutoff = timestamp - timedelta(minutes=self.window_minutes)
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            self.values.popleft()

    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics for the current window."""
        if not self.values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
            }

        vals = np.array(self.values)
        return {
            "count": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "sum": float(vals.sum()),
        }


@dataclass
class RealtimeSentimentConfig:
    """Configuration for real-time sentiment aggregation."""

    # Window sizes for multi-timeframe analysis (minutes)
    short_window: int = 15
    medium_window: int = 60
    long_window: int = 240

    # Momentum detection
    momentum_threshold: float = 0.15  # Significant momentum change
    momentum_lookback: int = 5  # Number of periods for momentum

    # Anomaly detection
    anomaly_std_threshold: float = 2.5  # Z-score threshold
    min_samples_for_anomaly: int = 10  # Min samples before detecting anomalies

    # Volume weighting
    use_volume_weighting: bool = True
    volume_decay_factor: float = 0.95  # Exponential decay for volume

    # Influence scoring (from GNN)
    use_influence_weighting: bool = True
    min_influence_score: float = 0.1  # Min influence to include


class RealtimeSentimentAggregator:
    """Real-time sentiment aggregation with multi-timeframe analysis.

    This class maintains sliding windows of sentiment data and calculates
    streaming statistics, momentum, and anomalies for live trading signals.

    Examples:
        >>> config = RealtimeSentimentConfig()
        >>> aggregator = RealtimeSentimentAggregator(config)
        >>>
        >>> # Add new sentiment datapoint
        >>> timestamp = datetime.now()
        >>> aggregator.add_sentiment(
        ...     asset="AAPL",
        ...     timestamp=timestamp,
        ...     sentiment=0.65,
        ...     volume=100,
        ...     influence_score=0.8
        ... )
        >>>
        >>> # Get current aggregated sentiment
        >>> signal = aggregator.get_signal("AAPL")
        >>> print(f"Sentiment: {signal['sentiment']}, Momentum: {signal['momentum']}")
    """

    def __init__(self, config: Optional[RealtimeSentimentConfig] = None):
        """Initialize real-time sentiment aggregator.

        Args:
            config: Configuration for sentiment aggregation
        """
        self.config = config or RealtimeSentimentConfig()

        # Asset-level sentiment windows
        self.asset_windows: Dict[str, Dict[str, SentimentWindow]] = {}

        # Historical values for momentum and anomaly detection
        self.historical_means: Dict[str, deque] = {}

        logger.info(
            "realtime_sentiment_aggregator_initialized",
            short_window=self.config.short_window,
            medium_window=self.config.medium_window,
            long_window=self.config.long_window,
        )

    def add_sentiment(
        self,
        asset: str,
        timestamp: datetime,
        sentiment: float,
        volume: Optional[int] = None,
        influence_score: Optional[float] = None,
    ) -> None:
        """Add a new sentiment observation.

        Args:
            asset: Asset identifier (e.g., "AAPL")
            timestamp: Timestamp of the observation
            sentiment: Sentiment score (typically -1 to 1)
            volume: Optional volume/engagement metric
            influence_score: Optional influence score from GNN
        """
        # Initialize windows for new assets
        if asset not in self.asset_windows:
            self.asset_windows[asset] = {
                "short": SentimentWindow(window_minutes=self.config.short_window),
                "medium": SentimentWindow(window_minutes=self.config.medium_window),
                "long": SentimentWindow(window_minutes=self.config.long_window),
            }
            self.historical_means[asset] = deque(maxlen=100)

        # Apply weighting
        weighted_sentiment = sentiment

        if self.config.use_influence_weighting and influence_score is not None:
            if influence_score >= self.config.min_influence_score:
                weighted_sentiment *= influence_score
            else:
                # Filter out low-influence posts
                return

        if self.config.use_volume_weighting and volume is not None:
            # Volume weighting with exponential decay
            volume_weight = min(volume / 100.0, 5.0)  # Cap at 5x
            weighted_sentiment *= volume_weight

        # Add to all windows
        for window in self.asset_windows[asset].values():
            window.add(timestamp, weighted_sentiment)

        logger.debug(
            "sentiment_added",
            asset=asset,
            sentiment=sentiment,
            weighted_sentiment=weighted_sentiment,
            timestamp=timestamp.isoformat(),
        )

    def get_signal(self, asset: str) -> Dict[str, float]:
        """Get current aggregated sentiment signal for an asset.

        Args:
            asset: Asset identifier

        Returns:
            Dict containing:
                - sentiment: Current sentiment score
                - momentum: Sentiment momentum (rate of change)
                - volatility: Sentiment volatility
                - anomaly_score: Anomaly detection score
                - confidence: Signal confidence based on sample size
        """
        if asset not in self.asset_windows:
            logger.warning("asset_not_found", asset=asset)
            return self._empty_signal()

        windows = self.asset_windows[asset]

        # Get multi-timeframe statistics
        short_stats = windows["short"].get_stats()
        medium_stats = windows["medium"].get_stats()
        long_stats = windows["long"].get_stats()

        # Current sentiment (short-term mean)
        sentiment = short_stats["mean"]

        # Calculate momentum (short vs medium term)
        momentum = self._calculate_momentum(short_stats["mean"], medium_stats["mean"])

        # Volatility (short-term std)
        volatility = short_stats["std"]

        # Anomaly detection
        anomaly_score = self._detect_anomaly(
            asset,
            sentiment,
            long_stats["mean"],
            long_stats["std"]
        )

        # Confidence based on sample size
        confidence = min(short_stats["count"] / 30.0, 1.0)

        # Store current mean for momentum tracking
        self.historical_means[asset].append(sentiment)

        signal = {
            "sentiment": sentiment,
            "momentum": momentum,
            "volatility": volatility,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "short_term": short_stats["mean"],
            "medium_term": medium_stats["mean"],
            "long_term": long_stats["mean"],
            "sample_count": short_stats["count"],
        }

        logger.info("signal_generated", asset=asset, **signal)
        return signal

    def _calculate_momentum(self, short_mean: float, medium_mean: float) -> float:
        """Calculate sentiment momentum.

        Args:
            short_mean: Short-term sentiment mean
            medium_mean: Medium-term sentiment mean

        Returns:
            Momentum score (positive = bullish acceleration, negative = bearish)
        """
        if medium_mean == 0:
            return 0.0

        # Percent change from medium to short term
        momentum = (short_mean - medium_mean) / abs(medium_mean)

        # Cap momentum at ±1.0
        return np.clip(momentum, -1.0, 1.0)

    def _detect_anomaly(
        self,
        asset: str,
        current: float,
        long_mean: float,
        long_std: float
    ) -> float:
        """Detect anomalous sentiment using z-score.

        Args:
            asset: Asset identifier
            current: Current sentiment value
            long_mean: Long-term mean
            long_std: Long-term standard deviation

        Returns:
            Anomaly score (0 = normal, 1 = anomalous)
        """
        # Need minimum samples for reliable anomaly detection
        if len(self.historical_means.get(asset, [])) < self.config.min_samples_for_anomaly:
            return 0.0

        if long_std == 0:
            return 0.0

        # Calculate z-score
        z_score = abs(current - long_mean) / long_std

        # Convert z-score to anomaly score (0-1)
        if z_score >= self.config.anomaly_std_threshold:
            # Sigmoid function to map z-score to 0-1
            anomaly_score = 1.0 / (1.0 + np.exp(-0.5 * (z_score - self.config.anomaly_std_threshold)))
            logger.warning(
                "anomaly_detected",
                asset=asset,
                z_score=z_score,
                anomaly_score=anomaly_score,
            )
            return anomaly_score

        return 0.0

    def get_multi_asset_signals(
        self,
        assets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get signals for multiple assets.

        Args:
            assets: List of asset identifiers (None = all tracked assets)

        Returns:
            DataFrame with columns: asset, sentiment, momentum, volatility, etc.
        """
        if assets is None:
            assets = list(self.asset_windows.keys())

        signals = []
        for asset in assets:
            signal = self.get_signal(asset)
            signal["asset"] = asset
            signals.append(signal)

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        df = df.sort_values("sentiment", ascending=False)

        return df

    def detect_divergence(
        self,
        asset: str,
        price_momentum: float
    ) -> Dict[str, float]:
        """Detect sentiment-price divergence.

        Args:
            asset: Asset identifier
            price_momentum: Price momentum (percent change)

        Returns:
            Dict containing divergence metrics
        """
        signal = self.get_signal(asset)
        sentiment_momentum = signal["momentum"]

        # Divergence occurs when sentiment and price move in opposite directions
        divergence_score = 0.0

        if abs(price_momentum) > 0.02 and abs(sentiment_momentum) > 0.05:
            # Significant movements in both
            if (price_momentum > 0 and sentiment_momentum < 0) or \
               (price_momentum < 0 and sentiment_momentum > 0):
                # Opposite directions
                divergence_score = min(
                    abs(price_momentum) * abs(sentiment_momentum) * 10.0,
                    1.0
                )

        result = {
            "divergence_score": divergence_score,
            "price_momentum": price_momentum,
            "sentiment_momentum": sentiment_momentum,
            "interpretation": self._interpret_divergence(
                price_momentum,
                sentiment_momentum,
                divergence_score
            ),
        }

        if divergence_score > 0.3:
            logger.warning("divergence_detected", asset=asset, **result)

        return result

    def _interpret_divergence(
        self,
        price_momentum: float,
        sentiment_momentum: float,
        divergence_score: float
    ) -> str:
        """Interpret divergence for trading signals.

        Args:
            price_momentum: Price momentum
            sentiment_momentum: Sentiment momentum
            divergence_score: Calculated divergence score

        Returns:
            Human-readable interpretation
        """
        if divergence_score < 0.2:
            return "no_divergence"

        if price_momentum > 0 and sentiment_momentum < 0:
            return "bearish_divergence"  # Price up, sentiment down → potential reversal
        elif price_momentum < 0 and sentiment_momentum > 0:
            return "bullish_divergence"  # Price down, sentiment up → potential reversal
        else:
            return "convergence"  # Same direction

    def get_relative_strength(
        self,
        assets: List[str]
    ) -> pd.DataFrame:
        """Calculate relative sentiment strength across assets.

        Args:
            assets: List of asset identifiers

        Returns:
            DataFrame with relative strength rankings
        """
        signals = self.get_multi_asset_signals(assets)

        if signals.empty:
            return pd.DataFrame()

        # Calculate z-scores for sentiment
        signals["sentiment_zscore"] = (
            signals["sentiment"] - signals["sentiment"].mean()
        ) / signals["sentiment"].std()

        # Calculate z-scores for momentum
        signals["momentum_zscore"] = (
            signals["momentum"] - signals["momentum"].mean()
        ) / signals["momentum"].std()

        # Combined strength score
        signals["relative_strength"] = (
            0.6 * signals["sentiment_zscore"] +
            0.4 * signals["momentum_zscore"]
        )

        # Rank assets
        signals["rank"] = signals["relative_strength"].rank(ascending=False)

        signals = signals.sort_values("relative_strength", ascending=False)

        return signals[[
            "asset",
            "sentiment",
            "momentum",
            "relative_strength",
            "rank",
            "confidence"
        ]]

    def _empty_signal(self) -> Dict[str, float]:
        """Return an empty signal dict."""
        return {
            "sentiment": 0.0,
            "momentum": 0.0,
            "volatility": 0.0,
            "anomaly_score": 0.0,
            "confidence": 0.0,
            "short_term": 0.0,
            "medium_term": 0.0,
            "long_term": 0.0,
            "sample_count": 0,
        }

    def reset_asset(self, asset: str) -> None:
        """Reset windows and history for an asset.

        Args:
            asset: Asset identifier
        """
        if asset in self.asset_windows:
            del self.asset_windows[asset]
        if asset in self.historical_means:
            del self.historical_means[asset]
        logger.info("asset_reset", asset=asset)

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics for the aggregator.

        Returns:
            Dict with summary information
        """
        n_assets = len(self.asset_windows)
        total_samples = sum(
            windows["short"].get_stats()["count"]
            for windows in self.asset_windows.values()
        )

        return {
            "n_assets_tracked": n_assets,
            "total_samples": total_samples,
            "avg_samples_per_asset": total_samples / n_assets if n_assets > 0 else 0,
            "assets": list(self.asset_windows.keys()),
        }
