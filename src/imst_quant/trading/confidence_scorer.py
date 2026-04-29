"""Signal confidence scoring for trading decisions.

Provides methods to assign confidence scores to trading signals based on:
- Model prediction probability
- Technical indicator agreement
- Volume confirmation
- Volatility regime
- Historical accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger()


class SignalConfidenceScorer:
    """Score trading signals with confidence metrics for better position sizing."""

    def __init__(
        self,
        probability_weight: float = 0.4,
        technical_weight: float = 0.3,
        volume_weight: float = 0.15,
        volatility_weight: float = 0.15,
    ):
        """Initialize the confidence scorer.

        Args:
            probability_weight: Weight for model probability (0-1)
            technical_weight: Weight for technical indicator agreement
            volume_weight: Weight for volume confirmation
            volatility_weight: Weight for volatility regime
        """
        # Normalize weights
        total = probability_weight + technical_weight + volume_weight + volatility_weight
        self.probability_weight = probability_weight / total
        self.technical_weight = technical_weight / total
        self.volume_weight = volume_weight / total
        self.volatility_weight = volatility_weight / total

        logger.info("Confidence scorer initialized",
                   weights={
                       "probability": self.probability_weight,
                       "technical": self.technical_weight,
                       "volume": self.volume_weight,
                       "volatility": self.volatility_weight,
                   })

    def score_signal(
        self,
        signal: int,
        prob: float,
        technical_score: float,
        volume_score: float,
        volatility_score: float,
    ) -> float:
        """Calculate confidence score for a trading signal.

        Args:
            signal: Trading signal (-1, 0, 1)
            prob: Model probability (0-1)
            technical_score: Technical indicator agreement score (0-1)
            volume_score: Volume confirmation score (0-1)
            volatility_score: Volatility regime score (0-1)

        Returns:
            Confidence score (0-1)

        Example:
            >>> scorer = SignalConfidenceScorer()
            >>> confidence = scorer.score_signal(1, 0.75, 0.8, 0.6, 0.7)
        """
        if signal == 0:
            return 0.0  # No confidence in neutral signals

        # Convert probability to distance from 0.5
        prob_score = abs(prob - 0.5) * 2  # Scale to 0-1

        # Weighted combination
        confidence = (
            self.probability_weight * prob_score +
            self.technical_weight * technical_score +
            self.volume_weight * volume_score +
            self.volatility_weight * volatility_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def technical_agreement_score(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        indicator_cols: Optional[List[str]] = None,
    ) -> pd.Series:
        """Calculate technical indicator agreement score.

        Args:
            df: DataFrame with signals and technical indicators
            signal_col: Column name for primary signal
            indicator_cols: List of indicator columns that produce signals (-1, 0, 1)

        Returns:
            Series with technical agreement scores (0-1)

        Example:
            >>> df = pd.DataFrame({
            ...     'signal': [1, -1, 1],
            ...     'rsi_signal': [1, -1, 0],
            ...     'macd_signal': [1, -1, 1]
            ... })
            >>> scorer = SignalConfidenceScorer()
            >>> scores = scorer.technical_agreement_score(df, indicator_cols=['rsi_signal', 'macd_signal'])
        """
        if indicator_cols is None:
            # Find all columns ending with _signal
            indicator_cols = [col for col in df.columns
                            if col.endswith("_signal") and col != signal_col]

        if not indicator_cols:
            logger.warning("No technical indicator columns found, returning zeros")
            return pd.Series(0.0, index=df.index)

        # Count agreements
        scores = []
        for idx, row in df.iterrows():
            primary_signal = row[signal_col]

            if primary_signal == 0:
                scores.append(0.0)
                continue

            # Count indicators agreeing with primary signal
            agreements = sum(
                1 for col in indicator_cols
                if row[col] == primary_signal
            )

            # Score is fraction of indicators agreeing
            score = agreements / len(indicator_cols)
            scores.append(score)

        return pd.Series(scores, index=df.index)

    def volume_confirmation_score(
        self,
        df: pd.DataFrame,
        volume_col: str = "volume",
        price_col: str = "close",
        signal_col: str = "signal",
        lookback: int = 20,
    ) -> pd.Series:
        """Calculate volume confirmation score.

        High volume on signal generation increases confidence.

        Args:
            df: DataFrame with volume and price data
            volume_col: Column name for volume
            price_col: Column name for price
            signal_col: Column name for signal
            lookback: Lookback period for average volume

        Returns:
            Series with volume confirmation scores (0-1)

        Example:
            >>> df = pd.DataFrame({
            ...     'close': [100, 105, 110],
            ...     'volume': [1000, 2000, 1500],
            ...     'signal': [1, 1, -1]
            ... })
            >>> scorer = SignalConfidenceScorer()
            >>> scores = scorer.volume_confirmation_score(df, lookback=2)
        """
        # Calculate average volume
        avg_volume = df[volume_col].rolling(window=lookback).mean()

        # Volume ratio (current / average)
        volume_ratio = df[volume_col] / avg_volume

        # Price change magnitude
        price_change = df[price_col].pct_change().abs()

        # Higher score when:
        # 1. Volume is above average (volume_ratio > 1)
        # 2. Price change is significant
        # 3. There's a non-zero signal

        scores = []
        for idx, row in df.iterrows():
            if row[signal_col] == 0:
                scores.append(0.0)
                continue

            vol_component = min(volume_ratio.loc[idx] / 2, 1.0) if not pd.isna(volume_ratio.loc[idx]) else 0.5
            price_component = min(price_change.loc[idx] * 10, 0.5) if not pd.isna(price_change.loc[idx]) else 0.0

            score = vol_component + price_component
            scores.append(min(score, 1.0))

        return pd.Series(scores, index=df.index)

    def volatility_regime_score(
        self,
        df: pd.DataFrame,
        returns_col: str = "returns",
        lookback: int = 20,
        low_vol_percentile: float = 0.33,
        high_vol_percentile: float = 0.67,
    ) -> pd.Series:
        """Calculate volatility regime score.

        Medium volatility is favorable, very high or low vol reduces confidence.

        Args:
            df: DataFrame with returns
            returns_col: Column name for returns
            lookback: Lookback window for volatility calculation
            low_vol_percentile: Percentile defining low volatility threshold
            high_vol_percentile: Percentile defining high volatility threshold

        Returns:
            Series with volatility regime scores (0-1)

        Example:
            >>> df = pd.DataFrame({'returns': [0.01, -0.02, 0.015, -0.01, 0.02]})
            >>> scorer = SignalConfidenceScorer()
            >>> scores = scorer.volatility_regime_score(df, lookback=3)
        """
        # Calculate rolling volatility
        rolling_vol = df[returns_col].rolling(window=lookback).std()

        # Historical volatility percentiles
        vol_low = rolling_vol.quantile(low_vol_percentile)
        vol_high = rolling_vol.quantile(high_vol_percentile)

        # Score calculation
        def vol_score(vol):
            if pd.isna(vol):
                return 0.5  # Neutral if no data

            if vol < vol_low:
                # Too low volatility (choppy market)
                return 0.4
            elif vol > vol_high:
                # Too high volatility (risky)
                return 0.5
            else:
                # Medium volatility (favorable)
                return 0.9

        return rolling_vol.apply(vol_score)

    def calculate_all_scores(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        prob_col: str = "prob_up",
        volume_col: str = "volume",
        price_col: str = "close",
        returns_col: Optional[str] = None,
        indicator_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calculate all confidence scores and combine them.

        Args:
            df: DataFrame with all necessary columns
            signal_col: Primary signal column
            prob_col: Probability column
            volume_col: Volume column
            price_col: Price column
            returns_col: Returns column (calculated if not provided)
            indicator_cols: List of technical indicator signal columns

        Returns:
            DataFrame with individual scores and combined confidence

        Example:
            >>> df = pd.DataFrame({
            ...     'signal': [1, -1, 1],
            ...     'prob_up': [0.7, 0.3, 0.75],
            ...     'volume': [1000, 1500, 1200],
            ...     'close': [100, 102, 105]
            ... })
            >>> scorer = SignalConfidenceScorer()
            >>> result = scorer.calculate_all_scores(df)
        """
        df_copy = df.copy()

        # Calculate returns if not provided
        if returns_col is None:
            df_copy["returns"] = df_copy[price_col].pct_change()
            returns_col = "returns"

        # Calculate individual scores
        df_copy["technical_score"] = self.technical_agreement_score(
            df_copy, signal_col=signal_col, indicator_cols=indicator_cols
        )

        df_copy["volume_score"] = self.volume_confirmation_score(
            df_copy,
            volume_col=volume_col,
            price_col=price_col,
            signal_col=signal_col,
        )

        df_copy["volatility_score"] = self.volatility_regime_score(
            df_copy, returns_col=returns_col
        )

        # Calculate combined confidence
        confidences = []
        for idx, row in df_copy.iterrows():
            if signal_col not in row or pd.isna(row[signal_col]):
                confidences.append(0.0)
                continue

            confidence = self.score_signal(
                signal=int(row[signal_col]),
                prob=row[prob_col] if prob_col in row else 0.5,
                technical_score=row["technical_score"],
                volume_score=row["volume_score"],
                volatility_score=row["volatility_score"],
            )
            confidences.append(confidence)

        df_copy["confidence"] = confidences

        logger.info("Confidence scores calculated",
                   rows=len(df_copy),
                   avg_confidence=np.mean(confidences),
                   high_confidence_pct=sum(1 for c in confidences if c > 0.7) / len(confidences))

        return df_copy

    def confidence_based_position_size(
        self,
        confidence: float,
        base_position_size: float = 0.1,
        min_confidence: float = 0.5,
        max_position_size: float = 0.2,
    ) -> float:
        """Calculate position size based on confidence score.

        Args:
            confidence: Signal confidence (0-1)
            base_position_size: Base position size (fraction of portfolio)
            min_confidence: Minimum confidence to take position
            max_position_size: Maximum position size allowed

        Returns:
            Position size as fraction of portfolio

        Example:
            >>> scorer = SignalConfidenceScorer()
            >>> position = scorer.confidence_based_position_size(0.8, base_position_size=0.1)
        """
        if confidence < min_confidence:
            return 0.0

        # Scale linearly from min_confidence to 1.0
        scaled_confidence = (confidence - min_confidence) / (1.0 - min_confidence)

        # Position size = base * scaled_confidence
        position_size = base_position_size * (1 + scaled_confidence)

        # Cap at maximum
        return min(position_size, max_position_size)
