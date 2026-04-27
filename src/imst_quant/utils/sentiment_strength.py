"""Sentiment Strength Analyzer: Advanced sentiment signal analysis and scoring.

This module provides sophisticated sentiment strength analysis by examining:
- Signal consistency across time windows
- Volume-weighted sentiment trends
- Sentiment volatility and stability
- Agreement/disagreement metrics across sources
- Sentiment momentum and acceleration
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class SentimentStrengthAnalyzer:
    """Analyzes sentiment signal strength and quality for trading decisions."""

    def __init__(
        self,
        short_window: int = 3,
        medium_window: int = 7,
        long_window: int = 14,
        min_volume: int = 10,
    ):
        """Initialize sentiment strength analyzer.

        Args:
            short_window: Short-term window for trend analysis (days)
            medium_window: Medium-term window (days)
            long_window: Long-term window (days)
            min_volume: Minimum post volume for reliable signal
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.min_volume = min_volume

    def analyze_sentiment_data(
        self, df: pd.DataFrame, asset_id: str
    ) -> Dict[str, float]:
        """Perform comprehensive sentiment strength analysis.

        Args:
            df: DataFrame with columns: date, asset_id, polarity, volume, subjectivity
            asset_id: Asset identifier to analyze

        Returns:
            Dictionary containing:
            - strength_score: Overall signal strength (0-100)
            - consistency: Cross-window consistency (0-1)
            - momentum: Sentiment momentum indicator (-1 to 1)
            - stability: Signal stability score (0-1)
            - conviction: High-conviction signal indicator (0-1)
            - volume_quality: Volume-based quality score (0-1)
        """
        df_asset = df[df["asset_id"] == asset_id].copy()

        if len(df_asset) < self.long_window:
            logger.warning(
                "insufficient_data",
                asset=asset_id,
                rows=len(df_asset),
                required=self.long_window,
            )
            return self._empty_result()

        df_asset = df_asset.sort_values("date")

        # Calculate window-based metrics
        consistency = self._calculate_consistency(df_asset)
        momentum = self._calculate_momentum(df_asset)
        stability = self._calculate_stability(df_asset)
        volume_quality = self._calculate_volume_quality(df_asset)
        conviction = self._calculate_conviction(df_asset)

        # Compute overall strength score
        strength_score = self._compute_strength_score(
            consistency=consistency,
            momentum=abs(momentum),
            stability=stability,
            volume_quality=volume_quality,
            conviction=conviction,
        )

        return {
            "strength_score": round(strength_score, 2),
            "consistency": round(consistency, 3),
            "momentum": round(momentum, 3),
            "stability": round(stability, 3),
            "conviction": round(conviction, 3),
            "volume_quality": round(volume_quality, 3),
        }

    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate cross-window sentiment consistency.

        High consistency means short/medium/long windows agree on direction.
        """
        if len(df) < self.long_window:
            return 0.0

        recent = df.tail(self.long_window)

        short_avg = recent.tail(self.short_window)["polarity"].mean()
        medium_avg = recent.tail(self.medium_window)["polarity"].mean()
        long_avg = recent["polarity"].mean()

        # All windows should have same sign
        signs = [np.sign(short_avg), np.sign(medium_avg), np.sign(long_avg)]
        if len(set(signs)) == 1 and signs[0] != 0:
            # Calculate magnitude agreement
            values = [abs(short_avg), abs(medium_avg), abs(long_avg)]
            std_dev = np.std(values)
            mean_val = np.mean(values)
            consistency = 1.0 - min(std_dev / (mean_val + 1e-6), 1.0)
        else:
            # Mixed signals = low consistency
            consistency = 0.0

        return consistency

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate sentiment momentum (rate of change).

        Positive momentum = sentiment improving
        Negative momentum = sentiment declining
        """
        if len(df) < self.medium_window:
            return 0.0

        recent = df.tail(self.medium_window)

        # Use linear regression to determine trend
        x = np.arange(len(recent))
        y = recent["polarity"].values

        if len(x) < 2:
            return 0.0

        slope = np.polyfit(x, y, 1)[0]

        # Normalize to -1 to 1 range
        momentum = np.clip(slope * 10, -1, 1)

        return float(momentum)

    def _calculate_stability(self, df: pd.DataFrame) -> float:
        """Calculate sentiment stability (inverse of volatility).

        High stability = consistent sentiment
        Low stability = erratic sentiment
        """
        if len(df) < self.medium_window:
            return 0.0

        recent = df.tail(self.medium_window)

        # Use coefficient of variation (inverse)
        std_dev = recent["polarity"].std()
        mean_abs = abs(recent["polarity"].mean())

        if mean_abs < 0.01:
            return 0.0

        cv = std_dev / mean_abs
        stability = 1.0 / (1.0 + cv)

        return stability

    def _calculate_volume_quality(self, df: pd.DataFrame) -> float:
        """Calculate volume-based quality score.

        Higher volume = more reliable signal
        """
        if len(df) < self.short_window:
            return 0.0

        recent = df.tail(self.short_window)
        avg_volume = recent["volume"].mean()

        # Sigmoid function to map volume to 0-1 quality score
        quality = 1.0 / (1.0 + np.exp(-0.1 * (avg_volume - self.min_volume)))

        return quality

    def _calculate_conviction(self, df: pd.DataFrame) -> float:
        """Calculate conviction score based on extreme sentiment + low subjectivity.

        High conviction = strong sentiment + factual (not subjective)
        """
        if len(df) < self.short_window:
            return 0.0

        recent = df.tail(self.short_window)

        avg_polarity_abs = abs(recent["polarity"].mean())
        avg_subjectivity = recent["subjectivity"].mean()

        # High polarity + low subjectivity = high conviction
        # Normalize: polarity in [0, 1], subjectivity in [0, 1]
        conviction = avg_polarity_abs * (1.0 - avg_subjectivity)

        return conviction

    def _compute_strength_score(
        self,
        consistency: float,
        momentum: float,
        stability: float,
        volume_quality: float,
        conviction: float,
    ) -> float:
        """Compute overall sentiment strength score (0-100).

        Weighted combination of all factors.
        """
        weights = {
            "consistency": 0.25,
            "momentum": 0.15,
            "stability": 0.20,
            "volume_quality": 0.20,
            "conviction": 0.20,
        }

        score = (
            weights["consistency"] * consistency
            + weights["momentum"] * momentum
            + weights["stability"] * stability
            + weights["volume_quality"] * volume_quality
            + weights["conviction"] * conviction
        )

        return score * 100

    def _empty_result(self) -> Dict[str, float]:
        """Return empty result for insufficient data."""
        return {
            "strength_score": 0.0,
            "consistency": 0.0,
            "momentum": 0.0,
            "stability": 0.0,
            "conviction": 0.0,
            "volume_quality": 0.0,
        }

    def batch_analyze(
        self, df: pd.DataFrame, asset_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Analyze sentiment strength for multiple assets.

        Args:
            df: Sentiment data with asset_id column
            asset_ids: List of assets to analyze (None = all)

        Returns:
            DataFrame with one row per asset containing strength metrics
        """
        if asset_ids is None:
            asset_ids = df["asset_id"].unique().tolist()

        results = []

        for asset_id in asset_ids:
            metrics = self.analyze_sentiment_data(df, asset_id)
            metrics["asset_id"] = asset_id
            results.append(metrics)

        df_results = pd.DataFrame(results)

        # Sort by strength score
        df_results = df_results.sort_values("strength_score", ascending=False)

        return df_results

    def export_report(
        self, df: pd.DataFrame, output_path: Path, top_n: int = 20
    ) -> None:
        """Export sentiment strength report to CSV.

        Args:
            df: Results from batch_analyze()
            output_path: Path to save CSV report
            top_n: Include top N strongest signals
        """
        top_signals = df.head(top_n)

        top_signals.to_csv(output_path, index=False)

        logger.info(
            "sentiment_report_exported",
            path=str(output_path),
            assets=len(top_signals),
        )


def analyze_sentiment_file(
    sentiment_path: Path,
    asset_id: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Analyze sentiment strength from a parquet file.

    Args:
        sentiment_path: Path to sentiment aggregates parquet
        asset_id: Specific asset to analyze (None = all assets)
        output_path: Optional path to export CSV report

    Returns:
        DataFrame with sentiment strength analysis
    """
    logger.info("loading_sentiment_data", path=str(sentiment_path))

    df = pd.read_parquet(sentiment_path)

    # Ensure required columns
    required_cols = ["date", "asset_id", "polarity", "volume", "subjectivity"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    analyzer = SentimentStrengthAnalyzer()

    if asset_id:
        # Single asset analysis
        result = analyzer.analyze_sentiment_data(df, asset_id)
        df_result = pd.DataFrame([result])
        df_result["asset_id"] = asset_id
    else:
        # Batch analysis
        df_result = analyzer.batch_analyze(df)

    if output_path:
        analyzer.export_report(df_result, output_path)

    return df_result
