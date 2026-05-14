"""Advanced signal strength and quality scoring.

This module provides sophisticated signal strength calculation using multiple
factors including volatility-adjusted momentum, trend persistence, volume
confirmation, and multi-timeframe alignment.

Example:
    >>> from imst_quant.utils.signal_strength import SignalStrengthAnalyzer
    >>> import polars as pl
    >>> prices = pl.Series("close", [100, 102, 105, 104, 108, 110])
    >>> volumes = pl.Series("volume", [1000, 1200, 1500, 900, 1600, 1800])
    >>> analyzer = SignalStrengthAnalyzer()
    >>> strength = analyzer.calculate_strength(prices, volumes)
    >>> print(f"Signal Strength: {strength:.2f}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import polars as pl


@dataclass
class SignalStrength:
    """Signal strength assessment results.

    Attributes:
        overall_strength: Overall signal strength from 0-100.
        momentum_score: Momentum quality score from 0-100.
        trend_score: Trend persistence score from 0-100.
        volume_score: Volume confirmation score from 0-100.
        volatility_score: Volatility-adjusted score from 0-100.
        confidence: Signal confidence level (low, medium, high).
        direction: Signal direction (bullish, bearish, neutral).
    """

    overall_strength: float
    momentum_score: float
    trend_score: float
    volume_score: float
    volatility_score: float
    confidence: str
    direction: str


class SignalStrengthAnalyzer:
    """Analyze signal strength using multiple technical factors.

    Calculates comprehensive signal strength scores incorporating:
    - Volatility-adjusted momentum (price change / ATR)
    - Trend persistence (% of days in same direction)
    - Volume confirmation (volume trend vs price trend)
    - Multi-timeframe alignment (if provided)

    Attributes:
        lookback_period: Period for trend and volume analysis (default: 14).
        momentum_weight: Weight for momentum score (default: 0.30).
        trend_weight: Weight for trend score (default: 0.25).
        volume_weight: Weight for volume score (default: 0.25).
        volatility_weight: Weight for volatility score (default: 0.20).
    """

    def __init__(
        self,
        lookback_period: int = 14,
        momentum_weight: float = 0.30,
        trend_weight: float = 0.25,
        volume_weight: float = 0.25,
        volatility_weight: float = 0.20,
    ):
        """Initialize signal strength analyzer.

        Args:
            lookback_period: Period for trend/volume analysis (default: 14 days).
            momentum_weight: Weight for momentum component (default: 0.30).
            trend_weight: Weight for trend component (default: 0.25).
            volume_weight: Weight for volume component (default: 0.25).
            volatility_weight: Weight for volatility component (default: 0.20).
        """
        self.lookback_period = lookback_period
        self.momentum_weight = momentum_weight
        self.trend_weight = trend_weight
        self.volume_weight = volume_weight
        self.volatility_weight = volatility_weight

    def calculate_atr(self, prices: pl.Series, period: int = 14) -> float:
        """Calculate Average True Range for volatility normalization.

        Args:
            prices: Series of closing prices.
            period: ATR calculation period (default: 14).

        Returns:
            ATR value.
        """
        if len(prices) < period + 1:
            return float(prices.std()) if len(prices) > 1 else 0.01

        # Simplified ATR using high-low ranges (assuming close prices)
        ranges = prices.diff().abs()
        atr = float(ranges.tail(period).mean())

        return max(atr, 0.01)  # Prevent division by zero

    def score_momentum(self, prices: pl.Series) -> Tuple[float, str]:
        """Score momentum quality with volatility adjustment.

        Args:
            prices: Series of prices.

        Returns:
            Tuple of (momentum_score, direction) where score is 0-100.
        """
        if len(prices) < 2:
            return 50.0, "neutral"

        # Calculate price change
        price_change = float(prices[-1] - prices[0])
        price_change_pct = price_change / float(prices[0]) if prices[0] != 0 else 0.0

        # Calculate ATR for normalization
        atr = self.calculate_atr(prices)
        current_price = float(prices[0])
        atr_pct = atr / current_price if current_price > 0 else 0.01

        # Volatility-adjusted momentum
        if atr_pct > 0:
            normalized_momentum = abs(price_change_pct) / atr_pct
        else:
            normalized_momentum = 0.0

        # Score from 0-100 (cap at 5 ATRs = 100)
        momentum_score = min(100, normalized_momentum * 20)

        # Determine direction
        if price_change_pct > 0.01:
            direction = "bullish"
        elif price_change_pct < -0.01:
            direction = "bearish"
        else:
            direction = "neutral"

        return momentum_score, direction

    def score_trend_persistence(self, prices: pl.Series) -> float:
        """Score trend persistence (consistency of direction).

        Args:
            prices: Series of prices.

        Returns:
            Trend persistence score from 0-100.
        """
        if len(prices) < self.lookback_period:
            return 50.0

        # Calculate daily returns
        returns = prices.diff().tail(self.lookback_period)

        # Calculate % of days moving in dominant direction
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        total_days = len(returns) - 1  # Exclude first NaN

        if total_days == 0:
            return 50.0

        # Persistence is % alignment with dominant direction
        dominant_days = max(positive_days, negative_days)
        persistence = dominant_days / total_days

        return float(persistence * 100)

    def score_volume_confirmation(
        self, prices: pl.Series, volumes: Optional[pl.Series] = None
    ) -> float:
        """Score volume confirmation of price moves.

        Args:
            prices: Series of prices.
            volumes: Optional series of volumes.

        Returns:
            Volume confirmation score from 0-100 (50 if no volume data).
        """
        if volumes is None or len(volumes) < self.lookback_period:
            return 50.0  # Neutral score if no volume data

        # Calculate price and volume trends
        price_returns = prices.diff().tail(self.lookback_period)
        volume_changes = volumes.diff().tail(self.lookback_period)

        # Correlation between abs(price change) and volume
        # Higher volume on bigger moves = confirmation
        abs_price_returns = price_returns.abs()

        # Simple correlation measure
        price_mean = float(abs_price_returns.mean())
        volume_mean = float(volume_changes.mean())

        if price_mean == 0 or volume_mean == 0:
            return 50.0

        # Count aligned days (big moves + high volume)
        aligned = 0
        total = 0

        for i in range(len(abs_price_returns)):
            if not abs_price_returns[i] is None and not volume_changes[i] is None:
                price_above = abs_price_returns[i] > price_mean
                volume_above = volume_changes[i] > volume_mean

                if price_above == volume_above:
                    aligned += 1
                total += 1

        if total == 0:
            return 50.0

        confirmation = aligned / total
        return float(confirmation * 100)

    def score_volatility_regime(self, prices: pl.Series) -> float:
        """Score based on current volatility regime.

        Lower volatility = higher score (more stable signal).

        Args:
            prices: Series of prices.

        Returns:
            Volatility score from 0-100.
        """
        if len(prices) < self.lookback_period + 1:
            return 50.0

        # Calculate recent and historical volatility
        returns = prices.pct_change().tail(self.lookback_period)
        current_vol = float(returns.std())

        all_returns = prices.pct_change()
        historical_vol = float(all_returns.std())

        if historical_vol == 0:
            return 50.0

        # Relative volatility (current vs historical)
        vol_ratio = current_vol / historical_vol

        # Score inversely to volatility (lower vol = higher score)
        # 1.0 ratio = 50, < 1.0 = higher, > 1.0 = lower
        if vol_ratio <= 1.0:
            score = 50 + (1.0 - vol_ratio) * 50  # 50-100 range
        else:
            score = 50 / vol_ratio  # 0-50 range

        return min(100, max(0, score))

    def calculate_strength(
        self, prices: pl.Series, volumes: Optional[pl.Series] = None
    ) -> SignalStrength:
        """Calculate comprehensive signal strength.

        Args:
            prices: Series of prices.
            volumes: Optional series of volumes.

        Returns:
            SignalStrength object with component scores and overall strength.
        """
        # Calculate component scores
        momentum_score, direction = self.score_momentum(prices)
        trend_score = self.score_trend_persistence(prices)
        volume_score = self.score_volume_confirmation(prices, volumes)
        volatility_score = self.score_volatility_regime(prices)

        # Weighted overall score
        overall_strength = (
            self.momentum_weight * momentum_score
            + self.trend_weight * trend_score
            + self.volume_weight * volume_score
            + self.volatility_weight * volatility_score
        )

        # Determine confidence level
        if overall_strength >= 70:
            confidence = "high"
        elif overall_strength >= 50:
            confidence = "medium"
        else:
            confidence = "low"

        return SignalStrength(
            overall_strength=overall_strength,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volume_score=volume_score,
            volatility_score=volatility_score,
            confidence=confidence,
            direction=direction,
        )


def compare_signal_strengths(
    strength1: SignalStrength, strength2: SignalStrength, asset1: str = "A", asset2: str = "B"
) -> str:
    """Compare two signal strengths and generate recommendation.

    Args:
        strength1: First signal strength.
        strength2: Second signal strength.
        asset1: Name of first asset (default: "A").
        asset2: Name of second asset (default: "B").

    Returns:
        Human-readable comparison and recommendation.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"SIGNAL STRENGTH COMPARISON: {asset1} vs {asset2}")
    lines.append("=" * 60)
    lines.append("")

    # Overall comparison
    lines.append(f"{asset1} Overall Strength: {strength1.overall_strength:.1f}/100 ({strength1.confidence})")
    lines.append(f"{asset2} Overall Strength: {strength2.overall_strength:.1f}/100 ({strength2.confidence})")
    lines.append("")

    # Component breakdown
    lines.append("Component Breakdown:")
    lines.append(f"  Momentum:    {asset1}={strength1.momentum_score:.1f}  {asset2}={strength2.momentum_score:.1f}")
    lines.append(f"  Trend:       {asset1}={strength1.trend_score:.1f}  {asset2}={strength2.trend_score:.1f}")
    lines.append(f"  Volume:      {asset1}={strength1.volume_score:.1f}  {asset2}={strength2.volume_score:.1f}")
    lines.append(f"  Volatility:  {asset1}={strength1.volatility_score:.1f}  {asset2}={strength2.volatility_score:.1f}")
    lines.append("")

    # Recommendation
    diff = abs(strength1.overall_strength - strength2.overall_strength)

    if diff < 10:
        rec = f"Similar strength - both are viable trades ({strength1.direction} vs {strength2.direction})"
    elif strength1.overall_strength > strength2.overall_strength:
        rec = f"{asset1} shows stronger signal ({strength1.direction} direction)"
    else:
        rec = f"{asset2} shows stronger signal ({strength2.direction} direction)"

    lines.append(f"Recommendation: {rec}")
    lines.append("=" * 60)

    return "\n".join(lines)
