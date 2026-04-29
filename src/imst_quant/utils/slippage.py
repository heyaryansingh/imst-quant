"""Slippage estimation utilities for realistic trade execution modeling.

Estimates transaction costs and price impact based on market conditions,
trade size, volatility, and liquidity metrics.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class SlippageEstimate:
    """Container for slippage estimation results."""

    estimated_slippage_bps: float
    price_impact_bps: float
    timing_cost_bps: float
    fixed_cost_bps: float
    confidence_level: str  # "high", "medium", "low"


class SlippageEstimator:
    """Estimates slippage for trade execution based on market conditions."""

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        volatility_multiplier: float = 2.0,
        liquidity_threshold: float = 1e6,
    ):
        """Initialize slippage estimator with base parameters.

        Args:
            base_slippage_bps: Base slippage in basis points (default: 5bps)
            volatility_multiplier: Multiplier for volatility impact (default: 2.0)
            liquidity_threshold: Liquidity threshold for low-volume penalty (default: 1M)
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier
        self.liquidity_threshold = liquidity_threshold

    def estimate(
        self,
        trade_value: float,
        avg_daily_volume: float,
        volatility: float,
        bid_ask_spread_bps: Optional[float] = None,
        is_market_order: bool = True,
    ) -> SlippageEstimate:
        """Estimate slippage for a trade.

        Args:
            trade_value: Trade value in dollars
            avg_daily_volume: Average daily trading volume in dollars
            volatility: Recent volatility (annualized standard deviation)
            bid_ask_spread_bps: Bid-ask spread in bps (optional, estimated if None)
            is_market_order: Whether using market order (vs limit)

        Returns:
            SlippageEstimate with breakdown of cost components
        """
        # 1. Price impact based on trade size relative to volume
        volume_fraction = trade_value / avg_daily_volume if avg_daily_volume > 0 else 1.0
        price_impact = self._compute_price_impact(volume_fraction)

        # 2. Timing cost based on volatility
        timing_cost = self._compute_timing_cost(volatility)

        # 3. Fixed cost (spread + fees)
        if bid_ask_spread_bps is None:
            # Estimate spread from volatility
            bid_ask_spread_bps = self._estimate_spread(volatility, avg_daily_volume)

        fixed_cost = bid_ask_spread_bps / 2.0  # Half-spread for crossing
        if is_market_order:
            fixed_cost += 1.0  # Additional cost for market orders

        # 4. Liquidity penalty for low-volume stocks
        liquidity_penalty = 0.0
        if avg_daily_volume < self.liquidity_threshold:
            liquidity_penalty = 5.0 * (1.0 - avg_daily_volume / self.liquidity_threshold)

        # Total slippage
        total_slippage = price_impact + timing_cost + fixed_cost + liquidity_penalty

        # Confidence based on data quality
        confidence = self._assess_confidence(avg_daily_volume, volatility)

        return SlippageEstimate(
            estimated_slippage_bps=total_slippage,
            price_impact_bps=price_impact,
            timing_cost_bps=timing_cost,
            fixed_cost_bps=fixed_cost,
            confidence_level=confidence,
        )

    def _compute_price_impact(self, volume_fraction: float) -> float:
        """Compute price impact from market impact models.

        Uses square-root impact model: impact ∝ sqrt(volume_fraction)

        Args:
            volume_fraction: Trade size as fraction of daily volume

        Returns:
            Price impact in basis points
        """
        # Square-root model with saturation
        impact_bps = 10.0 * np.sqrt(min(volume_fraction, 1.0))
        return float(impact_bps)

    def _compute_timing_cost(self, volatility: float) -> float:
        """Compute timing cost due to price drift during execution.

        Args:
            volatility: Annualized volatility

        Returns:
            Timing cost in basis points
        """
        # Assume execution takes ~1 minute, scale daily vol
        intraday_vol = volatility / np.sqrt(252 * 390)  # 390 minutes per trading day
        timing_bps = self.volatility_multiplier * intraday_vol * 10000  # Convert to bps
        return float(timing_bps)

    def _estimate_spread(self, volatility: float, avg_daily_volume: float) -> float:
        """Estimate bid-ask spread from volatility and volume.

        Args:
            volatility: Annualized volatility
            avg_daily_volume: Average daily volume

        Returns:
            Estimated spread in basis points
        """
        # Higher volatility → wider spread
        vol_component = volatility * 20.0

        # Lower volume → wider spread
        volume_component = 10.0 / np.log10(max(avg_daily_volume, 1000))

        spread_bps = vol_component + volume_component
        return max(1.0, min(spread_bps, 50.0))  # Clamp to [1, 50] bps

    def _assess_confidence(self, avg_daily_volume: float, volatility: float) -> str:
        """Assess confidence in slippage estimate.

        Args:
            avg_daily_volume: Average daily volume
            volatility: Volatility

        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if avg_daily_volume > 10e6 and volatility < 0.3:
            return "high"
        elif avg_daily_volume > 1e6 and volatility < 0.5:
            return "medium"
        else:
            return "low"

    def estimate_batch(
        self,
        trades_df: pl.DataFrame,
        trade_value_col: str = "trade_value",
        volume_col: str = "avg_daily_volume",
        volatility_col: str = "volatility",
    ) -> pl.DataFrame:
        """Estimate slippage for a batch of trades.

        Args:
            trades_df: DataFrame with trade and market data
            trade_value_col: Column name for trade value
            volume_col: Column name for average daily volume
            volatility_col: Column name for volatility

        Returns:
            DataFrame with added slippage estimate columns
        """
        slippage_estimates = []

        for row in trades_df.iter_rows(named=True):
            estimate = self.estimate(
                trade_value=row[trade_value_col],
                avg_daily_volume=row[volume_col],
                volatility=row[volatility_col],
            )
            slippage_estimates.append(
                {
                    "estimated_slippage_bps": estimate.estimated_slippage_bps,
                    "price_impact_bps": estimate.price_impact_bps,
                    "timing_cost_bps": estimate.timing_cost_bps,
                    "fixed_cost_bps": estimate.fixed_cost_bps,
                    "confidence": estimate.confidence_level,
                }
            )

        slippage_df = pl.DataFrame(slippage_estimates)
        return pl.concat([trades_df, slippage_df], how="horizontal")


def quick_slippage_estimate(
    trade_value: float,
    avg_daily_volume: float,
    volatility: float = 0.2,
) -> float:
    """Quick slippage estimate with default parameters.

    Args:
        trade_value: Trade value in dollars
        avg_daily_volume: Average daily volume in dollars
        volatility: Volatility (default: 0.2 = 20% annual)

    Returns:
        Estimated slippage in basis points
    """
    estimator = SlippageEstimator()
    estimate = estimator.estimate(trade_value, avg_daily_volume, volatility)
    return estimate.estimated_slippage_bps
