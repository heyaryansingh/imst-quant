"""Liquidity scoring and analysis for trading instruments.

Provides comprehensive liquidity metrics including bid-ask spread,
volume depth, market impact, and composite liquidity scores.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class LiquidityMetrics:
    """Container for liquidity metrics."""
    bid_ask_spread: float
    effective_spread: float
    volume: float
    turnover_ratio: float
    depth: float
    resilience: float
    composite_score: float


class LiquidityScorer:
    """Calculate comprehensive liquidity scores for trading instruments."""

    @staticmethod
    def bid_ask_spread_score(
        bid: float,
        ask: float,
        mid_price: Optional[float] = None
    ) -> float:
        """Calculate bid-ask spread score.

        Args:
            bid: Best bid price
            ask: Best ask price
            mid_price: Optional mid price (default: (bid+ask)/2)

        Returns:
            Spread as percentage of mid price (lower is more liquid)
        """
        mid = mid_price or (bid + ask) / 2
        if mid <= 0:
            return float('inf')

        spread_pct = (ask - bid) / mid
        return spread_pct

    @staticmethod
    def effective_spread_score(
        execution_price: float,
        mid_price: float,
        side: str
    ) -> float:
        """Calculate effective spread from actual execution.

        Args:
            execution_price: Actual execution price
            mid_price: Mid price at time of trade
            side: 'buy' or 'sell'

        Returns:
            Effective spread as percentage (lower is better)
        """
        if side == 'buy':
            effective_spread = 2 * (execution_price - mid_price) / mid_price
        else:
            effective_spread = 2 * (mid_price - execution_price) / mid_price

        return abs(effective_spread)

    @staticmethod
    def volume_score(
        volume: float,
        avg_volume: float,
        lookback_std: float
    ) -> float:
        """Calculate volume-based liquidity score.

        Args:
            volume: Current or period volume
            avg_volume: Average historical volume
            lookback_std: Standard deviation of volume

        Returns:
            Z-score of volume (higher is more liquid)
        """
        if lookback_std <= 0:
            return 0.0

        z_score = (volume - avg_volume) / lookback_std
        return z_score

    @staticmethod
    def turnover_ratio(
        volume: float,
        shares_outstanding: float,
        price: float
    ) -> float:
        """Calculate turnover ratio.

        Args:
            volume: Trading volume (shares)
            shares_outstanding: Total shares outstanding
            price: Current price

        Returns:
            Turnover ratio (higher is more liquid)
        """
        if shares_outstanding <= 0:
            return 0.0

        return volume / shares_outstanding

    @staticmethod
    def market_depth_score(
        order_book: pd.DataFrame,
        price_levels: int = 5
    ) -> float:
        """Calculate market depth from order book.

        Args:
            order_book: DataFrame with columns [price, size, side]
            price_levels: Number of price levels to consider

        Returns:
            Average depth across price levels
        """
        bids = order_book[order_book['side'] == 'bid'].nlargest(price_levels, 'price')
        asks = order_book[order_book['side'] == 'ask'].nsmallest(price_levels, 'price')

        bid_depth = bids['size'].sum()
        ask_depth = asks['size'].sum()

        avg_depth = (bid_depth + ask_depth) / 2
        return avg_depth

    @staticmethod
    def price_resilience(
        prices: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> float:
        """Calculate price resilience after large trades.

        Measures how quickly price returns to equilibrium after shock.

        Args:
            prices: Time series of prices
            volumes: Time series of volumes
            window: Rolling window for analysis

        Returns:
            Resilience score (higher is more resilient)
        """
        # Identify large trades (above 75th percentile)
        volume_threshold = volumes.quantile(0.75)
        large_trades = volumes > volume_threshold

        if large_trades.sum() < 5:
            return 0.0

        # Calculate price impact and recovery
        impacts = []
        for idx in np.where(large_trades)[0]:
            if idx + window >= len(prices):
                continue

            pre_price = prices.iloc[idx]
            post_price = prices.iloc[idx + 1]
            recovery_price = prices.iloc[idx + window]

            # Impact
            impact = abs(post_price - pre_price) / pre_price

            # Recovery (how much of impact was recovered)
            if impact > 0:
                recovery = 1 - abs(recovery_price - pre_price) / (abs(post_price - pre_price) + 1e-10)
                impacts.append(max(0, recovery))

        if not impacts:
            return 0.0

        return np.mean(impacts)


class AmihudIlliquidity:
    """Amihud (2002) illiquidity measure.

    Measures price impact per dollar of trading volume.
    """

    @staticmethod
    def daily_illiquidity(
        returns: pd.Series,
        volumes: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """Calculate daily Amihud illiquidity.

        Args:
            returns: Daily returns
            volumes: Daily volumes
            prices: Daily prices

        Returns:
            Daily illiquidity scores (lower is more liquid)
        """
        dollar_volume = volumes * prices
        illiquidity = returns.abs() / (dollar_volume + 1e-10)

        return illiquidity

    @staticmethod
    def average_illiquidity(
        returns: pd.Series,
        volumes: pd.Series,
        prices: pd.Series,
        window: int = 252
    ) -> float:
        """Calculate average Amihud illiquidity over period.

        Args:
            returns: Daily returns
            volumes: Daily volumes
            prices: Daily prices
            window: Rolling window size

        Returns:
            Average illiquidity (lower is more liquid)
        """
        daily_illiq = AmihudIlliquidity.daily_illiquidity(returns, volumes, prices)
        return daily_illiq.tail(window).mean()


class LiquidityProvider:
    """Assess liquidity provision and market making metrics."""

    @staticmethod
    def quote_stability(
        bid_prices: pd.Series,
        ask_prices: pd.Series,
        window: int = 100
    ) -> float:
        """Measure quote stability.

        Args:
            bid_prices: Time series of best bid prices
            ask_prices: Time series of best ask prices
            window: Window size for analysis

        Returns:
            Quote stability score (higher is more stable)
        """
        mid_prices = (bid_prices + ask_prices) / 2
        spreads = ask_prices - bid_prices

        # Calculate coefficient of variation
        mid_cv = mid_prices.tail(window).std() / (mid_prices.tail(window).mean() + 1e-10)
        spread_cv = spreads.tail(window).std() / (spreads.tail(window).mean() + 1e-10)

        # Lower CV = higher stability
        stability = 1 / (1 + mid_cv + spread_cv)

        return stability

    @staticmethod
    def time_at_best(
        quote_updates: pd.DataFrame,
        window: int = 1000
    ) -> float:
        """Calculate percentage of time at best bid/offer.

        Args:
            quote_updates: DataFrame with columns [timestamp, is_best_bid, is_best_offer]
            window: Number of updates to consider

        Returns:
            Fraction of time at best quote (higher is better liquidity provision)
        """
        recent = quote_updates.tail(window)
        time_at_best = (
            recent['is_best_bid'].sum() + recent['is_best_offer'].sum()
        ) / (2 * len(recent))

        return time_at_best


class CompositeLiquidityScore:
    """Calculate composite liquidity score from multiple metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize with custom weights.

        Args:
            weights: Optional dict mapping metric names to weights
        """
        self.weights = weights or {
            'spread': 0.25,
            'volume': 0.20,
            'depth': 0.20,
            'resilience': 0.15,
            'turnover': 0.10,
            'amihud': 0.10
        }

    def calculate_score(
        self,
        spread_bps: float,
        volume_zscore: float,
        depth: float,
        resilience: float,
        turnover: float,
        amihud_illiq: float
    ) -> float:
        """Calculate composite liquidity score.

        Args:
            spread_bps: Bid-ask spread in basis points
            volume_zscore: Z-score of volume
            depth: Market depth score
            resilience: Price resilience score
            turnover: Turnover ratio
            amihud_illiq: Amihud illiquidity measure

        Returns:
            Composite score (0-100, higher is more liquid)
        """
        # Normalize each metric to 0-1 scale
        # Spread: lower is better, cap at 50 bps
        spread_score = max(0, 1 - min(spread_bps / 50, 1))

        # Volume: higher z-score is better, normalize to 0-1
        volume_score = max(0, min(1, (volume_zscore + 3) / 6))

        # Depth: normalize assuming 1M is excellent
        depth_score = min(depth / 1e6, 1)

        # Resilience: already 0-1
        resilience_score = max(0, min(resilience, 1))

        # Turnover: higher is better, cap at 20%
        turnover_score = min(turnover / 0.20, 1)

        # Amihud: lower is better (illiquidity), cap at 1e-6
        amihud_score = max(0, 1 - min(amihud_illiq / 1e-6, 1))

        # Weighted composite
        composite = (
            self.weights['spread'] * spread_score +
            self.weights['volume'] * volume_score +
            self.weights['depth'] * depth_score +
            self.weights['resilience'] * resilience_score +
            self.weights['turnover'] * turnover_score +
            self.weights['amihud'] * amihud_score
        )

        # Scale to 0-100
        return composite * 100

    def score_dataframe(
        self,
        metrics_df: pd.DataFrame
    ) -> pd.Series:
        """Score multiple instruments from metrics DataFrame.

        Args:
            metrics_df: DataFrame with columns for each metric

        Returns:
            Series of composite scores indexed by instrument
        """
        scores = metrics_df.apply(
            lambda row: self.calculate_score(
                spread_bps=row.get('spread_bps', 0),
                volume_zscore=row.get('volume_zscore', 0),
                depth=row.get('depth', 0),
                resilience=row.get('resilience', 0),
                turnover=row.get('turnover', 0),
                amihud_illiq=row.get('amihud_illiq', 0)
            ),
            axis=1
        )

        return scores


def liquidity_classification(score: float) -> str:
    """Classify liquidity based on composite score.

    Args:
        score: Composite liquidity score (0-100)

    Returns:
        Classification string
    """
    if score >= 80:
        return "Highly Liquid"
    elif score >= 60:
        return "Liquid"
    elif score >= 40:
        return "Moderately Liquid"
    elif score >= 20:
        return "Illiquid"
    else:
        return "Highly Illiquid"


def analyze_liquidity_over_time(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """Analyze liquidity trends over time.

    Args:
        prices: DataFrame of prices (dates x assets)
        volumes: DataFrame of volumes (dates x assets)
        window: Rolling window size

    Returns:
        DataFrame with liquidity metrics over time
    """
    results = []

    for asset in prices.columns:
        asset_prices = prices[asset].dropna()
        asset_volumes = volumes[asset].dropna()

        # Align
        common_idx = asset_prices.index.intersection(asset_volumes.index)
        asset_prices = asset_prices.loc[common_idx]
        asset_volumes = asset_volumes.loc[common_idx]

        if len(asset_prices) < window:
            continue

        # Calculate returns
        returns = asset_prices.pct_change().dropna()

        # Rolling liquidity metrics
        for i in range(window, len(asset_prices)):
            window_prices = asset_prices.iloc[i-window:i]
            window_volumes = asset_volumes.iloc[i-window:i]
            window_returns = returns.iloc[i-window:i]

            # Amihud illiquidity
            amihud = AmihudIlliquidity.average_illiquidity(
                window_returns, window_volumes, window_prices
            )

            # Volume z-score
            vol_mean = window_volumes.mean()
            vol_std = window_volumes.std()
            current_vol = asset_volumes.iloc[i]
            vol_zscore = (current_vol - vol_mean) / (vol_std + 1e-10)

            # Turnover (assuming shares outstanding = 100M)
            turnover = current_vol / 100e6

            results.append({
                'date': asset_prices.index[i],
                'asset': asset,
                'amihud_illiq': amihud,
                'volume_zscore': vol_zscore,
                'turnover': turnover,
                'price': asset_prices.iloc[i]
            })

    return pd.DataFrame(results)


def rank_by_liquidity(
    instruments: List[str],
    scores: pd.Series
) -> pd.DataFrame:
    """Rank instruments by liquidity score.

    Args:
        instruments: List of instrument names
        scores: Series of liquidity scores

    Returns:
        DataFrame with rankings
    """
    ranking = pd.DataFrame({
        'instrument': instruments,
        'liquidity_score': scores.values,
        'classification': scores.apply(liquidity_classification).values
    })

    ranking = ranking.sort_values('liquidity_score', ascending=False)
    ranking['rank'] = range(1, len(ranking) + 1)

    return ranking[['rank', 'instrument', 'liquidity_score', 'classification']]
