"""Pairs trading recommendation engine.

This module provides functions to find and recommend cointegrated pairs
for pairs trading strategies with actionable entry/exit signals.

Example:
    >>> from imst_quant.utils.pairs_recommender import find_trading_pairs
    >>> import polars as pl
    >>> prices = pl.read_parquet("data/gold/features.parquet")
    >>> pairs = find_trading_pairs(prices, max_pairs=10)
    >>> for pair in pairs:
    ...     print(f"{pair.asset1} vs {pair.asset2}: Score {pair.score:.3f}")
"""

from dataclasses import dataclass
from typing import List, Optional

import polars as pl
from statsmodels.tsa.stattools import coint

from imst_quant.utils.cointegration import (
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    test_cointegration,
)


@dataclass
class PairRecommendation:
    """Recommendation for a trading pair.

    Attributes:
        asset1: First asset in the pair.
        asset2: Second asset in the pair.
        score: Quality score (0-1, higher is better).
        hedge_ratio: Hedge ratio for the spread.
        current_zscore: Current z-score of the spread.
        half_life: Half-life of mean reversion in days.
        p_value: P-value from cointegration test.
        signal: Trading signal ('BUY', 'SELL', 'HOLD').
        entry_threshold: Recommended z-score for entry.
        exit_threshold: Recommended z-score for exit.
    """
    asset1: str
    asset2: str
    score: float
    hedge_ratio: float
    current_zscore: float
    half_life: float
    p_value: float
    signal: str
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5


def find_trading_pairs(
    prices: pl.DataFrame,
    price_col: str = "close",
    asset_col: str = "asset_id",
    date_col: str = "date",
    max_pairs: int = 10,
    min_correlation: float = 0.7,
    max_pvalue: float = 0.05,
    lookback_days: int = 90,
) -> List[PairRecommendation]:
    """Find and rank pairs for pairs trading.

    Args:
        prices: DataFrame with columns [date, asset_id, close].
        price_col: Name of the price column (default: 'close').
        asset_col: Name of the asset identifier column (default: 'asset_id').
        date_col: Name of the date column (default: 'date').
        max_pairs: Maximum number of pairs to return (default: 10).
        min_correlation: Minimum correlation threshold (default: 0.7).
        max_pvalue: Maximum p-value for cointegration (default: 0.05).
        lookback_days: Days to use for analysis (default: 90).

    Returns:
        List of PairRecommendation objects sorted by score (best first).
    """
    # Filter to recent data
    if lookback_days > 0:
        cutoff_date = prices[date_col].max() - pl.duration(days=lookback_days)
        prices = prices.filter(pl.col(date_col) >= cutoff_date)

    # Pivot to wide format: date x assets
    prices_wide = prices.pivot(
        values=price_col,
        index=date_col,
        columns=asset_col,
    ).sort(date_col)

    assets = [c for c in prices_wide.columns if c != date_col]

    if len(assets) < 2:
        return []

    # Calculate correlations
    corr_matrix = prices_wide.select(assets).to_pandas().corr()

    recommendations = []

    # Test all pairs
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i + 1:]:
            # Check correlation threshold
            corr = abs(corr_matrix.loc[asset1, asset2])
            if corr < min_correlation:
                continue

            # Extract price series
            y = prices_wide[asset1].to_numpy()
            x = prices_wide[asset2].to_numpy()

            # Remove NaN values
            mask = ~(pl.Series(y).is_null() | pl.Series(x).is_null()).to_numpy()
            y_clean = y[mask]
            x_clean = x[mask]

            if len(y_clean) < 30:  # Need minimum data
                continue

            # Test cointegration
            try:
                result = test_cointegration(
                    pl.Series("y", y_clean),
                    pl.Series("x", x_clean)
                )

                if result.p_value > max_pvalue:
                    continue

                # Calculate metrics
                hedge_ratio = calculate_hedge_ratio(
                    pl.Series("y", y_clean),
                    pl.Series("x", x_clean)
                )
                spread = calculate_spread(
                    pl.Series("y", y_clean),
                    pl.Series("x", x_clean),
                    hedge_ratio
                )
                zscore = calculate_zscore(spread)
                current_zscore = float(zscore[-1])

                # Calculate half-life
                half_life = _calculate_half_life(spread)

                # Generate signal
                signal = _generate_signal(current_zscore)

                # Calculate quality score
                score = _calculate_pair_score(
                    result.p_value,
                    half_life,
                    abs(current_zscore),
                    corr
                )

                recommendations.append(
                    PairRecommendation(
                        asset1=asset1,
                        asset2=asset2,
                        score=score,
                        hedge_ratio=hedge_ratio,
                        current_zscore=current_zscore,
                        half_life=half_life,
                        p_value=result.p_value,
                        signal=signal,
                    )
                )
            except Exception:
                continue

    # Sort by score and return top pairs
    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations[:max_pairs]


def _calculate_half_life(spread: pl.Series) -> float:
    """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.

    Args:
        spread: Series of spread values.

    Returns:
        Half-life in days.
    """
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    # Remove first NaN
    spread_lag = spread_lag[1:]
    spread_diff = spread_diff[1:]

    # OLS regression: spread_diff ~ spread_lag
    x = spread_lag.to_numpy()
    y = spread_diff.to_numpy()

    # Simple OLS
    x_mean = x.mean()
    y_mean = y.mean()
    beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

    if beta >= 0:
        return float('inf')

    half_life = -np.log(2) / beta
    return max(1.0, float(half_life))


def _generate_signal(zscore: float, entry_threshold: float = 2.0) -> str:
    """Generate trading signal based on z-score.

    Args:
        zscore: Current z-score of the spread.
        entry_threshold: Threshold for entry signals.

    Returns:
        'BUY', 'SELL', or 'HOLD'.
    """
    if zscore > entry_threshold:
        return 'SELL'  # Spread too high, short spread
    elif zscore < -entry_threshold:
        return 'BUY'  # Spread too low, long spread
    else:
        return 'HOLD'


def _calculate_pair_score(
    p_value: float,
    half_life: float,
    abs_zscore: float,
    correlation: float,
) -> float:
    """Calculate quality score for a trading pair.

    Score is between 0-1, with higher being better. Factors:
    - Lower p-value is better (stronger cointegration)
    - Lower half-life is better (faster mean reversion)
    - Higher |z-score| is better (more trading opportunity)
    - Higher correlation is better (more predictable)

    Args:
        p_value: Cointegration p-value.
        half_life: Half-life of mean reversion.
        abs_zscore: Absolute value of current z-score.
        correlation: Correlation coefficient.

    Returns:
        Quality score between 0-1.
    """
    # Cointegration strength (0-1, lower p-value is better)
    coint_score = max(0, 1 - (p_value / 0.05))

    # Mean reversion speed (0-1, prefer 5-30 days)
    if half_life <= 5:
        speed_score = 0.5  # Too fast, may be noise
    elif half_life <= 30:
        speed_score = 1.0
    elif half_life <= 90:
        speed_score = 0.7
    else:
        speed_score = 0.3  # Too slow

    # Trading opportunity (0-1, prefer |z| > 1.5)
    opportunity_score = min(1.0, abs_zscore / 3.0)

    # Correlation strength (0-1)
    corr_score = correlation

    # Weighted average
    score = (
        0.4 * coint_score +
        0.3 * speed_score +
        0.2 * opportunity_score +
        0.1 * corr_score
    )

    return score


def generate_pairs_report(
    pairs: List[PairRecommendation],
    output_path: Optional[str] = None,
) -> str:
    """Generate a formatted report of trading pair recommendations.

    Args:
        pairs: List of PairRecommendation objects.
        output_path: Optional path to save report (default: return as string).

    Returns:
        Formatted report string.
    """
    lines = ["=" * 80]
    lines.append("PAIRS TRADING RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    for i, pair in enumerate(pairs, 1):
        lines.append(f"Pair #{i}: {pair.asset1} vs {pair.asset2}")
        lines.append("-" * 80)
        lines.append(f"  Quality Score:     {pair.score:.3f} / 1.000")
        lines.append(f"  Hedge Ratio:       {pair.hedge_ratio:.4f}")
        lines.append(f"  Current Z-Score:   {pair.current_zscore:+.3f}")
        lines.append(f"  Half-Life:         {pair.half_life:.1f} days")
        lines.append(f"  P-Value:           {pair.p_value:.4f}")
        lines.append(f"  Signal:            {pair.signal}")
        lines.append(f"  Entry Threshold:   ±{pair.entry_threshold:.1f}")
        lines.append(f"  Exit Threshold:    ±{pair.exit_threshold:.1f}")
        lines.append("")

        if pair.signal == 'BUY':
            lines.append(f"  📈 Action: LONG the spread ({pair.asset1} - {pair.hedge_ratio:.3f}*{pair.asset2})")
        elif pair.signal == 'SELL':
            lines.append(f"  📉 Action: SHORT the spread ({pair.asset1} - {pair.hedge_ratio:.3f}*{pair.asset2})")
        else:
            lines.append(f"  ⏸️  Action: Wait for entry signal")
        lines.append("")
        lines.append("")

    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


# Import numpy for calculations
import numpy as np
