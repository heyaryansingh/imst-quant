"""Market regime transition detection and analysis.

Identifies and analyzes transitions between different market regimes:
- Trend to consolidation
- Bull to bear market
- High to low volatility
- Risk-on to risk-off

Helps adapt trading strategies based on regime changes.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm


class RegimeType(Enum):
    """Market regime classifications."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


@dataclass
class RegimeTransition:
    """Single regime change event."""
    date: datetime
    from_regime: RegimeType
    to_regime: RegimeType
    confidence: float  # 0-1
    trigger_metric: str
    trigger_value: float


@dataclass
class RegimeStats:
    """Statistics for a specific regime period."""
    regime: RegimeType
    start_date: datetime
    end_date: datetime
    duration_days: int
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float


def detect_volatility_regimes(
    returns: pd.Series,
    window: int = 20,
    threshold_percentile: float = 0.7
) -> pd.Series:
    """Classify periods into high/low volatility regimes.

    Args:
        returns: Daily returns series
        window: Rolling window for volatility calculation
        threshold_percentile: Percentile to separate high/low vol (0-1)

    Returns:
        Series with regime labels: HIGH_VOL or LOW_VOL
    """
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

    # Define threshold
    threshold = rolling_vol.quantile(threshold_percentile)

    # Classify regimes
    regimes = pd.Series(index=returns.index, dtype=str)
    regimes[rolling_vol > threshold] = RegimeType.HIGH_VOL.value
    regimes[rolling_vol <= threshold] = RegimeType.LOW_VOL.value

    return regimes


def detect_trend_regimes(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50
) -> pd.Series:
    """Classify periods into trending vs. consolidation regimes.

    Uses dual moving averages and ADX-like logic.

    Args:
        prices: Price series
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        Series with regime labels: BULL_TREND, BEAR_TREND, or CONSOLIDATION
    """
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()

    # Calculate average directional movement
    high_low = prices.rolling(window=short_window).apply(lambda x: x.max() - x.min())
    atr = prices.pct_change().abs().rolling(window=short_window).mean()
    trend_strength = high_low / (atr * short_window)

    regimes = pd.Series(index=prices.index, dtype=str)

    # Strong trend when MA spread is wide AND directional movement is high
    strong_trend_mask = trend_strength > trend_strength.quantile(0.6)

    bull_mask = (sma_short > sma_long) & strong_trend_mask
    bear_mask = (sma_short < sma_long) & strong_trend_mask

    regimes[bull_mask] = RegimeType.BULL_TREND.value
    regimes[bear_mask] = RegimeType.BEAR_TREND.value
    regimes[~(bull_mask | bear_mask)] = RegimeType.CONSOLIDATION.value

    return regimes


def detect_risk_sentiment_regimes(
    spy_returns: pd.Series,
    vix: pd.Series,
    high_yield_spread: Optional[pd.Series] = None
) -> pd.Series:
    """Classify market into risk-on vs. risk-off regimes.

    Uses SPY performance, VIX levels, and optionally credit spreads.

    Args:
        spy_returns: S&P 500 returns
        vix: VIX index values
        high_yield_spread: Optional high-yield credit spread

    Returns:
        Series with RISK_ON or RISK_OFF labels
    """
    # VIX threshold (above 20 = fear, below 15 = complacency)
    vix_regime = pd.Series(index=vix.index, dtype=str)
    vix_regime[vix > 20] = "fear"
    vix_regime[vix < 15] = "calm"
    vix_regime[(vix >= 15) & (vix <= 20)] = "neutral"

    # SPY momentum (10-day MA return)
    spy_momentum = spy_returns.rolling(window=10).mean()

    regimes = pd.Series(index=spy_returns.index, dtype=str)

    # Risk-on: low VIX + positive momentum
    risk_on_mask = (vix_regime == "calm") & (spy_momentum > 0)

    # Risk-off: high VIX or negative momentum
    risk_off_mask = (vix_regime == "fear") | (spy_momentum < -0.01)

    if high_yield_spread is not None:
        # Widening spreads = risk-off
        spread_change = high_yield_spread.pct_change(periods=20)
        risk_off_mask = risk_off_mask | (spread_change > 0.1)

    regimes[risk_on_mask] = RegimeType.RISK_ON.value
    regimes[risk_off_mask] = RegimeType.RISK_OFF.value
    regimes[~(risk_on_mask | risk_off_mask)] = "neutral"

    return regimes


def identify_transitions(
    regimes: pd.Series,
    min_regime_duration: int = 5
) -> List[RegimeTransition]:
    """Extract regime transitions from a regime classification series.

    Args:
        regimes: Series of regime labels
        min_regime_duration: Minimum days for a regime to be valid (reduces noise)

    Returns:
        List of RegimeTransition objects
    """
    # Filter out short-lived regimes
    regime_runs = (regimes != regimes.shift()).cumsum()
    regime_lengths = regimes.groupby(regime_runs).transform('count')
    filtered_regimes = regimes.copy()
    filtered_regimes[regime_lengths < min_regime_duration] = np.nan
    filtered_regimes = filtered_regimes.fillna(method='ffill')

    transitions = []
    prev_regime = None

    for date, regime in filtered_regimes.items():
        if prev_regime is not None and regime != prev_regime:
            try:
                from_regime = RegimeType(prev_regime)
                to_regime = RegimeType(regime)

                transitions.append(RegimeTransition(
                    date=date,
                    from_regime=from_regime,
                    to_regime=to_regime,
                    confidence=1.0,  # Simple classification has full confidence
                    trigger_metric="classification",
                    trigger_value=0.0,
                ))
            except ValueError:
                # Skip if regime value doesn't match enum
                pass

        prev_regime = regime

    return transitions


def regime_performance_stats(
    returns: pd.Series,
    regimes: pd.Series
) -> Dict[str, RegimeStats]:
    """Compute performance metrics for each regime.

    Args:
        returns: Daily returns
        regimes: Regime classification series

    Returns:
        Dictionary mapping regime names to RegimeStats
    """
    results = {}

    for regime_value in regimes.unique():
        if pd.isna(regime_value):
            continue

        mask = regimes == regime_value
        regime_returns = returns[mask]

        if len(regime_returns) == 0:
            continue

        # Find regime periods
        regime_blocks = mask.astype(int).diff().fillna(0)
        starts = regime_blocks[regime_blocks == 1].index
        ends = regime_blocks[regime_blocks == -1].index

        # Handle edge cases
        if mask.iloc[0]:
            starts = [regime_returns.index[0]] + list(starts)
        if mask.iloc[-1]:
            ends = list(ends) + [regime_returns.index[-1]]

        # Compute stats across all periods
        avg_return = regime_returns.mean()
        volatility = regime_returns.std() * np.sqrt(252)
        sharpe = (avg_return * 252) / volatility if volatility > 0 else 0.0

        # Max drawdown
        cum_returns = (1 + regime_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        # Use first/last dates
        start_date = starts[0] if starts else regime_returns.index[0]
        end_date = ends[-1] if ends else regime_returns.index[-1]
        duration = (end_date - start_date).days

        try:
            regime_type = RegimeType(regime_value)
        except ValueError:
            continue

        results[regime_value] = RegimeStats(
            regime=regime_type,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration,
            avg_return=avg_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )

    return results


def transition_impact_analysis(
    returns: pd.Series,
    transitions: List[RegimeTransition],
    forward_days: int = 20
) -> pd.DataFrame:
    """Analyze return patterns following regime transitions.

    Args:
        returns: Daily returns
        transitions: List of regime transitions
        forward_days: Days to analyze post-transition

    Returns:
        DataFrame with transition details and forward returns
    """
    results = []

    for trans in transitions:
        # Find returns following transition
        trans_idx = returns.index.get_loc(trans.date)

        if trans_idx + forward_days >= len(returns):
            continue

        forward_rets = returns.iloc[trans_idx:trans_idx + forward_days]

        results.append({
            "date": trans.date,
            "from_regime": trans.from_regime.value,
            "to_regime": trans.to_regime.value,
            "forward_1d": forward_rets.iloc[1] if len(forward_rets) > 1 else np.nan,
            "forward_5d": forward_rets.iloc[1:6].sum() if len(forward_rets) > 5 else np.nan,
            "forward_20d": forward_rets.iloc[1:21].sum() if len(forward_rets) > 20 else np.nan,
            "forward_vol_20d": forward_rets.iloc[1:21].std() * np.sqrt(252) if len(forward_rets) > 20 else np.nan,
        })

    return pd.DataFrame(results)


def detect_breakout_regimes(
    prices: pd.Series,
    volume: pd.Series,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    volume_threshold: float = 1.5
) -> pd.Series:
    """Detect breakout regimes using Bollinger Bands and volume.

    Args:
        prices: Price series
        volume: Volume series
        bollinger_window: Window for BB calculation
        bollinger_std: Number of standard deviations for bands
        volume_threshold: Multiple of average volume to confirm breakout

    Returns:
        Series with BREAKOUT or CONSOLIDATION labels
    """
    # Bollinger Bands
    sma = prices.rolling(window=bollinger_window).mean()
    std = prices.rolling(window=bollinger_window).std()
    upper_band = sma + bollinger_std * std
    lower_band = sma - bollinger_std * std

    # Volume spike
    avg_volume = volume.rolling(window=bollinger_window).mean()
    volume_spike = volume > (avg_volume * volume_threshold)

    # Breakout: price crosses band + volume spike
    upper_breakout = (prices > upper_band) & volume_spike
    lower_breakout = (prices < lower_band) & volume_spike

    regimes = pd.Series(RegimeType.CONSOLIDATION.value, index=prices.index)
    regimes[upper_breakout | lower_breakout] = RegimeType.BREAKOUT.value

    return regimes
