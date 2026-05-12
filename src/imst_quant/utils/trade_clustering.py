"""
Trade clustering and pattern analysis.

Provides methods to identify clusters of trades, analyze trade patterns,
and detect behavioral patterns in trading activity.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def cluster_trades_by_features(
    trades: pd.DataFrame,
    features: List[str],
    n_clusters: int = 5,
    method: str = 'kmeans',
) -> pd.DataFrame:
    """
    Cluster trades based on specified features.

    Args:
        trades: DataFrame with trade data
        features: List of feature columns to use for clustering
        n_clusters: Number of clusters (for kmeans)
        method: Clustering method ('kmeans' or 'dbscan')

    Returns:
        DataFrame with cluster assignments
    """
    # Extract features and scale
    X = trades[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Unknown method: {method}")

    clusters = clusterer.fit_predict(X_scaled)

    # Add cluster assignments
    result = trades.copy()
    result['cluster'] = clusters

    return result


def identify_trade_sequences(
    trades: pd.DataFrame,
    time_threshold: pd.Timedelta = pd.Timedelta(hours=1),
    symbol_col: str = 'symbol',
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Identify sequences of related trades (e.g., scaling in/out).

    Args:
        trades: DataFrame with trade data
        time_threshold: Maximum time between trades in a sequence
        symbol_col: Name of symbol column
        time_col: Name of timestamp column

    Returns:
        DataFrame with sequence_id assigned to each trade
    """
    trades = trades.sort_values(time_col)
    trades['sequence_id'] = -1

    sequence_id = 0

    for symbol in trades[symbol_col].unique():
        symbol_trades = trades[trades[symbol_col] == symbol].copy()

        if len(symbol_trades) == 0:
            continue

        current_sequence = sequence_id
        prev_time = None

        for idx in symbol_trades.index:
            current_time = trades.loc[idx, time_col]

            if prev_time is None or (current_time - prev_time) > time_threshold:
                current_sequence = sequence_id
                sequence_id += 1

            trades.loc[idx, 'sequence_id'] = current_sequence
            prev_time = current_time

    return trades


def analyze_trade_timing_patterns(
    trades: pd.DataFrame,
    time_col: str = 'timestamp',
) -> Dict[str, pd.Series]:
    """
    Analyze patterns in trade timing (hour of day, day of week).

    Args:
        trades: DataFrame with trade data
        time_col: Name of timestamp column

    Returns:
        Dictionary with timing pattern analysis
    """
    trades = trades.copy()
    trades['hour'] = trades[time_col].dt.hour
    trades['day_of_week'] = trades[time_col].dt.dayofweek
    trades['week_of_month'] = (trades[time_col].dt.day - 1) // 7 + 1

    patterns = {
        'hour_distribution': trades.groupby('hour').size(),
        'day_of_week_distribution': trades.groupby('day_of_week').size(),
        'week_of_month_distribution': trades.groupby('week_of_month').size(),
    }

    return patterns


def detect_revenge_trading(
    trades: pd.DataFrame,
    loss_threshold: float = -0.02,
    time_window: pd.Timedelta = pd.Timedelta(hours=2),
    pnl_col: str = 'pnl',
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Detect potential revenge trading patterns.

    Flags trades that occur shortly after losses and may indicate emotional trading.

    Args:
        trades: DataFrame with trade data
        loss_threshold: PnL threshold to consider a loss
        time_window: Time window after loss to check for revenge trading
        pnl_col: Name of PnL column
        time_col: Name of timestamp column

    Returns:
        DataFrame with potential revenge trades flagged
    """
    trades = trades.sort_values(time_col).copy()
    trades['potential_revenge_trade'] = False

    for i in range(1, len(trades)):
        prev_pnl = trades.iloc[i - 1][pnl_col]
        current_time = trades.iloc[i][time_col]
        prev_time = trades.iloc[i - 1][time_col]

        time_diff = current_time - prev_time

        # Check if previous trade was a loss and current trade is within time window
        if prev_pnl < loss_threshold and time_diff < time_window:
            trades.iloc[i, trades.columns.get_loc('potential_revenge_trade')] = True

    return trades


def analyze_position_sizing_consistency(
    trades: pd.DataFrame,
    size_col: str = 'size',
) -> Dict[str, float]:
    """
    Analyze consistency in position sizing.

    Args:
        trades: DataFrame with trade data
        size_col: Name of position size column

    Returns:
        Dictionary with position sizing metrics
    """
    sizes = trades[size_col].values

    metrics = {
        'mean_size': float(np.mean(sizes)),
        'median_size': float(np.median(sizes)),
        'std_size': float(np.std(sizes)),
        'cv_size': float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) != 0 else 0,
        'min_size': float(np.min(sizes)),
        'max_size': float(np.max(sizes)),
        'size_range': float(np.max(sizes) - np.min(sizes)),
    }

    return metrics


def identify_overtrading_periods(
    trades: pd.DataFrame,
    window: pd.Timedelta = pd.Timedelta(days=1),
    threshold_multiplier: float = 2.0,
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Identify periods of potential overtrading.

    Args:
        trades: DataFrame with trade data
        window: Rolling window for counting trades
        threshold_multiplier: Multiplier of average to flag overtrading
        time_col: Name of timestamp column

    Returns:
        DataFrame with overtrading periods flagged
    """
    trades = trades.sort_values(time_col).copy()

    # Count trades in rolling window
    trades['trade_count'] = 1
    trades.set_index(time_col, inplace=True)

    rolling_count = trades['trade_count'].rolling(window).sum()
    avg_count = rolling_count.mean()
    threshold = avg_count * threshold_multiplier

    trades['overtrading'] = rolling_count > threshold
    trades['rolling_trade_count'] = rolling_count

    trades.reset_index(inplace=True)

    return trades


def analyze_win_loss_clustering(
    trades: pd.DataFrame,
    pnl_col: str = 'pnl',
) -> Dict[str, float]:
    """
    Analyze clustering of wins and losses (streaks).

    Args:
        trades: DataFrame with trade data
        pnl_col: Name of PnL column

    Returns:
        Dictionary with streak analysis
    """
    trades = trades.copy()
    trades['win'] = trades[pnl_col] > 0

    # Identify streaks
    trades['streak_id'] = (trades['win'] != trades['win'].shift()).cumsum()
    streak_lengths = trades.groupby('streak_id').size()

    win_streaks = trades[trades['win']].groupby('streak_id').size()
    loss_streaks = trades[~trades['win']].groupby('streak_id').size()

    metrics = {
        'avg_win_streak': float(win_streaks.mean()) if len(win_streaks) > 0 else 0,
        'max_win_streak': int(win_streaks.max()) if len(win_streaks) > 0 else 0,
        'avg_loss_streak': float(loss_streaks.mean()) if len(loss_streaks) > 0 else 0,
        'max_loss_streak': int(loss_streaks.max()) if len(loss_streaks) > 0 else 0,
        'total_streaks': len(streak_lengths),
    }

    return metrics


def detect_correlated_trades(
    trades: pd.DataFrame,
    returns_col: str = 'return',
    correlation_threshold: float = 0.7,
) -> List[Tuple[int, int, float]]:
    """
    Detect pairs of trades with highly correlated returns.

    Useful for identifying redundant or overly similar positions.

    Args:
        trades: DataFrame with trade data
        returns_col: Name of returns column
        correlation_threshold: Threshold for flagging high correlation

    Returns:
        List of (trade_i, trade_j, correlation) tuples
    """
    correlated_pairs = []

    n_trades = len(trades)
    returns = trades[returns_col].values

    for i in range(n_trades):
        for j in range(i + 1, n_trades):
            # Calculate correlation (simple approach for illustration)
            # In practice, might want time-series correlation
            if abs(returns[i] - returns[j]) / (abs(returns[i]) + abs(returns[j]) + 1e-10) < (1 - correlation_threshold):
                correlated_pairs.append((i, j, correlation_threshold))

    return correlated_pairs


def analyze_trade_size_vs_confidence(
    trades: pd.DataFrame,
    size_col: str = 'size',
    confidence_col: Optional[str] = 'confidence',
) -> Dict[str, float]:
    """
    Analyze relationship between trade size and confidence/conviction.

    Args:
        trades: DataFrame with trade data
        size_col: Name of size column
        confidence_col: Name of confidence/conviction column (if available)

    Returns:
        Dictionary with analysis results
    """
    if confidence_col is None or confidence_col not in trades.columns:
        return {'error': 'Confidence column not available'}

    # Calculate correlation
    correlation = trades[size_col].corr(trades[confidence_col])

    # Analyze quartiles
    quartiles = pd.qcut(trades[confidence_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    avg_size_by_quartile = trades.groupby(quartiles)[size_col].mean()

    return {
        'correlation': float(correlation),
        'avg_size_q1': float(avg_size_by_quartile['Q1']),
        'avg_size_q2': float(avg_size_by_quartile['Q2']),
        'avg_size_q3': float(avg_size_by_quartile['Q3']),
        'avg_size_q4': float(avg_size_by_quartile['Q4']),
    }


def identify_optimal_holding_periods(
    trades: pd.DataFrame,
    entry_col: str = 'entry_time',
    exit_col: str = 'exit_time',
    pnl_col: str = 'pnl',
    bins: int = 10,
) -> pd.DataFrame:
    """
    Identify optimal holding periods based on historical performance.

    Args:
        trades: DataFrame with trade data
        entry_col: Name of entry time column
        exit_col: Name of exit time column
        pnl_col: Name of PnL column
        bins: Number of bins for holding period categories

    Returns:
        DataFrame with holding period analysis
    """
    trades = trades.copy()
    trades['holding_period'] = (trades[exit_col] - trades[entry_col]).dt.total_seconds() / 3600  # hours

    # Bin holding periods
    trades['holding_period_bin'] = pd.cut(trades['holding_period'], bins=bins)

    # Analyze performance by holding period
    analysis = trades.groupby('holding_period_bin').agg({
        pnl_col: ['mean', 'median', 'std', 'count'],
        'holding_period': 'mean',
    }).round(4)

    analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]

    return analysis


def detect_emotional_trading_signals(
    trades: pd.DataFrame,
    pnl_col: str = 'pnl',
    time_col: str = 'timestamp',
    size_col: str = 'size',
) -> pd.DataFrame:
    """
    Detect potential emotional trading patterns.

    Flags trades that may indicate emotional decisions:
    - Increased position size after losses
    - Rapid trading after losses
    - Unusually large positions

    Args:
        trades: DataFrame with trade data
        pnl_col: Name of PnL column
        time_col: Name of timestamp column
        size_col: Name of position size column

    Returns:
        DataFrame with emotional trading flags
    """
    trades = trades.sort_values(time_col).copy()

    # Flag 1: Increased size after loss
    trades['prev_pnl'] = trades[pnl_col].shift(1)
    trades['size_change'] = trades[size_col].pct_change()
    trades['size_increase_after_loss'] = (
        (trades['prev_pnl'] < 0) &
        (trades['size_change'] > 0.2)  # 20% increase
    )

    # Flag 2: Unusually large position
    mean_size = trades[size_col].mean()
    std_size = trades[size_col].std()
    trades['unusually_large'] = trades[size_col] > (mean_size + 2 * std_size)

    # Flag 3: Rapid trading (identified in overtrading function)

    # Combined emotional trading score
    trades['emotional_trading_score'] = (
        trades['size_increase_after_loss'].astype(int) +
        trades['unusually_large'].astype(int)
    )

    return trades
