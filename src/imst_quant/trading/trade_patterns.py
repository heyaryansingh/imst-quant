"""Trade clustering and pattern recognition.

Identifies recurring patterns in trading behavior:
- Winning/losing clusters
- Time-of-day patterns
- Day-of-week effects
- Seasonal patterns
- Consecutive trade analysis
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN


@dataclass
class TradePattern:
    """Identified trading pattern."""
    pattern_type: str
    description: str
    frequency: int
    avg_pnl: float
    win_rate: float
    confidence: float  # 0-1
    recommendations: List[str]


def cluster_trades_by_behavior(
    trades_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    eps: float = 0.5,
    min_samples: int = 3
) -> pd.DataFrame:
    """Cluster trades using DBSCAN to find behavioral patterns.

    Args:
        trades_df: DataFrame with trade data (pnl, duration, size, etc.)
        features: List of feature columns to use for clustering
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples per cluster

    Returns:
        DataFrame with added 'cluster' column
    """
    if features is None:
        features = ['pnl', 'hold_duration_hours', 'position_size']

    # Standardize features
    X = trades_df[features].copy()
    X = (X - X.mean()) / X.std()

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    trades_df['cluster'] = clustering.fit_predict(X)

    return trades_df


def analyze_time_of_day_patterns(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Find patterns based on trade entry time.

    Args:
        trades_df: DataFrame with 'entry_time' and 'pnl' columns

    Returns:
        DataFrame with hourly statistics
    """
    trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour

    hourly_stats = trades_df.groupby('hour').agg({
        'pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean()],
        'cluster': lambda x: x.mode()[0] if len(x) > 0 else -1
    }).reset_index()

    hourly_stats.columns = ['hour', 'count', 'avg_pnl', 'pnl_std', 'win_rate', 'dominant_cluster']

    # Statistical significance test
    overall_mean = trades_df['pnl'].mean()
    hourly_stats['significantly_better'] = False

    for idx, row in hourly_stats.iterrows():
        hour_trades = trades_df[trades_df['hour'] == row['hour']]['pnl']
        if len(hour_trades) >= 5:
            t_stat, p_value = stats.ttest_1samp(hour_trades, overall_mean)
            if p_value < 0.05 and row['avg_pnl'] > overall_mean:
                hourly_stats.at[idx, 'significantly_better'] = True

    return hourly_stats


def analyze_day_of_week_patterns(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Find patterns based on day of week.

    Args:
        trades_df: DataFrame with 'entry_time' and 'pnl' columns

    Returns:
        DataFrame with daily statistics
    """
    trades_df['day_of_week'] = pd.to_datetime(trades_df['entry_time']).dt.day_name()

    daily_stats = trades_df.groupby('day_of_week').agg({
        'pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean()],
    }).reset_index()

    daily_stats.columns = ['day', 'count', 'avg_pnl', 'pnl_std', 'win_rate']

    # Order by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats['day'] = pd.Categorical(daily_stats['day'], categories=day_order, ordered=True)
    daily_stats = daily_stats.sort_values('day').reset_index(drop=True)

    return daily_stats


def detect_winning_streaks(trades_df: pd.DataFrame, min_streak_length: int = 3) -> List[Dict]:
    """Identify winning streaks and their characteristics.

    Args:
        trades_df: DataFrame sorted by time with 'pnl' column
        min_streak_length: Minimum trades in a streak

    Returns:
        List of dictionaries with streak information
    """
    streaks = []
    current_streak = []
    current_streak_pnl = 0.0

    for idx, row in trades_df.iterrows():
        if row['pnl'] > 0:
            current_streak.append(idx)
            current_streak_pnl += row['pnl']
        else:
            if len(current_streak) >= min_streak_length:
                streaks.append({
                    'start_idx': current_streak[0],
                    'end_idx': current_streak[-1],
                    'length': len(current_streak),
                    'total_pnl': current_streak_pnl,
                    'avg_pnl': current_streak_pnl / len(current_streak),
                })
            current_streak = []
            current_streak_pnl = 0.0

    # Handle final streak
    if len(current_streak) >= min_streak_length:
        streaks.append({
            'start_idx': current_streak[0],
            'end_idx': current_streak[-1],
            'length': len(current_streak),
            'total_pnl': current_streak_pnl,
            'avg_pnl': current_streak_pnl / len(current_streak),
        })

    return streaks


def detect_losing_streaks(trades_df: pd.DataFrame, min_streak_length: int = 3) -> List[Dict]:
    """Identify losing streaks and their characteristics.

    Args:
        trades_df: DataFrame sorted by time with 'pnl' column
        min_streak_length: Minimum trades in a streak

    Returns:
        List of dictionaries with streak information
    """
    streaks = []
    current_streak = []
    current_streak_pnl = 0.0

    for idx, row in trades_df.iterrows():
        if row['pnl'] < 0:
            current_streak.append(idx)
            current_streak_pnl += row['pnl']
        else:
            if len(current_streak) >= min_streak_length:
                streaks.append({
                    'start_idx': current_streak[0],
                    'end_idx': current_streak[-1],
                    'length': len(current_streak),
                    'total_pnl': current_streak_pnl,
                    'avg_pnl': current_streak_pnl / len(current_streak),
                })
            current_streak = []
            current_streak_pnl = 0.0

    # Handle final streak
    if len(current_streak) >= min_streak_length:
        streaks.append({
            'start_idx': current_streak[0],
            'end_idx': current_streak[-1],
            'length': len(current_streak),
            'total_pnl': current_streak_pnl,
            'avg_pnl': current_streak_pnl / len(current_streak),
        })

    return streaks


def analyze_consecutive_trades(
    trades_df: pd.DataFrame,
    max_lag: int = 5
) -> Dict[int, Dict[str, float]]:
    """Analyze autocorrelation in trade outcomes.

    Checks if previous wins/losses predict future outcomes.

    Args:
        trades_df: DataFrame with 'pnl' column sorted by time
        max_lag: Maximum lag to check

    Returns:
        Dictionary mapping lag to {correlation, win_after_win, loss_after_loss}
    """
    results = {}
    wins = (trades_df['pnl'] > 0).astype(int)

    for lag in range(1, max_lag + 1):
        if len(wins) <= lag:
            continue

        # Autocorrelation
        corr = wins.autocorr(lag=lag)

        # Conditional probabilities
        win_indices = wins[wins == 1].index
        loss_indices = wins[wins == 0].index

        # Win after win
        waw_count = 0
        waw_total = 0
        for idx in win_indices:
            if idx + lag < len(wins):
                waw_total += 1
                if wins.iloc[idx + lag] == 1:
                    waw_count += 1

        # Loss after loss
        lal_count = 0
        lal_total = 0
        for idx in loss_indices:
            if idx + lag < len(wins):
                lal_total += 1
                if wins.iloc[idx + lag] == 0:
                    lal_count += 1

        results[lag] = {
            'correlation': corr,
            'win_after_win': waw_count / waw_total if waw_total > 0 else 0.0,
            'loss_after_loss': lal_count / lal_total if lal_total > 0 else 0.0,
        }

    return results


def analyze_seasonal_patterns(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Identify monthly/quarterly seasonal patterns.

    Args:
        trades_df: DataFrame with 'entry_time' and 'pnl' columns

    Returns:
        DataFrame with monthly statistics
    """
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
    trades_df['quarter'] = pd.to_datetime(trades_df['entry_time']).dt.quarter

    monthly_stats = trades_df.groupby('month').agg({
        'pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean()],
    }).reset_index()

    monthly_stats.columns = ['month', 'count', 'avg_pnl', 'pnl_std', 'win_rate']

    # Add quarter
    monthly_stats['quarter'] = ((monthly_stats['month'] - 1) // 3) + 1

    return monthly_stats


def identify_overtrading_periods(
    trades_df: pd.DataFrame,
    window_days: int = 5,
    normal_threshold: float = 2.0
) -> pd.DataFrame:
    """Find periods of excessive trading (potential overtrading).

    Args:
        trades_df: DataFrame with 'entry_time' and 'pnl' columns
        window_days: Rolling window for trade counting
        normal_threshold: Std deviations above mean to flag overtrading

    Returns:
        DataFrame with flagged overtrading periods
    """
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date

    # Count trades per day
    daily_counts = trades_df.groupby('entry_date').size().reset_index(name='trade_count')
    daily_counts['entry_date'] = pd.to_datetime(daily_counts['entry_date'])

    # Rolling average
    daily_counts['rolling_avg'] = daily_counts['trade_count'].rolling(window=window_days, min_periods=1).mean()
    daily_counts['rolling_std'] = daily_counts['trade_count'].rolling(window=window_days, min_periods=1).std()

    # Flag overtrading
    daily_counts['overtrading'] = (
        daily_counts['trade_count'] >
        daily_counts['rolling_avg'] + normal_threshold * daily_counts['rolling_std']
    )

    # Join with PnL to see performance during overtrading
    daily_pnl = trades_df.groupby('entry_date')['pnl'].agg(['sum', 'mean', 'count']).reset_index()
    daily_pnl['entry_date'] = pd.to_datetime(daily_pnl['entry_date'])

    result = daily_counts.merge(daily_pnl, on='entry_date', how='left')

    return result[result['overtrading']]


def pattern_summary(trades_df: pd.DataFrame) -> List[TradePattern]:
    """Generate comprehensive pattern summary with recommendations.

    Args:
        trades_df: DataFrame with trade data

    Returns:
        List of TradePattern objects
    """
    patterns = []

    # Time of day patterns
    hourly = analyze_time_of_day_patterns(trades_df)
    best_hours = hourly[hourly['significantly_better']]

    if len(best_hours) > 0:
        patterns.append(TradePattern(
            pattern_type="time_of_day",
            description=f"Best performance during hours: {', '.join(map(str, best_hours['hour'].tolist()))}",
            frequency=int(best_hours['count'].sum()),
            avg_pnl=float(best_hours['avg_pnl'].mean()),
            win_rate=float(best_hours['win_rate'].mean()),
            confidence=0.85,
            recommendations=[
                f"Focus trading during {best_hours['hour'].iloc[0]}:00-{best_hours['hour'].iloc[-1]}:00",
                "Reduce or avoid trading during off-peak hours"
            ]
        ))

    # Day of week patterns
    daily = analyze_day_of_week_patterns(trades_df)
    best_days = daily.nlargest(2, 'avg_pnl')

    if len(best_days) > 0:
        patterns.append(TradePattern(
            pattern_type="day_of_week",
            description=f"Strongest performance on {', '.join(best_days['day'].tolist())}",
            frequency=int(best_days['count'].sum()),
            avg_pnl=float(best_days['avg_pnl'].mean()),
            win_rate=float(best_days['win_rate'].mean()),
            confidence=0.75,
            recommendations=[
                f"Increase position sizing on {best_days['day'].iloc[0]}",
                "Review strategy for underperforming days"
            ]
        ))

    # Streak analysis
    win_streaks = detect_winning_streaks(trades_df)
    if win_streaks:
        avg_streak_length = np.mean([s['length'] for s in win_streaks])
        patterns.append(TradePattern(
            pattern_type="winning_streaks",
            description=f"Average winning streak: {avg_streak_length:.1f} trades",
            frequency=len(win_streaks),
            avg_pnl=float(np.mean([s['total_pnl'] for s in win_streaks])),
            win_rate=1.0,
            confidence=0.90,
            recommendations=[
                "Consider pyramiding during winning streaks",
                "Set profit targets to lock in streak gains"
            ]
        ))

    # Overtrading detection
    overtrading = identify_overtrading_periods(trades_df)
    if len(overtrading) > 0:
        avg_pnl_overtrade = overtrading['mean'].mean()
        patterns.append(TradePattern(
            pattern_type="overtrading",
            description=f"Overtrading detected on {len(overtrading)} days",
            frequency=len(overtrading),
            avg_pnl=float(avg_pnl_overtrade),
            win_rate=float((overtrading['sum'] > 0).mean()),
            confidence=0.80,
            recommendations=[
                "Set daily trade limits to prevent overtrading",
                "Require higher conviction for trades after X trades per day",
                "Review mental state during high-frequency trading periods"
            ]
        ))

    return patterns
