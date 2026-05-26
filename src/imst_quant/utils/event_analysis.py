"""Event-driven analysis for earnings, economic releases, and market catalysts.

This module provides tools for analyzing price reactions to scheduled and unscheduled
events, calculating event-driven returns, and detecting abnormal trading activity
around corporate and macroeconomic events.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of market events."""
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    ECONOMIC_RELEASE = "economic_release"
    FED_ANNOUNCEMENT = "fed_announcement"
    MERGER = "merger"
    SPLIT = "split"
    GUIDANCE = "guidance"
    PRODUCT_LAUNCH = "product_launch"


@dataclass
class MarketEvent:
    """Market event definition."""
    event_id: str
    event_type: EventType
    ticker: str
    event_date: datetime
    announcement_time: str  # "BMO", "AMC", "market_hours"
    metadata: Dict = None


class EventStudyAnalyzer:
    """Event study analysis framework."""

    def __init__(
        self,
        prices: pd.DataFrame,
        events: List[MarketEvent]
    ):
        """Initialize event study analyzer.

        Args:
            prices: Price data (index: dates, columns: tickers)
            events: List of market events to analyze
        """
        self.prices = prices
        self.events = events
        self.returns = prices.pct_change()

    def calculate_abnormal_returns(
        self,
        event: MarketEvent,
        estimation_window: int = 120,
        event_window: Tuple[int, int] = (-1, 1)
    ) -> Dict[str, Union[float, pd.Series]]:
        """Calculate abnormal returns around event.

        Args:
            event: Market event to analyze
            estimation_window: Days for estimation period
            event_window: (days_before, days_after) event

        Returns:
            Dictionary with abnormal return metrics
        """
        ticker = event.ticker
        event_date = event.event_date

        # Find event date index
        try:
            event_idx = self.returns.index.get_loc(event_date)
        except KeyError:
            return {"error": "Event date not in price data"}

        # Estimation period (before event)
        est_start = max(0, event_idx - estimation_window - event_window[0])
        est_end = event_idx + event_window[0]

        # Get returns for estimation
        estimation_returns = self.returns[ticker].iloc[est_start:est_end]

        # Market model: estimate alpha and beta
        market_returns = self.returns.mean(axis=1).iloc[est_start:est_end]

        # Simple regression
        X = np.column_stack([np.ones(len(market_returns)), market_returns])
        y = estimation_returns

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, market_beta = beta[0], beta[1]
        except:
            alpha, market_beta = 0, 1

        # Event period returns
        event_start = event_idx + event_window[0]
        event_end = min(len(self.returns), event_idx + event_window[1] + 1)

        actual_returns = self.returns[ticker].iloc[event_start:event_end]
        market_event_returns = self.returns.mean(axis=1).iloc[event_start:event_end]

        # Expected returns using market model
        expected_returns = alpha + market_beta * market_event_returns

        # Abnormal returns
        abnormal_returns = actual_returns - expected_returns

        # Cumulative abnormal return (CAR)
        car = abnormal_returns.sum()

        # T-statistic
        estimation_std = estimation_returns.std()
        n_event_days = len(abnormal_returns)
        t_stat = car / (estimation_std * np.sqrt(n_event_days))

        return {
            'event_id': event.event_id,
            'ticker': ticker,
            'event_date': event_date,
            'car': car,
            't_statistic': t_stat,
            'abnormal_returns': abnormal_returns,
            'avg_abnormal_return': abnormal_returns.mean(),
            'max_abnormal_return': abnormal_returns.max(),
            'min_abnormal_return': abnormal_returns.min(),
            'estimation_alpha': alpha,
            'estimation_beta': market_beta
        }

    def earnings_surprise_analysis(
        self,
        earnings_data: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """Analyze price reactions to earnings surprises.

        Args:
            earnings_data: DataFrame with actual, expected EPS and dates
            price_col: Column name for prices

        Returns:
            DataFrame with earnings surprise analysis
        """
        results = []

        for _, row in earnings_data.iterrows():
            ticker = row['ticker']
            event_date = row['earnings_date']
            actual_eps = row['actual_eps']
            expected_eps = row['expected_eps']

            # Calculate surprise
            surprise = (actual_eps - expected_eps) / abs(expected_eps) if expected_eps != 0 else 0

            # Create event
            event = MarketEvent(
                event_id=f"{ticker}_{event_date}",
                event_type=EventType.EARNINGS,
                ticker=ticker,
                event_date=event_date,
                announcement_time=row.get('announcement_time', 'AMC')
            )

            # Calculate abnormal returns
            ar_result = self.calculate_abnormal_returns(event)

            if 'error' not in ar_result:
                results.append({
                    'ticker': ticker,
                    'earnings_date': event_date,
                    'surprise_pct': surprise * 100,
                    'car': ar_result['car'],
                    't_stat': ar_result['t_statistic'],
                    'avg_ar': ar_result['avg_abnormal_return']
                })

        return pd.DataFrame(results)

    def calculate_announcement_drift(
        self,
        event: MarketEvent,
        pre_window: int = 10,
        post_window: int = 30
    ) -> Dict[str, float]:
        """Calculate post-announcement drift.

        Args:
            event: Market event
            pre_window: Days before event
            post_window: Days after event

        Returns:
            Dictionary with drift metrics
        """
        ticker = event.ticker
        event_date = event.event_date

        try:
            event_idx = self.returns.index.get_loc(event_date)
        except KeyError:
            return {"error": "Event date not found"}

        # Pre-event returns
        pre_start = max(0, event_idx - pre_window)
        pre_returns = self.returns[ticker].iloc[pre_start:event_idx]

        # Post-event returns
        post_end = min(len(self.returns), event_idx + post_window)
        post_returns = self.returns[ticker].iloc[event_idx+1:post_end]

        # Calculate cumulative returns
        pre_cum = (1 + pre_returns).prod() - 1
        post_cum = (1 + post_returns).prod() - 1

        # Drift: continuation of pre-event momentum
        momentum_alignment = np.sign(pre_cum) == np.sign(post_cum)

        return {
            'pre_event_return': pre_cum,
            'post_event_return': post_cum,
            'drift_magnitude': post_cum,
            'momentum_continuation': momentum_alignment,
            'pre_window': pre_window,
            'post_window': post_window
        }

    def volume_spike_detection(
        self,
        volumes: pd.DataFrame,
        event: MarketEvent,
        lookback: int = 20,
        threshold: float = 2.0
    ) -> Dict[str, Union[float, bool]]:
        """Detect abnormal volume around events.

        Args:
            volumes: Volume data
            event: Market event
            lookback: Days for average volume calculation
            threshold: Multiple of average for spike detection

        Returns:
            Dictionary with volume metrics
        """
        ticker = event.ticker
        event_date = event.event_date

        try:
            event_idx = volumes.index.get_loc(event_date)
        except KeyError:
            return {"error": "Event date not found"}

        # Average volume before event
        hist_start = max(0, event_idx - lookback)
        avg_volume = volumes[ticker].iloc[hist_start:event_idx].mean()

        # Event day volume
        event_volume = volumes[ticker].iloc[event_idx]

        # Volume ratio
        volume_ratio = event_volume / avg_volume if avg_volume > 0 else 0

        # Detect spike
        is_spike = volume_ratio > threshold

        return {
            'avg_volume': avg_volume,
            'event_volume': event_volume,
            'volume_ratio': volume_ratio,
            'is_spike': is_spike,
            'threshold': threshold
        }


class EconomicReleaseAnalyzer:
    """Analyze market reactions to economic releases."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        economic_calendar: pd.DataFrame
    ):
        """Initialize economic release analyzer.

        Args:
            market_data: Market returns or prices
            economic_calendar: DataFrame with release dates and values
        """
        self.market_data = market_data
        self.economic_calendar = economic_calendar
        self.returns = market_data.pct_change() if 'close' in market_data.columns else market_data

    def calculate_surprise_impact(
        self,
        indicator: str,
        window_hours: int = 24
    ) -> pd.DataFrame:
        """Calculate market impact of economic surprises.

        Args:
            indicator: Economic indicator name
            window_hours: Hours around release to measure impact

        Returns:
            DataFrame with surprise impact analysis
        """
        # Filter calendar for this indicator
        releases = self.economic_calendar[
            self.economic_calendar['indicator'] == indicator
        ].copy()

        results = []

        for _, release in releases.iterrows():
            actual = release['actual']
            expected = release['expected']
            release_date = release['release_date']

            # Calculate surprise
            if expected != 0:
                surprise = (actual - expected) / abs(expected)
            else:
                surprise = 0

            # Get market returns around release
            try:
                release_idx = self.returns.index.get_loc(release_date)

                # Intraday not available, use next day
                next_day_return = self.returns.iloc[release_idx + 1].mean()

                results.append({
                    'release_date': release_date,
                    'indicator': indicator,
                    'surprise': surprise,
                    'market_return': next_day_return,
                    'actual': actual,
                    'expected': expected
                })
            except (KeyError, IndexError):
                continue

        df = pd.DataFrame(results)

        # Calculate correlation between surprise and returns
        if len(df) > 0:
            correlation = df['surprise'].corr(df['market_return'])
            df['surprise_sensitivity'] = correlation

        return df

    def fed_announcement_analysis(
        self,
        fed_dates: List[datetime],
        rate_changes: List[float]
    ) -> pd.DataFrame:
        """Analyze market reaction to Fed announcements.

        Args:
            fed_dates: List of Fed announcement dates
            rate_changes: List of rate changes (bps)

        Returns:
            DataFrame with Fed announcement analysis
        """
        results = []

        for date, rate_change in zip(fed_dates, rate_changes):
            try:
                date_idx = self.returns.index.get_loc(date)

                # Get returns on announcement day and next day
                announcement_return = self.returns.iloc[date_idx].mean()
                next_day_return = self.returns.iloc[date_idx + 1].mean()

                # Week after
                week_end = min(len(self.returns), date_idx + 5)
                week_returns = self.returns.iloc[date_idx:week_end].mean(axis=1)
                week_cum = (1 + week_returns).prod() - 1

                results.append({
                    'announcement_date': date,
                    'rate_change_bps': rate_change,
                    'announcement_day_return': announcement_return,
                    'next_day_return': next_day_return,
                    'week_return': week_cum,
                    'is_hike': rate_change > 0,
                    'is_cut': rate_change < 0
                })
            except (KeyError, IndexError):
                continue

        return pd.DataFrame(results)


def calculate_event_clustering(
    events: List[MarketEvent],
    window_days: int = 5
) -> pd.DataFrame:
    """Identify clusters of events in time.

    Args:
        events: List of market events
        window_days: Days for clustering window

    Returns:
        DataFrame with event clusters
    """
    # Sort events by date
    sorted_events = sorted(events, key=lambda e: e.event_date)

    clusters = []
    current_cluster = [sorted_events[0]]

    for i in range(1, len(sorted_events)):
        event = sorted_events[i]
        prev_event = sorted_events[i-1]

        # Check if within window
        if (event.event_date - prev_event.event_date).days <= window_days:
            current_cluster.append(event)
        else:
            # Save cluster and start new one
            if len(current_cluster) > 1:
                clusters.append({
                    'cluster_size': len(current_cluster),
                    'start_date': current_cluster[0].event_date,
                    'end_date': current_cluster[-1].event_date,
                    'tickers': [e.ticker for e in current_cluster]
                })
            current_cluster = [event]

    # Add last cluster
    if len(current_cluster) > 1:
        clusters.append({
            'cluster_size': len(current_cluster),
            'start_date': current_cluster[0].event_date,
            'end_date': current_cluster[-1].event_date,
            'tickers': [e.ticker for e in current_cluster]
        })

    return pd.DataFrame(clusters)


def identify_event_tradable_patterns(
    event_returns: pd.DataFrame,
    min_observations: int = 20,
    min_car_threshold: float = 0.02
) -> Dict[str, Dict]:
    """Identify statistically significant event patterns.

    Args:
        event_returns: DataFrame with event CARs and metadata
        min_observations: Minimum events needed
        min_car_threshold: Minimum average CAR for significance

    Returns:
        Dictionary with tradable patterns
    """
    patterns = {}

    # Group by event type
    for event_type in event_returns['event_type'].unique():
        type_data = event_returns[event_returns['event_type'] == event_type]

        if len(type_data) < min_observations:
            continue

        avg_car = type_data['car'].mean()
        std_car = type_data['car'].std()
        win_rate = (type_data['car'] > 0).mean()

        # T-test for significance
        t_stat = avg_car / (std_car / np.sqrt(len(type_data)))

        if abs(avg_car) > min_car_threshold and abs(t_stat) > 2:
            patterns[event_type] = {
                'avg_car': avg_car,
                'std_car': std_car,
                'win_rate': win_rate,
                't_statistic': t_stat,
                'observations': len(type_data),
                'tradable': True
            }

    return patterns
