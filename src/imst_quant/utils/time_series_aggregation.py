"""Time-series data aggregation utilities for quantitative trading analysis.

This module provides utilities for aggregating and resampling time-series data with
support for multiple aggregation methods, rolling windows, and statistical features.

Features:
- Multi-timeframe resampling (tick -> minute -> hour -> daily)
- Custom aggregation functions (OHLCV, vwap, custom stats)
- Rolling window computations
- Gap filling and data alignment

Example:
    >>> from imst_quant.utils.time_series_aggregation import TimeSeriesAggregator
    >>> agg = TimeSeriesAggregator(tick_data)
    >>> minute_bars = agg.resample('1T', method='ohlcv')
    >>> hourly_vwap = agg.compute_vwap('1H')
"""

from typing import Dict, List, Optional, Callable, Union, Literal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


AggMethod = Literal['ohlcv', 'vwap', 'mean', 'median', 'sum', 'count', 'custom']


class TimeSeriesAggregator:
    """Aggregate and resample time-series trading data.

    Args:
        data: DataFrame with datetime index and price/volume columns
        price_col: Name of price column (default: 'price')
        volume_col: Name of volume column (default: 'volume')

    Attributes:
        data: Input time-series data
        price_col: Price column name
        volume_col: Volume column name

    Example:
        >>> data = pd.DataFrame({
        ...     'price': [100, 101, 102, 101, 100],
        ...     'volume': [1000, 1500, 2000, 1800, 1200]
        ... }, index=pd.date_range('2024-01-01', periods=5, freq='1min'))
        >>> agg = TimeSeriesAggregator(data)
        >>> bars = agg.resample('5T', method='ohlcv')
    """

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume'
    ):
        self.data = data.copy()
        self.price_col = price_col
        self.volume_col = volume_col

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

    def resample(
        self,
        freq: str,
        method: AggMethod = 'ohlcv',
        custom_func: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Resample time-series data to specified frequency.

        Args:
            freq: Pandas frequency string ('1T', '5T', '1H', '1D', etc.)
            method: Aggregation method ('ohlcv', 'vwap', 'mean', etc.)
            custom_func: Custom aggregation function (required if method='custom')

        Returns:
            Resampled DataFrame

        Example:
            >>> # Create 5-minute OHLCV bars
            >>> bars = agg.resample('5T', method='ohlcv')
            >>> # Create hourly VWAP
            >>> vwap = agg.resample('1H', method='vwap')
        """
        if method == 'ohlcv':
            return self._resample_ohlcv(freq)
        elif method == 'vwap':
            return self._resample_vwap(freq)
        elif method == 'mean':
            return self.data.resample(freq).mean()
        elif method == 'median':
            return self.data.resample(freq).median()
        elif method == 'sum':
            return self.data.resample(freq).sum()
        elif method == 'count':
            return self.data.resample(freq).count()
        elif method == 'custom':
            if custom_func is None:
                raise ValueError("custom_func required when method='custom'")
            return self.data.resample(freq).apply(custom_func)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _resample_ohlcv(self, freq: str) -> pd.DataFrame:
        """Resample to OHLCV bars."""
        resampled = pd.DataFrame()

        resampled['open'] = self.data[self.price_col].resample(freq).first()
        resampled['high'] = self.data[self.price_col].resample(freq).max()
        resampled['low'] = self.data[self.price_col].resample(freq).min()
        resampled['close'] = self.data[self.price_col].resample(freq).last()

        if self.volume_col in self.data.columns:
            resampled['volume'] = self.data[self.volume_col].resample(freq).sum()

        # Add trade count
        resampled['trade_count'] = self.data[self.price_col].resample(freq).count()

        return resampled.dropna()

    def _resample_vwap(self, freq: str) -> pd.DataFrame:
        """Resample to VWAP (Volume-Weighted Average Price)."""
        if self.volume_col not in self.data.columns:
            raise ValueError(f"Volume column '{self.volume_col}' not found")

        resampler = self.data.resample(freq)

        # VWAP = sum(price * volume) / sum(volume)
        vwap = (
            (self.data[self.price_col] * self.data[self.volume_col])
            .resample(freq)
            .sum()
            / self.data[self.volume_col].resample(freq).sum()
        )

        result = pd.DataFrame({
            'vwap': vwap,
            'volume': resampler[self.volume_col].sum(),
            'trade_count': resampler[self.price_col].count()
        })

        return result.dropna()

    def compute_vwap(self, freq: str = '1D') -> pd.Series:
        """Compute Volume-Weighted Average Price.

        Args:
            freq: Frequency for VWAP calculation

        Returns:
            Series of VWAP values

        Example:
            >>> daily_vwap = agg.compute_vwap('1D')
        """
        result = self._resample_vwap(freq)
        return result['vwap']

    def rolling_aggregation(
        self,
        window: int,
        agg_func: Union[str, Callable],
        center: bool = False
    ) -> pd.DataFrame:
        """Apply rolling window aggregation.

        Args:
            window: Window size in number of periods
            agg_func: Aggregation function ('mean', 'std', 'sum', or callable)
            center: Whether to center the window

        Returns:
            DataFrame with rolling aggregations

        Example:
            >>> # 20-period rolling mean
            >>> rolling_mean = agg.rolling_aggregation(20, 'mean')
            >>> # 20-period rolling volatility
            >>> rolling_vol = agg.rolling_aggregation(20, 'std')
        """
        rolling = self.data.rolling(window=window, center=center)

        if isinstance(agg_func, str):
            return getattr(rolling, agg_func)()
        else:
            return rolling.apply(agg_func)

    def multi_timeframe_features(
        self,
        timeframes: List[str],
        features: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Generate features across multiple timeframes.

        Args:
            timeframes: List of frequency strings ('1T', '5T', '1H', etc.)
            features: List of statistical features to compute

        Returns:
            DataFrame with multi-timeframe features

        Example:
            >>> # Compute mean and std across multiple timeframes
            >>> mtf = agg.multi_timeframe_features(
            ...     ['5T', '15T', '1H'],
            ...     ['mean', 'std']
            ... )
        """
        result = self.data[[self.price_col]].copy()

        for tf in timeframes:
            for feat in features:
                col_name = f'{self.price_col}_{tf}_{feat}'

                if feat == 'mean':
                    result[col_name] = self.data[self.price_col].resample(tf).mean().reindex(result.index, method='ffill')
                elif feat == 'std':
                    result[col_name] = self.data[self.price_col].resample(tf).std().reindex(result.index, method='ffill')
                elif feat == 'min':
                    result[col_name] = self.data[self.price_col].resample(tf).min().reindex(result.index, method='ffill')
                elif feat == 'max':
                    result[col_name] = self.data[self.price_col].resample(tf).max().reindex(result.index, method='ffill')
                elif feat == 'sum':
                    result[col_name] = self.data[self.price_col].resample(tf).sum().reindex(result.index, method='ffill')
                elif feat == 'count':
                    result[col_name] = self.data[self.price_col].resample(tf).count().reindex(result.index, method='ffill')

        return result

    def fill_gaps(
        self,
        freq: str,
        method: Literal['ffill', 'bfill', 'interpolate', 'zero'] = 'ffill'
    ) -> pd.DataFrame:
        """Fill gaps in time-series data.

        Args:
            freq: Target frequency string
            method: Gap filling method

        Returns:
            DataFrame with filled gaps

        Example:
            >>> # Fill missing minutes with forward fill
            >>> filled = agg.fill_gaps('1T', method='ffill')
        """
        # Create complete time index
        start = self.data.index.min()
        end = self.data.index.max()
        complete_index = pd.date_range(start, end, freq=freq)

        # Reindex with complete time index
        filled = self.data.reindex(complete_index)

        # Apply fill method
        if method == 'ffill':
            filled = filled.fillna(method='ffill')
        elif method == 'bfill':
            filled = filled.fillna(method='bfill')
        elif method == 'interpolate':
            filled = filled.interpolate(method='time')
        elif method == 'zero':
            filled = filled.fillna(0)

        return filled

    def compute_returns(
        self,
        periods: int = 1,
        method: Literal['simple', 'log'] = 'simple'
    ) -> pd.Series:
        """Compute price returns.

        Args:
            periods: Number of periods for return calculation
            method: Return calculation method ('simple' or 'log')

        Returns:
            Series of returns

        Example:
            >>> # Simple 1-period returns
            >>> returns = agg.compute_returns(periods=1)
            >>> # Log returns
            >>> log_returns = agg.compute_returns(periods=1, method='log')
        """
        prices = self.data[self.price_col]

        if method == 'simple':
            return prices.pct_change(periods=periods)
        elif method == 'log':
            return np.log(prices / prices.shift(periods))
        else:
            raise ValueError(f"Unknown method: {method}")

    def align_multiple_series(
        self,
        other_series: Dict[str, pd.Series],
        method: Literal['inner', 'outer', 'left', 'right'] = 'inner'
    ) -> pd.DataFrame:
        """Align multiple time-series to common index.

        Args:
            other_series: Dictionary of series to align {name: series}
            method: Join method for alignment

        Returns:
            DataFrame with aligned series

        Example:
            >>> # Align price data with external signals
            >>> aligned = agg.align_multiple_series({
            ...     'signal_a': signal_a_series,
            ...     'signal_b': signal_b_series
            ... }, method='inner')
        """
        result = self.data.copy()

        for name, series in other_series.items():
            result = result.join(series.rename(name), how=method)

        return result


def aggregate_order_flow(
    trades: pd.DataFrame,
    freq: str = '1T',
    classify_trades: bool = True
) -> pd.DataFrame:
    """Aggregate order flow data from individual trades.

    Args:
        trades: DataFrame with trade data (price, volume, optionally side)
        freq: Aggregation frequency
        classify_trades: Whether to classify trades as buy/sell

    Returns:
        DataFrame with aggregated order flow metrics

    Example:
        >>> order_flow = aggregate_order_flow(trades, freq='5T')
    """
    if 'side' not in trades.columns and classify_trades:
        # Classify trades using tick rule
        trades = trades.copy()
        trades['side'] = np.where(
            trades['price'] > trades['price'].shift(1),
            'buy',
            'sell'
        )

    resampled = pd.DataFrame()

    # Volume-based metrics
    resampled['total_volume'] = trades['volume'].resample(freq).sum()

    if 'side' in trades.columns:
        buy_volume = trades[trades['side'] == 'buy']['volume'].resample(freq).sum()
        sell_volume = trades[trades['side'] == 'sell']['volume'].resample(freq).sum()

        resampled['buy_volume'] = buy_volume
        resampled['sell_volume'] = sell_volume
        resampled['volume_imbalance'] = buy_volume - sell_volume
        resampled['volume_imbalance_ratio'] = (
            (buy_volume - sell_volume) / (buy_volume + sell_volume)
        ).replace([np.inf, -np.inf], 0)

    # Trade count metrics
    resampled['trade_count'] = trades['price'].resample(freq).count()

    # Price impact metrics
    resampled['price_range'] = (
        trades['price'].resample(freq).max() -
        trades['price'].resample(freq).min()
    )
    resampled['price_volatility'] = trades['price'].resample(freq).std()

    return resampled.fillna(0)


def compute_time_weighted_average(
    data: pd.DataFrame,
    value_col: str = 'price',
    freq: str = '1D'
) -> pd.Series:
    """Compute Time-Weighted Average Price (TWAP).

    Args:
        data: DataFrame with datetime index and price column
        value_col: Name of value column
        freq: Aggregation frequency

    Returns:
        Series of TWAP values

    Example:
        >>> twap = compute_time_weighted_average(data, freq='1H')
    """
    # Calculate time differences
    time_diffs = data.index.to_series().diff().dt.total_seconds().fillna(0)

    # Weight each price by time duration
    weighted_values = data[value_col] * time_diffs

    # Aggregate by frequency
    twap = (
        weighted_values.resample(freq).sum() /
        time_diffs.resample(freq).sum()
    )

    return twap.dropna()
