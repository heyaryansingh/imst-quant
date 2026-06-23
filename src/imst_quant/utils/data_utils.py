"""Data processing utilities for feature engineering and normalization.

Provides reusable functions for common data transformations in the trading pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union
import structlog

logger = structlog.get_logger()


def normalize_series(
    series: pd.Series,
    method: str = "zscore",
    window: Optional[int] = None,
) -> pd.Series:
    """Normalize a pandas Series using various methods.

    Args:
        series: Input series to normalize
        method: Normalization method:
            - 'zscore': (x - mean) / std
            - 'minmax': (x - min) / (max - min)
            - 'robust': (x - median) / IQR
            - 'percent': percentage change from first value
        window: Rolling window size for rolling normalization (None = global)

    Returns:
        Normalized series

    Example:
        >>> s = pd.Series([100, 105, 110, 108, 115])
        >>> normalize_series(s, method='zscore')
    """
    if method == "zscore":
        if window:
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
            return (series - mean) / std
        else:
            return (series - series.mean()) / series.std()

    elif method == "minmax":
        if window:
            rolling_min = series.rolling(window).min()
            rolling_max = series.rolling(window).max()
            return (series - rolling_min) / (rolling_max - rolling_min)
        else:
            return (series - series.min()) / (series.max() - series.min())

    elif method == "robust":
        if window:
            median = series.rolling(window).median()
            q75 = series.rolling(window).quantile(0.75)
            q25 = series.rolling(window).quantile(0.25)
            iqr = q75 - q25
            return (series - median) / iqr
        else:
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            return (series - median) / iqr

    elif method == "percent":
        first_value = series.iloc[0]
        return (series - first_value) / first_value

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str = "1D",
    timestamp_col: str = "timestamp",
    ohlcv_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Resample OHLCV data to target frequency.

    Args:
        df: DataFrame with OHLCV data
        target_freq: Target frequency ('1H', '4H', '1D', '1W', etc.)
        timestamp_col: Name of timestamp column
        ohlcv_cols: Dict mapping to OHLCV columns. Default:
            {'open': 'open', 'high': 'high', 'low': 'low',
             'close': 'close', 'volume': 'volume'}

    Returns:
        Resampled DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        ...     'open': np.random.randn(100) + 100,
        ...     'high': np.random.randn(100) + 101,
        ...     'low': np.random.randn(100) + 99,
        ...     'close': np.random.randn(100) + 100,
        ...     'volume': np.random.randint(1000, 10000, 100)
        ... })
        >>> daily = resample_ohlcv(df, '1D')
    """
    if ohlcv_cols is None:
        ohlcv_cols = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

    # Set timestamp as index
    df_indexed = df.set_index(timestamp_col)

    # Resample with OHLCV aggregation
    resampled = df_indexed.resample(target_freq).agg({
        ohlcv_cols["open"]: "first",
        ohlcv_cols["high"]: "max",
        ohlcv_cols["low"]: "min",
        ohlcv_cols["close"]: "last",
        ohlcv_cols["volume"]: "sum",
    })

    # Reset index
    resampled = resampled.reset_index()

    logger.info("Data resampled",
               original_freq=df_indexed.index.inferred_freq,
               target_freq=target_freq,
               rows_before=len(df),
               rows_after=len(resampled))

    return resampled


def handle_missing_data(
    df: pd.DataFrame,
    method: str = "forward_fill",
    columns: Optional[List[str]] = None,
    max_gap: Optional[int] = None,
) -> pd.DataFrame:
    """Handle missing data with various strategies.

    Args:
        df: DataFrame with potential missing values
        method: Imputation method:
            - 'forward_fill': Forward fill (carry last value)
            - 'backward_fill': Backward fill
            - 'interpolate': Linear interpolation
            - 'drop': Drop rows with missing values
            - 'zero': Fill with zeros
            - 'mean': Fill with column mean
        columns: List of columns to process (None = all numeric)
        max_gap: Maximum gap size to fill (None = fill all gaps)

    Returns:
        DataFrame with missing values handled

    Example:
        >>> df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        >>> handle_missing_data(df, method='interpolate')
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if method == "forward_fill":
            if max_gap:
                df_copy[col] = df_copy[col].fillna(method="ffill", limit=max_gap)
            else:
                df_copy[col] = df_copy[col].fillna(method="ffill")

        elif method == "backward_fill":
            if max_gap:
                df_copy[col] = df_copy[col].fillna(method="bfill", limit=max_gap)
            else:
                df_copy[col] = df_copy[col].fillna(method="bfill")

        elif method == "interpolate":
            if max_gap:
                df_copy[col] = df_copy[col].interpolate(method="linear", limit=max_gap)
            else:
                df_copy[col] = df_copy[col].interpolate(method="linear")

        elif method == "zero":
            df_copy[col] = df_copy[col].fillna(0)

        elif method == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

        elif method == "drop":
            df_copy = df_copy.dropna(subset=[col])

        else:
            raise ValueError(f"Unknown method: {method}")

    logger.info("Missing data handled",
               method=method,
               columns=columns,
               rows_before=len(df),
               rows_after=len(df_copy))

    return df_copy


def calculate_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: Union[int, List[int]] = 1,
    method: str = "simple",
) -> pd.DataFrame:
    """Calculate returns over various periods.

    Args:
        df: DataFrame with price data
        price_col: Column name for prices
        periods: Period(s) for return calculation (e.g., 1, 5, 20 days)
        method: Return calculation method:
            - 'simple': (P_t - P_{t-1}) / P_{t-1}
            - 'log': ln(P_t / P_{t-1})

    Returns:
        DataFrame with return columns added

    Example:
        >>> df = pd.DataFrame({'close': [100, 105, 110, 108]})
        >>> calculate_returns(df, periods=[1, 2])
    """
    df_copy = df.copy()

    if isinstance(periods, int):
        periods = [periods]

    for period in periods:
        col_name = f"return_{period}d"

        if method == "simple":
            df_copy[col_name] = df_copy[price_col].pct_change(periods=period)
        elif method == "log":
            df_copy[col_name] = np.log(df_copy[price_col] / df_copy[price_col].shift(period))
        else:
            raise ValueError(f"Unknown method: {method}")

    return df_copy


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Detect outliers in numerical columns.

    Args:
        df: Input DataFrame
        columns: Columns to check (None = all numeric)
        method: Detection method:
            - 'iqr': Interquartile range (1.5 * IQR)
            - 'zscore': Z-score (threshold standard deviations)
            - 'mad': Median absolute deviation
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outlier flags (True = outlier)

    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 3, 100, 4, 5]})
        >>> outliers = detect_outliers(df, method='iqr')
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_flags = pd.DataFrame(index=df.index)

    for col in columns:
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_flags[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)

        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_flags[f"{col}_outlier"] = z_scores > threshold

        elif method == "mad":
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            modified_z = 0.6745 * (df[col] - median) / mad
            outlier_flags[f"{col}_outlier"] = np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

    return outlier_flags


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    drop_na: bool = True,
) -> pd.DataFrame:
    """Create lagged versions of columns for time series modeling.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods (e.g., [1, 2, 5, 10])
        drop_na: Whether to drop rows with NaN from lagging

    Returns:
        DataFrame with lagged feature columns

    Example:
        >>> df = pd.DataFrame({'price': [100, 105, 110, 108, 115]})
        >>> create_lagged_features(df, columns=['price'], lags=[1, 2])
    """
    df_copy = df.copy()

    for col in columns:
        for lag in lags:
            df_copy[f"{col}_lag_{lag}"] = df_copy[col].shift(lag)

    if drop_na:
        df_copy = df_copy.dropna()

    logger.info("Lagged features created",
               columns=columns,
               lags=lags,
               new_features=len(columns) * len(lags))

    return df_copy


def winsorize_series(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Cap extreme values at specified quantiles (winsorization).

    Args:
        series: Input series
        lower_quantile: Lower quantile threshold (e.g., 0.01 for 1%)
        upper_quantile: Upper quantile threshold (e.g., 0.99 for 99%)

    Returns:
        Winsorized series

    Example:
        >>> s = pd.Series([1, 2, 3, 100, 4, 5, -50])
        >>> winsorize_series(s, lower_quantile=0.1, upper_quantile=0.9)
    """
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)

    return series.clip(lower=lower_bound, upper=upper_bound)


def cross_sectional_rank(
    df: pd.DataFrame,
    column: str,
    timestamp_col: str = "timestamp",
    normalize: bool = True,
) -> pd.Series:
    """Compute cross-sectional rank normalization.

    Common in quantitative finance for creating market-neutral signals by ranking
    assets within each time period.

    Args:
        df: DataFrame with multi-asset time series data
        column: Column to rank
        timestamp_col: Column containing timestamps
        normalize: If True, normalize ranks to [-1, 1] range

    Returns:
        Series with cross-sectional ranks

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        ...     'ticker': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
        ...     'return': [0.02, 0.01, -0.01, 0.03]
        ... })
        >>> ranks = cross_sectional_rank(df, 'return', normalize=True)
    """
    # Rank within each timestamp
    ranks = df.groupby(timestamp_col)[column].rank(method='average', pct=normalize)

    if normalize:
        # Scale from [0, 1] to [-1, 1]
        ranks = 2 * (ranks - 0.5)

    logger.info("Cross-sectional ranks computed",
                column=column,
                normalize=normalize,
                unique_timestamps=df[timestamp_col].nunique())

    return ranks


def momentum_zscore(
    df: pd.DataFrame,
    price_col: str = "close",
    lookback: int = 20,
    zscore_window: int = 60,
) -> pd.Series:
    """Calculate momentum z-score signal.

    Computes momentum (rate of change) and then standardizes it using a rolling
    z-score to identify extreme momentum events.

    Args:
        df: DataFrame with price data
        price_col: Column containing prices
        lookback: Period for momentum calculation (default: 20 days)
        zscore_window: Window for z-score standardization (default: 60 days)

    Returns:
        Series with momentum z-scores

    Example:
        >>> df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100})
        >>> signal = momentum_zscore(df, lookback=10, zscore_window=30)
    """
    # Calculate momentum (percentage change)
    momentum = df[price_col].pct_change(periods=lookback)

    # Calculate rolling z-score
    rolling_mean = momentum.rolling(window=zscore_window).mean()
    rolling_std = momentum.rolling(window=zscore_window).std()

    zscore = (momentum - rolling_mean) / rolling_std

    logger.info("Momentum z-score computed",
                lookback=lookback,
                zscore_window=zscore_window,
                mean_zscore=zscore.mean(),
                std_zscore=zscore.std())

    return zscore


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    annualization_factor: int = 252,
) -> pd.Series:
    """Calculate rolling Sharpe ratio.

    Useful for tracking strategy performance over time.

    Args:
        returns: Series of returns
        window: Rolling window size
        annualization_factor: Factor to annualize (252 for daily, 12 for monthly)

    Returns:
        Series with rolling Sharpe ratios

    Example:
        >>> returns = pd.Series(np.random.randn(100) * 0.01)
        >>> sharpe = calculate_rolling_sharpe(returns, window=20)
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    # Annualized Sharpe
    sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization_factor)

    logger.info("Rolling Sharpe ratio computed",
                window=window,
                annualization_factor=annualization_factor,
                mean_sharpe=sharpe.mean())

    return sharpe


def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Calculate rolling correlation between two series.

    Useful for monitoring changing relationships between assets or factors.

    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        min_periods: Minimum observations required (default: window)

    Returns:
        Series with rolling correlations

    Example:
        >>> stock_returns = pd.Series(np.random.randn(100) * 0.01)
        >>> market_returns = pd.Series(np.random.randn(100) * 0.01)
        >>> rolling_corr = calculate_rolling_correlation(stock_returns, market_returns, window=20)
    """
    if min_periods is None:
        min_periods = window

    rolling_corr = series1.rolling(window=window, min_periods=min_periods).corr(series2)

    logger.info("Rolling correlation computed",
                window=window,
                min_periods=min_periods,
                mean_corr=rolling_corr.mean(),
                std_corr=rolling_corr.std())

    return rolling_corr


def calculate_autocorrelation(
    series: pd.Series,
    max_lags: int = 20,
) -> pd.Series:
    """Calculate autocorrelation function (ACF) for a time series.

    Useful for detecting mean reversion, momentum, or patterns in returns.

    Args:
        series: Input time series
        max_lags: Maximum number of lags to compute (default: 20)

    Returns:
        Series with autocorrelations indexed by lag

    Example:
        >>> returns = pd.Series(np.random.randn(100) * 0.01)
        >>> acf = calculate_autocorrelation(returns, max_lags=10)
        >>> print(f"Lag-1 autocorr: {acf.iloc[1]:.3f}")
    """
    autocorr_values = [series.autocorr(lag=lag) for lag in range(max_lags + 1)]
    acf = pd.Series(autocorr_values, index=range(max_lags + 1))

    logger.info("Autocorrelation computed",
                max_lags=max_lags,
                lag1_autocorr=acf.iloc[1] if len(acf) > 1 else None,
                significant_lags=sum(abs(acf) > 2 / np.sqrt(len(series))))

    return acf


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: Optional[int] = None,
) -> Union[float, pd.Series]:
    """Calculate beta (systematic risk) of an asset relative to the market.

    Args:
        asset_returns: Asset return series
        market_returns: Market/benchmark return series
        window: Rolling window size (None = single beta for entire period)

    Returns:
        Beta value(s) - float if window=None, Series if rolling

    Example:
        >>> stock = pd.Series(np.random.randn(100) * 0.02)
        >>> market = pd.Series(np.random.randn(100) * 0.01)
        >>> beta = calculate_beta(stock, market)
        >>> rolling_beta = calculate_beta(stock, market, window=60)
    """
    if window is None:
        # Calculate single beta
        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance
        logger.info("Beta calculated", beta=beta)
        return beta
    else:
        # Calculate rolling beta
        rolling_cov = asset_returns.rolling(window=window).cov(market_returns)
        rolling_var = market_returns.rolling(window=window).var()

        beta = rolling_cov / rolling_var
        logger.info("Rolling beta calculated",
                    window=window,
                    mean_beta=beta.mean(),
                    std_beta=beta.std())
        return beta
