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
