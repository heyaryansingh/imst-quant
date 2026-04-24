"""Volatility forecasting module for IMST-Quant.

This module provides volatility forecasting methods including:
- EWMA (Exponentially Weighted Moving Average) volatility
- Simple GARCH(1,1) volatility estimation
- Historical volatility with different estimators
- Volatility cone analysis for term structure

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.Series(np.random.randn(252) * 0.01)
    >>>
    >>> # EWMA volatility forecast
    >>> ewma_vol = ewma_volatility(returns, span=20)
    >>>
    >>> # GARCH(1,1) volatility
    >>> garch_vol = garch_volatility(returns)
    >>>
    >>> # Volatility cone
    >>> cone = volatility_cone(returns, windows=[5, 10, 21, 63, 126, 252])
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolatilityForecast:
    """Result of volatility forecasting.

    Attributes:
        current: Current volatility estimate (annualized).
        forecast_1d: 1-day ahead forecast.
        forecast_5d: 5-day ahead forecast.
        forecast_21d: 21-day ahead forecast.
        method: Forecasting method used.
        half_life: Half-life of volatility (in days) if mean-reverting.
    """

    current: float
    forecast_1d: float
    forecast_5d: float
    forecast_21d: float
    method: str
    half_life: Optional[float] = None


@dataclass
class VolatilityCone:
    """Volatility cone statistics across different time windows.

    Attributes:
        windows: List of lookback windows in days.
        current: Current volatility at each window.
        min_vol: Minimum historical volatility at each window.
        max_vol: Maximum historical volatility at each window.
        median_vol: Median historical volatility at each window.
        percentile_25: 25th percentile at each window.
        percentile_75: 75th percentile at each window.
    """

    windows: list[int]
    current: list[float]
    min_vol: list[float]
    max_vol: list[float]
    median_vol: list[float]
    percentile_25: list[float]
    percentile_75: list[float]


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculate EWMA (RiskMetrics-style) volatility.

    Args:
        returns: Series of returns.
        span: EWMA span parameter (decay factor lambda = 1 - 2/(span+1)).
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of EWMA volatility estimates.

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008])
        >>> vol = ewma_volatility(returns, span=10)
    """
    # Calculate exponentially weighted variance
    ewma_var = returns.ewm(span=span, adjust=False).var()
    ewma_vol = np.sqrt(ewma_var)

    if annualize:
        ewma_vol = ewma_vol * np.sqrt(trading_days)

    return ewma_vol


def simple_garch_volatility(
    returns: pd.Series,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculate GARCH(1,1) volatility using fixed parameters.

    GARCH(1,1): sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Args:
        returns: Series of returns.
        omega: Long-run variance weight.
        alpha: Weight on lagged squared returns.
        beta: Weight on lagged variance.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of GARCH volatility estimates.

    Note:
        For true GARCH estimation, use the arch package.
        This is a simplified implementation with fixed parameters.
    """
    n = len(returns)
    variance = np.zeros(n)

    # Initialize with sample variance
    initial_var = returns.iloc[:min(20, n)].var()
    variance[0] = initial_var

    # GARCH recursion
    for t in range(1, n):
        variance[t] = (
            omega
            + alpha * returns.iloc[t - 1] ** 2
            + beta * variance[t - 1]
        )

    vol = pd.Series(np.sqrt(variance), index=returns.index)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def garch_volatility(
    returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Estimate GARCH(1,1) volatility with estimated parameters.

    Uses moment matching for quick parameter estimation.

    Args:
        returns: Series of returns.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of GARCH volatility estimates.
    """
    # Estimate parameters using moment matching
    # This is a simplified approach - for production use arch package
    sample_var = returns.var()
    sample_kurtosis = returns.kurtosis()

    # Typical GARCH parameters
    # alpha + beta should be < 1 for stationarity
    alpha = min(0.15, max(0.05, sample_kurtosis / 50))
    beta = min(0.92, max(0.80, 0.95 - alpha))
    omega = sample_var * (1 - alpha - beta)

    return simple_garch_volatility(
        returns, omega=omega, alpha=alpha, beta=beta,
        annualize=annualize, trading_days=trading_days
    )


def historical_volatility(
    returns: pd.Series,
    window: int = 21,
    method: str = "close_to_close",
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculate historical volatility using various estimators.

    Args:
        returns: Series of returns.
        window: Rolling window size.
        method: Estimation method - "close_to_close", "parkinson", "garman_klass".
                Note: parkinson and garman_klass require OHLC data passed differently.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of rolling volatility estimates.
    """
    if method == "close_to_close":
        vol = returns.rolling(window=window).std()
    else:
        # Default to close-to-close for returns data
        vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculate Parkinson volatility estimator using high-low range.

    More efficient than close-to-close when high/low data is available.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        window: Rolling window size.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of Parkinson volatility estimates.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2))
    variance = factor * (log_hl ** 2)
    vol = np.sqrt(variance.rolling(window=window).mean())

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Calculate Garman-Klass volatility estimator using OHLC data.

    Most efficient estimator when OHLC data is available.

    Args:
        open_: Series of open prices.
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        window: Rolling window size.
        annualize: Whether to annualize the volatility.
        trading_days: Number of trading days per year.

    Returns:
        Series of Garman-Klass volatility estimates.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    variance = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    vol = np.sqrt(variance.rolling(window=window).mean())

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def volatility_forecast(
    returns: pd.Series,
    method: str = "ewma",
    trading_days: int = 252,
) -> VolatilityForecast:
    """Generate volatility forecasts for multiple horizons.

    Args:
        returns: Series of returns.
        method: Forecasting method - "ewma", "garch", "historical".
        trading_days: Number of trading days per year.

    Returns:
        VolatilityForecast with current and multi-horizon forecasts.
    """
    if method == "ewma":
        vol_series = ewma_volatility(returns, span=20, annualize=True)
        current = vol_series.iloc[-1]
        # EWMA forecasts decay toward long-run mean
        long_run = returns.std() * np.sqrt(trading_days)
        decay = 2 / 21  # Approximate decay rate
        forecast_1d = current
        forecast_5d = current * (1 - decay) ** 5 + long_run * (1 - (1 - decay) ** 5)
        forecast_21d = current * (1 - decay) ** 21 + long_run * (1 - (1 - decay) ** 21)
        half_life = np.log(2) / decay

    elif method == "garch":
        vol_series = garch_volatility(returns, annualize=True)
        current = vol_series.iloc[-1]
        long_run = returns.std() * np.sqrt(trading_days)
        # GARCH mean reversion
        decay = 0.05
        forecast_1d = current
        forecast_5d = current * (1 - decay) ** 5 + long_run * (1 - (1 - decay) ** 5)
        forecast_21d = current * (1 - decay) ** 21 + long_run * (1 - (1 - decay) ** 21)
        half_life = np.log(2) / decay

    else:  # historical
        vol_series = historical_volatility(returns, window=21, annualize=True)
        current = vol_series.iloc[-1]
        forecast_1d = current
        forecast_5d = current
        forecast_21d = current
        half_life = None

    return VolatilityForecast(
        current=float(current),
        forecast_1d=float(forecast_1d),
        forecast_5d=float(forecast_5d),
        forecast_21d=float(forecast_21d),
        method=method,
        half_life=half_life,
    )


def volatility_cone(
    returns: pd.Series,
    windows: Optional[list[int]] = None,
    trading_days: int = 252,
) -> VolatilityCone:
    """Calculate volatility cone across different time horizons.

    The volatility cone shows the historical range of realized volatility
    at different lookback windows, useful for identifying whether current
    volatility is high or low relative to history.

    Args:
        returns: Series of returns.
        windows: List of lookback windows (default: [5, 10, 21, 63, 126, 252]).
        trading_days: Number of trading days per year.

    Returns:
        VolatilityCone with statistics at each window.

    Example:
        >>> returns = pd.Series(np.random.randn(500) * 0.01)
        >>> cone = volatility_cone(returns)
        >>> # Check if current vol is above median at 21-day window
        >>> current_21d = cone.current[cone.windows.index(21)]
        >>> median_21d = cone.median_vol[cone.windows.index(21)]
    """
    if windows is None:
        windows = [5, 10, 21, 63, 126, 252]

    current_vols = []
    min_vols = []
    max_vols = []
    median_vols = []
    p25_vols = []
    p75_vols = []

    for w in windows:
        if len(returns) < w:
            continue

        rolling_vol = returns.rolling(window=w).std() * np.sqrt(trading_days)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) == 0:
            continue

        current_vols.append(float(rolling_vol.iloc[-1]))
        min_vols.append(float(rolling_vol.min()))
        max_vols.append(float(rolling_vol.max()))
        median_vols.append(float(rolling_vol.median()))
        p25_vols.append(float(rolling_vol.quantile(0.25)))
        p75_vols.append(float(rolling_vol.quantile(0.75)))

    return VolatilityCone(
        windows=windows[:len(current_vols)],
        current=current_vols,
        min_vol=min_vols,
        max_vol=max_vols,
        median_vol=median_vols,
        percentile_25=p25_vols,
        percentile_75=p75_vols,
    )


def volatility_term_structure(
    returns: pd.Series,
    windows: Optional[list[int]] = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate current volatility term structure.

    Args:
        returns: Series of returns.
        windows: List of lookback windows.
        trading_days: Number of trading days per year.

    Returns:
        DataFrame with volatility at each horizon.
    """
    if windows is None:
        windows = [5, 10, 21, 42, 63, 126, 252]

    result = []
    for w in windows:
        if len(returns) >= w:
            vol = returns.iloc[-w:].std() * np.sqrt(trading_days)
            result.append({"window": w, "volatility": vol})

    return pd.DataFrame(result)


def compare_volatility_methods(
    returns: pd.Series,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compare different volatility estimation methods.

    Args:
        returns: Series of returns.
        trading_days: Number of trading days per year.

    Returns:
        DataFrame comparing volatility methods.
    """
    ewma_20 = ewma_volatility(returns, span=20, trading_days=trading_days)
    ewma_60 = ewma_volatility(returns, span=60, trading_days=trading_days)
    hist_21 = historical_volatility(returns, window=21, trading_days=trading_days)
    hist_63 = historical_volatility(returns, window=63, trading_days=trading_days)
    garch = garch_volatility(returns, trading_days=trading_days)

    return pd.DataFrame({
        "ewma_20": ewma_20,
        "ewma_60": ewma_60,
        "historical_21": hist_21,
        "historical_63": hist_63,
        "garch": garch,
    })
