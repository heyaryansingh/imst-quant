"""Cointegration analysis for pairs trading and statistical arbitrage.

Provides tools for identifying cointegrated pairs, calculating hedge ratios,
and generating mean-reversion signals for pairs trading strategies.

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Simulate two cointegrated series
    >>> n = 500
    >>> x = np.cumsum(np.random.randn(n))
    >>> y = 0.8 * x + np.random.randn(n) * 0.5  # Cointegrated with x
    >>> series1 = pd.Series(x, name="ASSET_A")
    >>> series2 = pd.Series(y, name="ASSET_B")
    >>> # Test for cointegration
    >>> result = test_cointegration(series1, series2)
    >>> print(f"Cointegrated: {result.is_cointegrated}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CointegrationResult:
    """Result from cointegration test.

    Attributes:
        is_cointegrated: Whether the pair is cointegrated at significance level.
        adf_statistic: ADF test statistic on the spread.
        p_value: P-value of the ADF test.
        critical_values: Critical values at 1%, 5%, 10% levels.
        hedge_ratio: Optimal hedge ratio (beta) from OLS regression.
        half_life: Estimated mean reversion half-life in periods.
        spread_mean: Mean of the spread series.
        spread_std: Standard deviation of the spread.
    """

    is_cointegrated: bool
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    half_life: float
    spread_mean: float
    spread_std: float


@dataclass
class PairsTradingSignal:
    """Signal for pairs trading strategy.

    Attributes:
        zscore: Current z-score of the spread.
        position: Suggested position (-1=short pair, 0=neutral, 1=long pair).
        spread_value: Current spread value.
        entry_threshold: Z-score threshold for entry.
        exit_threshold: Z-score threshold for exit.
        stop_loss_threshold: Z-score threshold for stop loss.
    """

    zscore: float
    position: int
    spread_value: float
    entry_threshold: float
    exit_threshold: float
    stop_loss_threshold: float


def calculate_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    method: str = "ols",
) -> Tuple[float, float]:
    """Calculate optimal hedge ratio between two series.

    The hedge ratio determines how many units of asset X to trade
    for each unit of asset Y to create a stationary spread.

    Args:
        y: Dependent variable series (asset to hedge).
        x: Independent variable series (hedging instrument).
        method: Method for estimation ("ols" or "tls").

    Returns:
        Tuple of (hedge_ratio, intercept).

    Example:
        >>> y = pd.Series([100, 102, 101, 103, 104])
        >>> x = pd.Series([50, 51, 50.5, 51.5, 52])
        >>> beta, alpha = calculate_hedge_ratio(y, x)
    """
    if method == "ols":
        # Ordinary Least Squares: y = alpha + beta * x
        x_vals = x.values
        y_vals = y.values

        # Add constant for intercept
        x_with_const = np.column_stack([np.ones(len(x_vals)), x_vals])
        coeffs = np.linalg.lstsq(x_with_const, y_vals, rcond=None)[0]

        intercept = coeffs[0]
        hedge_ratio = coeffs[1]

    elif method == "tls":
        # Total Least Squares (orthogonal regression)
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        # SVD-based TLS
        cov = np.cov(x_centered, y_centered)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        min_idx = np.argmin(eigenvalues)
        hedge_ratio = -eigenvectors[0, min_idx] / eigenvectors[1, min_idx]
        intercept = y.mean() - hedge_ratio * x.mean()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ols' or 'tls'.")

    return float(hedge_ratio), float(intercept)


def calculate_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: Optional[float] = None,
) -> pd.Series:
    """Calculate the spread between two cointegrated series.

    The spread = y - hedge_ratio * x. For cointegrated pairs,
    this spread should be stationary.

    Args:
        y: First series (typically the one you're long).
        x: Second series (hedging instrument).
        hedge_ratio: Hedge ratio to use. If None, calculated via OLS.

    Returns:
        Series representing the spread.
    """
    if hedge_ratio is None:
        hedge_ratio, _ = calculate_hedge_ratio(y, x)

    spread = y - hedge_ratio * x
    return spread


def adf_test(
    series: pd.Series,
    max_lag: Optional[int] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Perform Augmented Dickey-Fuller test for stationarity.

    Tests the null hypothesis that a unit root is present (non-stationary).
    Rejection of null suggests the series is stationary.

    Args:
        series: Time series to test.
        max_lag: Maximum lag to include. If None, uses int(12 * (n/100)^0.25).

    Returns:
        Tuple of (adf_statistic, p_value, critical_values_dict).
    """
    series = series.dropna()
    n = len(series)

    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)

    # First difference
    diff = series.diff().dropna()
    y_lag = series.shift(1).dropna()

    # Align series
    common_idx = diff.index.intersection(y_lag.index)
    diff = diff.loc[common_idx]
    y_lag = y_lag.loc[common_idx]

    # Build lagged differences for augmentation
    lagged_diffs = pd.DataFrame()
    for i in range(1, max_lag + 1):
        lagged = diff.shift(i)
        lagged_diffs[f"lag_{i}"] = lagged

    # Align all
    common_idx = diff.index
    for col in lagged_diffs.columns:
        common_idx = common_idx.intersection(lagged_diffs[col].dropna().index)

    diff = diff.loc[common_idx]
    y_lag = y_lag.loc[common_idx]
    lagged_diffs = lagged_diffs.loc[common_idx]

    # Regression: diff_y = alpha + beta * y_lag + sum(gamma_i * diff_y_lag_i)
    X = pd.concat([pd.Series(1, index=y_lag.index, name="const"), y_lag], axis=1)
    if len(lagged_diffs.columns) > 0:
        X = pd.concat([X, lagged_diffs], axis=1)

    y = diff.values
    X_vals = X.values

    # OLS
    try:
        coeffs = np.linalg.lstsq(X_vals, y, rcond=None)[0]
        residuals = y - X_vals @ coeffs
        sse = (residuals ** 2).sum()
        mse = sse / (len(y) - len(coeffs))
        var_coef = mse * np.linalg.inv(X_vals.T @ X_vals)
        se = np.sqrt(np.diag(var_coef))

        # t-statistic for the coefficient on y_lag (index 1)
        adf_stat = coeffs[1] / se[1]
    except Exception:
        adf_stat = 0.0

    # Approximate critical values (MacKinnon 1994 for n=250, with constant)
    critical_values = {
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    }

    # Approximate p-value using interpolation
    if adf_stat < -3.43:
        p_value = 0.005
    elif adf_stat < -2.86:
        p_value = 0.03
    elif adf_stat < -2.57:
        p_value = 0.07
    elif adf_stat < -1.94:
        p_value = 0.15
    else:
        p_value = 0.5 + 0.5 * (1 - np.exp(-0.5 * (adf_stat + 1.94)))
        p_value = min(1.0, p_value)

    return float(adf_stat), float(p_value), critical_values


def calculate_half_life(spread: pd.Series) -> float:
    """Calculate mean reversion half-life of a spread.

    Uses an AR(1) model: spread_t = phi * spread_{t-1} + epsilon
    Half-life = -ln(2) / ln(phi)

    Args:
        spread: Stationary spread series.

    Returns:
        Half-life in periods (e.g., days).
    """
    spread = spread.dropna()

    y = spread.diff().dropna()
    x = spread.shift(1).dropna()

    # Align
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx].values
    x = x.loc[common_idx].values

    # OLS: delta_spread = theta * spread_{t-1}
    theta = (x @ y) / (x @ x) if (x @ x) > 0 else 0

    phi = 1 + theta

    if phi <= 0 or phi >= 1:
        # Not mean reverting or unstable
        return float("inf")

    half_life = -np.log(2) / np.log(phi)
    return max(1.0, float(half_life))


def test_cointegration(
    y: pd.Series,
    x: pd.Series,
    significance: float = 0.05,
) -> CointegrationResult:
    """Test for cointegration between two time series (Engle-Granger method).

    Steps:
    1. Estimate hedge ratio via OLS
    2. Calculate spread (residuals)
    3. Test spread for stationarity using ADF test

    Args:
        y: First time series.
        x: Second time series.
        significance: Significance level for cointegration test.

    Returns:
        CointegrationResult with test statistics and parameters.

    Example:
        >>> y = pd.Series(np.cumsum(np.random.randn(200)))
        >>> x = pd.Series(np.cumsum(np.random.randn(200)))
        >>> result = test_cointegration(y, x)
        >>> print(f"P-value: {result.p_value:.3f}")
    """
    # Calculate hedge ratio and spread
    hedge_ratio, _ = calculate_hedge_ratio(y, x)
    spread = calculate_spread(y, x, hedge_ratio)

    # ADF test on spread
    adf_stat, p_value, critical_values = adf_test(spread)

    # Determine if cointegrated
    is_cointegrated = p_value < significance

    # Calculate half-life
    half_life = calculate_half_life(spread)

    # Spread statistics
    spread_mean = float(spread.mean())
    spread_std = float(spread.std())

    return CointegrationResult(
        is_cointegrated=is_cointegrated,
        adf_statistic=adf_stat,
        p_value=p_value,
        critical_values=critical_values,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        spread_mean=spread_mean,
        spread_std=spread_std,
    )


def calculate_zscore(
    spread: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate rolling z-score of the spread.

    Z-score = (spread - rolling_mean) / rolling_std

    Args:
        spread: Spread series.
        window: Rolling window for mean/std calculation.

    Returns:
        Series of z-scores.
    """
    rolling_mean = spread.rolling(window, min_periods=1).mean()
    rolling_std = spread.rolling(window, min_periods=1).std()

    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0)


def generate_pairs_signal(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    zscore_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_threshold: float = 4.0,
) -> PairsTradingSignal:
    """Generate pairs trading signal based on z-score.

    Trading logic:
    - Long pair (buy Y, sell X) when z-score < -entry_threshold
    - Short pair (sell Y, buy X) when z-score > entry_threshold
    - Exit when |z-score| < exit_threshold
    - Stop loss when |z-score| > stop_loss_threshold

    Args:
        y: First asset price series.
        x: Second asset price series.
        hedge_ratio: Hedge ratio for spread calculation.
        zscore_window: Window for z-score calculation.
        entry_threshold: Z-score threshold to enter position.
        exit_threshold: Z-score threshold to exit position.
        stop_loss_threshold: Z-score threshold for stop loss.

    Returns:
        PairsTradingSignal with current signal and parameters.
    """
    spread = calculate_spread(y, x, hedge_ratio)
    zscore = calculate_zscore(spread, zscore_window)

    current_zscore = zscore.iloc[-1] if len(zscore) > 0 else 0.0
    current_spread = spread.iloc[-1] if len(spread) > 0 else 0.0

    # Determine position
    if abs(current_zscore) > stop_loss_threshold:
        position = 0  # Stop loss - exit
    elif current_zscore < -entry_threshold:
        position = 1  # Long pair (buy Y, sell X)
    elif current_zscore > entry_threshold:
        position = -1  # Short pair (sell Y, buy X)
    elif abs(current_zscore) < exit_threshold:
        position = 0  # Exit/neutral
    else:
        position = 0  # In between - maintain neutral or existing

    return PairsTradingSignal(
        zscore=float(current_zscore),
        position=position,
        spread_value=float(current_spread),
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_threshold=stop_loss_threshold,
    )


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
) -> List[Dict]:
    """Scan a universe of assets for cointegrated pairs.

    Tests all pairwise combinations for cointegration.

    Args:
        prices: DataFrame with assets as columns, dates as index.
        significance: Significance level for cointegration test.
        min_half_life: Minimum acceptable half-life (too fast = noise).
        max_half_life: Maximum acceptable half-life (too slow = not useful).

    Returns:
        List of dicts with pair info, sorted by p-value.

    Example:
        >>> prices = pd.DataFrame({
        ...     "A": np.cumsum(np.random.randn(200)),
        ...     "B": np.cumsum(np.random.randn(200)),
        ...     "C": np.cumsum(np.random.randn(200)),
        ... })
        >>> pairs = find_cointegrated_pairs(prices)
    """
    assets = prices.columns.tolist()
    n_assets = len(assets)
    results = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            asset_y = assets[i]
            asset_x = assets[j]

            try:
                result = test_cointegration(
                    prices[asset_y],
                    prices[asset_x],
                    significance,
                )

                if result.is_cointegrated:
                    if min_half_life <= result.half_life <= max_half_life:
                        results.append({
                            "asset_y": asset_y,
                            "asset_x": asset_x,
                            "p_value": result.p_value,
                            "adf_statistic": result.adf_statistic,
                            "hedge_ratio": result.hedge_ratio,
                            "half_life": result.half_life,
                            "spread_mean": result.spread_mean,
                            "spread_std": result.spread_std,
                        })
            except Exception:
                continue

    # Sort by p-value (most significant first)
    results.sort(key=lambda x: x["p_value"])
    return results


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Calculate rolling hedge ratio for dynamic hedging.

    Hedge ratios can drift over time, so rolling estimation
    helps maintain optimal hedging.

    Args:
        y: Dependent variable series.
        x: Independent variable series.
        window: Rolling window size.

    Returns:
        Series of rolling hedge ratios.
    """
    def calc_beta(y_window, x_window):
        if len(y_window) < 2:
            return np.nan
        cov = np.cov(y_window, x_window)[0, 1]
        var_x = np.var(x_window)
        return cov / var_x if var_x > 0 else np.nan

    hedge_ratios = []
    for i in range(len(y)):
        if i < window:
            hedge_ratios.append(np.nan)
        else:
            y_win = y.iloc[i - window:i].values
            x_win = x.iloc[i - window:i].values
            hedge_ratios.append(calc_beta(y_win, x_win))

    return pd.Series(hedge_ratios, index=y.index)


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 0.0001,
) -> Tuple[pd.Series, pd.Series]:
    """Estimate dynamic hedge ratio using Kalman filter.

    The Kalman filter provides smooth, adaptive hedge ratio estimates
    that respond to changing market conditions.

    Args:
        y: Dependent variable series.
        x: Independent variable series.
        delta: Process variance (higher = more responsive).

    Returns:
        Tuple of (hedge_ratio_series, intercept_series).
    """
    n = len(y)

    # State: [intercept, beta]
    theta = np.zeros((n, 2))
    P = np.zeros((n, 2, 2))
    R = np.zeros(n)

    # Initial estimates
    theta[0] = [0, 1]  # Start with beta=1
    P[0] = np.eye(2) * 10  # High initial uncertainty

    # Observation noise (will be estimated)
    Ve = 0.001

    # Process noise
    Vw = delta / (1 - delta) * np.eye(2)

    for t in range(1, n):
        # Observation
        F = np.array([1, x.iloc[t]])

        # Prediction
        theta_pred = theta[t - 1]
        P_pred = P[t - 1] + Vw

        # Update
        y_pred = F @ theta_pred
        e = y.iloc[t] - y_pred  # Innovation

        Q = F @ P_pred @ F.T + Ve
        K = P_pred @ F.T / Q  # Kalman gain

        theta[t] = theta_pred + K * e
        P[t] = P_pred - np.outer(K, K) * Q
        R[t] = e

    intercept = pd.Series(theta[:, 0], index=y.index)
    hedge_ratio = pd.Series(theta[:, 1], index=y.index)

    return hedge_ratio, intercept
