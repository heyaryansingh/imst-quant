"""Mean reversion detection for identifying mean-reverting price behavior.

This module provides tools for detecting and quantifying mean-reverting
behavior in price series, useful for pairs trading, statistical arbitrage,
and mean reversion trading strategies.

Key concepts:
- Hurst exponent: H < 0.5 indicates mean reversion, H > 0.5 indicates trending
- ADF test: Tests null hypothesis that a unit root is present (non-stationary)
- Variance ratio: Compares variance at different lags to detect random walk
- Half-life: Time for series to revert halfway to the mean

Example:
    >>> import polars as pl
    >>> import numpy as np
    >>> from imst_quant.utils.mean_reversion import (
    ...     test_mean_reversion,
    ...     hurst_exponent,
    ...     find_mean_reverting_pairs,
    ... )
    >>> # Test a single price series
    >>> prices = pl.Series("price", np.cumsum(np.random.randn(500)) + 100)
    >>> result = test_mean_reversion(prices)
    >>> print(f"Hurst: {result.hurst_exponent:.3f}, Mean reverting: {result.is_mean_reverting}")
    >>>
    >>> # Find mean-reverting pairs in a multi-asset dataset
    >>> df = pl.DataFrame({
    ...     "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
    ...     "asset_id": ["A", "B", "C"] * 2,
    ...     "close": [100.0, 50.0, 75.0, 101.0, 50.5, 75.2],
    ... })
    >>> pairs = find_mean_reverting_pairs(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Union

import numpy as np
import polars as pl


# Type alias for flexible series input
SeriesLike = Union[pl.Series, np.ndarray, List[float]]


@dataclass
class MeanReversionResult:
    """Result from mean reversion analysis.

    Attributes:
        half_life: Estimated half-life in periods (time to revert halfway to mean).
        hurst_exponent: Hurst exponent (H < 0.5 = mean-reverting, H > 0.5 = trending).
        adf_statistic: Augmented Dickey-Fuller test statistic.
        adf_pvalue: P-value of the ADF test.
        is_mean_reverting: Whether the series exhibits mean-reverting behavior.
        variance_ratio: Variance ratio test statistic.
        confidence: Confidence level of mean reversion detection ("high", "medium", "low").
    """

    half_life: float
    hurst_exponent: float
    adf_statistic: float
    adf_pvalue: float
    is_mean_reverting: bool
    variance_ratio: float
    confidence: str


@dataclass
class VarianceRatioResult:
    """Result from variance ratio test.

    Attributes:
        vr: Variance ratio value (VR = 1 suggests random walk).
        z_stat: Z-statistic for the test.
        p_value: P-value of the test.
        is_random_walk: Whether the null hypothesis (random walk) cannot be rejected.
    """

    vr: float
    z_stat: float
    p_value: float
    is_random_walk: bool


def _to_numpy(series: SeriesLike) -> np.ndarray:
    """Convert various input types to numpy array.

    Args:
        series: Input series (pl.Series, np.ndarray, or list).

    Returns:
        Numpy array of float64 values.
    """
    if isinstance(series, pl.Series):
        return series.drop_nulls().to_numpy().astype(np.float64)
    elif isinstance(series, np.ndarray):
        return series[~np.isnan(series)].astype(np.float64)
    else:
        arr = np.array(series, dtype=np.float64)
        return arr[~np.isnan(arr)]


def hurst_exponent(
    series: SeriesLike,
    max_lag: int = 100,
) -> float:
    """Calculate Hurst exponent using rescaled range (R/S) analysis.

    The Hurst exponent characterizes the long-term memory of a time series:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (Brownian motion)
    - H > 0.5: Persistent (trending)

    Args:
        series: Price or return series to analyze.
        max_lag: Maximum lag to consider in R/S analysis.

    Returns:
        Hurst exponent value between 0 and 1.

    Example:
        >>> import numpy as np
        >>> # Mean-reverting series (Ornstein-Uhlenbeck process)
        >>> ou = np.cumsum(np.random.randn(1000) - 0.1 * np.arange(1000) * 0.001)
        >>> h = hurst_exponent(ou)
        >>> print(f"Hurst: {h:.3f}")
    """
    data = _to_numpy(series)
    n = len(data)

    if n < 20:
        return 0.5  # Not enough data, assume random walk

    # Use a range of lags for R/S analysis
    max_lag = min(max_lag, n // 2)
    lags = []
    rs_values = []

    for lag in range(10, max_lag + 1, max(1, (max_lag - 10) // 20)):
        # Number of sub-series
        n_subseries = n // lag
        if n_subseries < 1:
            continue

        rs_sum = 0.0
        valid_count = 0

        for i in range(n_subseries):
            start = i * lag
            end = start + lag
            subseries = data[start:end]

            # Mean-adjusted cumulative sum
            mean_val = np.mean(subseries)
            cumulative = np.cumsum(subseries - mean_val)

            # Range
            r = np.max(cumulative) - np.min(cumulative)

            # Standard deviation
            s = np.std(subseries, ddof=1)

            if s > 0:
                rs_sum += r / s
                valid_count += 1

        if valid_count > 0:
            lags.append(lag)
            rs_values.append(rs_sum / valid_count)

    if len(lags) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    log_lags = np.log(np.array(lags))
    log_rs = np.log(np.array(rs_values))

    # OLS for slope (Hurst exponent)
    n_points = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_xx = np.sum(log_lags * log_lags)

    denominator = n_points * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return 0.5

    hurst = (n_points * sum_xy - sum_x * sum_y) / denominator

    # Clamp to valid range
    return float(max(0.0, min(1.0, hurst)))


def variance_ratio_test(
    series: SeriesLike,
    lag: int = 2,
) -> Dict[str, float]:
    """Perform Lo-MacKinlay variance ratio test.

    Tests whether a series follows a random walk by comparing the variance
    of k-period returns to the variance of 1-period returns.

    Under the random walk hypothesis, VR(k) = 1.
    VR < 1 suggests mean reversion, VR > 1 suggests momentum.

    Args:
        series: Price series to test.
        lag: Lag period for variance comparison.

    Returns:
        Dictionary with:
        - vr: Variance ratio
        - z_stat: Z-statistic
        - p_value: P-value (two-sided)
        - is_random_walk: Whether null hypothesis cannot be rejected at 5%

    Example:
        >>> import numpy as np
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> result = variance_ratio_test(prices, lag=5)
        >>> print(f"VR: {result['vr']:.3f}, Random walk: {result['is_random_walk']}")
    """
    data = _to_numpy(series)
    n = len(data)

    if n < lag + 10:
        return {
            "vr": 1.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "is_random_walk": True,
        }

    # Calculate returns
    returns = np.diff(np.log(data[data > 0]))
    n_returns = len(returns)

    if n_returns < lag + 5:
        return {
            "vr": 1.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "is_random_walk": True,
        }

    # Variance of 1-period returns
    var_1 = np.var(returns, ddof=1)

    if var_1 < 1e-10:
        return {
            "vr": 1.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "is_random_walk": True,
        }

    # k-period returns
    k_returns = np.array([
        np.sum(returns[i:i + lag])
        for i in range(n_returns - lag + 1)
    ])

    # Variance of k-period returns
    var_k = np.var(k_returns, ddof=1)

    # Variance ratio
    vr = var_k / (lag * var_1)

    # Heteroscedasticity-consistent standard error (Lo-MacKinlay)
    # Using asymptotic variance under heteroscedasticity
    theta = 0.0
    for j in range(1, lag):
        delta_j = 0.0
        for t in range(j, n_returns):
            delta_j += (returns[t] ** 2) * (returns[t - j] ** 2)
        delta_j = delta_j / ((var_1 ** 2) * n_returns)
        theta += ((2 * (lag - j)) / lag) ** 2 * delta_j

    # Avoid division by zero
    if theta < 1e-10:
        theta = 2 * (2 * lag - 1) * (lag - 1) / (3 * lag * n_returns)

    se = np.sqrt(theta)
    z_stat = (vr - 1) / se if se > 0 else 0.0

    # Two-sided p-value using normal approximation
    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

    return {
        "vr": float(vr),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "is_random_walk": p_value > 0.05,
    }


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function approximation.

    Args:
        x: Z-score value.

    Returns:
        Cumulative probability.
    """
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def estimate_half_life(series: SeriesLike) -> float:
    """Estimate mean reversion half-life using OLS regression.

    Uses an AR(1) model: delta_y = alpha + beta * y_lag + epsilon
    Half-life = -ln(2) / ln(1 + beta)

    The half-life represents the expected time for the series to revert
    halfway to its mean value.

    Args:
        series: Price or spread series.

    Returns:
        Half-life in periods. Returns inf if series is not mean-reverting.

    Example:
        >>> import numpy as np
        >>> # Simulate mean-reverting process
        >>> n = 500
        >>> theta = 0.1  # Mean reversion speed
        >>> y = np.zeros(n)
        >>> for i in range(1, n):
        ...     y[i] = y[i-1] - theta * y[i-1] + np.random.randn() * 0.5
        >>> hl = estimate_half_life(y)
        >>> print(f"Half-life: {hl:.1f} periods")
    """
    data = _to_numpy(series)
    n = len(data)

    if n < 10:
        return float("inf")

    # Lagged values
    y_lag = data[:-1]
    delta_y = np.diff(data)

    # OLS: delta_y = alpha + beta * y_lag
    # Using matrix form: [1, y_lag] @ [alpha, beta]' = delta_y
    n_obs = len(delta_y)
    X = np.column_stack([np.ones(n_obs), y_lag])

    # Normal equations: (X'X)^-1 X'y
    try:
        XtX = X.T @ X
        Xty = X.T @ delta_y
        coeffs = np.linalg.solve(XtX, Xty)
        beta = coeffs[1]
    except np.linalg.LinAlgError:
        return float("inf")

    # Half-life calculation
    # AR(1): y_t = (1 + beta) * y_{t-1} + ...
    # phi = 1 + beta
    phi = 1 + beta

    if phi <= 0 or phi >= 1:
        # Not mean reverting (explosive or unit root)
        return float("inf")

    half_life = -np.log(2) / np.log(phi)

    return max(1.0, float(half_life))


def adf_test(
    series: SeriesLike,
    max_lag: int | None = None,
) -> Dict[str, float]:
    """Perform Augmented Dickey-Fuller test for stationarity.

    Tests the null hypothesis that a unit root is present (non-stationary).
    Rejection of null suggests the series is stationary/mean-reverting.

    Implementation follows standard ADF methodology with OLS estimation.

    Args:
        series: Price or spread series to test.
        max_lag: Maximum lag to include. If None, uses Schwert criterion.

    Returns:
        Dictionary with:
        - adf_statistic: ADF test statistic
        - p_value: Approximate p-value
        - critical_1pct: Critical value at 1% level
        - critical_5pct: Critical value at 5% level
        - critical_10pct: Critical value at 10% level
        - is_stationary: Whether null is rejected at 5% level

    Example:
        >>> import numpy as np
        >>> # Stationary series
        >>> stationary = np.random.randn(500)
        >>> result = adf_test(stationary)
        >>> print(f"ADF: {result['adf_statistic']:.3f}, Stationary: {result['is_stationary']}")
    """
    data = _to_numpy(series)
    n = len(data)

    if n < 20:
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "critical_1pct": -3.43,
            "critical_5pct": -2.86,
            "critical_10pct": -2.57,
            "is_stationary": False,
        }

    # Schwert criterion for max lag
    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)

    # First difference
    diff = np.diff(data)
    y_lag = data[:-1]

    # Align series for lagged differences
    # We need: diff[t], y_lag[t], diff[t-1], diff[t-2], ...
    start_idx = max_lag
    n_obs = n - 1 - max_lag

    if n_obs < 10:
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "critical_1pct": -3.43,
            "critical_5pct": -2.86,
            "critical_10pct": -2.57,
            "is_stationary": False,
        }

    # Build design matrix
    # X = [constant, y_lag, lagged_diffs]
    y = diff[start_idx:]
    X_cols = [np.ones(n_obs), y_lag[start_idx:]]

    for lag_i in range(1, max_lag + 1):
        X_cols.append(diff[start_idx - lag_i : n - 1 - lag_i])

    X = np.column_stack(X_cols)

    # OLS estimation
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        coeffs = XtX_inv @ (X.T @ y)

        # Residuals and standard errors
        residuals = y - X @ coeffs
        sse = np.sum(residuals ** 2)
        mse = sse / (n_obs - len(coeffs))
        var_coef = mse * np.diag(XtX_inv)
        se = np.sqrt(var_coef)

        # t-statistic for coefficient on y_lag (index 1)
        adf_stat = coeffs[1] / se[1] if se[1] > 0 else 0.0
    except np.linalg.LinAlgError:
        adf_stat = 0.0

    # Critical values (MacKinnon 1994, with constant, n > 250)
    critical_values = {
        "critical_1pct": -3.43,
        "critical_5pct": -2.86,
        "critical_10pct": -2.57,
    }

    # Approximate p-value using interpolation
    if adf_stat < -3.96:
        p_value = 0.001
    elif adf_stat < -3.43:
        p_value = 0.01
    elif adf_stat < -2.86:
        p_value = 0.05
    elif adf_stat < -2.57:
        p_value = 0.10
    elif adf_stat < -1.94:
        p_value = 0.30
    elif adf_stat < -1.61:
        p_value = 0.50
    else:
        # Tail approximation
        p_value = min(1.0, 0.5 + 0.3 * (adf_stat + 1.61))

    return {
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        **critical_values,
        "is_stationary": adf_stat < -2.86,
    }


def test_mean_reversion(
    prices: SeriesLike,
    method: str = "all",
) -> MeanReversionResult:
    """Comprehensive mean reversion analysis on a price series.

    Combines multiple tests to determine if a series exhibits
    mean-reverting behavior:
    - Augmented Dickey-Fuller test for stationarity
    - Hurst exponent via R/S analysis
    - Variance ratio test
    - Half-life estimation

    Args:
        prices: Price series (not returns) to analyze.
        method: Testing method - "all" runs all tests, or specify
                "adf", "hurst", "variance_ratio".

    Returns:
        MeanReversionResult with comprehensive analysis.

    Example:
        >>> import polars as pl
        >>> import numpy as np
        >>> # Create mean-reverting series
        >>> n = 500
        >>> theta = 0.05
        >>> spread = np.zeros(n)
        >>> for i in range(1, n):
        ...     spread[i] = spread[i-1] * (1 - theta) + np.random.randn() * 0.5
        >>> prices = pl.Series("spread", spread + 100)
        >>> result = test_mean_reversion(prices)
        >>> print(f"Mean reverting: {result.is_mean_reverting}")
        >>> print(f"Half-life: {result.half_life:.1f} periods")
        >>> print(f"Confidence: {result.confidence}")
    """
    data = _to_numpy(prices)

    if len(data) < 30:
        return MeanReversionResult(
            half_life=float("inf"),
            hurst_exponent=0.5,
            adf_statistic=0.0,
            adf_pvalue=1.0,
            is_mean_reverting=False,
            variance_ratio=1.0,
            confidence="low",
        )

    # Run all tests
    adf_result = adf_test(data)
    hurst = hurst_exponent(data)
    vr_result = variance_ratio_test(data, lag=2)
    half_life = estimate_half_life(data)

    # Determine if mean-reverting based on multiple criteria
    # Criteria:
    # 1. ADF p-value < 0.10 (reject unit root at 10%)
    # 2. Hurst < 0.5 (anti-persistent)
    # 3. Variance ratio < 1 (mean-reverting)
    # 4. Half-life is finite and reasonable

    adf_suggests_mr = adf_result["p_value"] < 0.10
    hurst_suggests_mr = hurst < 0.5
    vr_suggests_mr = vr_result["vr"] < 1.0
    half_life_reasonable = 1.0 < half_life < 252  # Between 1 day and 1 year

    # Count evidence for mean reversion
    evidence_count = sum([
        adf_suggests_mr,
        hurst_suggests_mr,
        vr_suggests_mr,
        half_life_reasonable,
    ])

    # Determine overall classification
    is_mean_reverting = evidence_count >= 2

    # Confidence level
    if evidence_count >= 4:
        confidence = "high"
    elif evidence_count >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return MeanReversionResult(
        half_life=half_life,
        hurst_exponent=hurst,
        adf_statistic=adf_result["adf_statistic"],
        adf_pvalue=adf_result["p_value"],
        is_mean_reverting=is_mean_reverting,
        variance_ratio=vr_result["vr"],
        confidence=confidence,
    )


def rolling_hurst(
    series: SeriesLike,
    window: int = 100,
) -> pl.DataFrame:
    """Calculate rolling Hurst exponent over time.

    Useful for detecting changes in mean reversion behavior over time.

    Args:
        series: Price series to analyze.
        window: Rolling window size for Hurst calculation.

    Returns:
        DataFrame with columns: period, hurst.

    Example:
        >>> import numpy as np
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> rolling = rolling_hurst(prices, window=50)
        >>> print(rolling.tail())
    """
    data = _to_numpy(series)
    n = len(data)

    if n < window:
        return pl.DataFrame({
            "period": pl.Series([], dtype=pl.Int64),
            "hurst": pl.Series([], dtype=pl.Float64),
        })

    periods = []
    hurst_values = []

    for i in range(window, n + 1):
        window_data = data[i - window : i]
        h = hurst_exponent(window_data, max_lag=min(50, window // 2))
        periods.append(i - 1)  # 0-indexed period
        hurst_values.append(h)

    return pl.DataFrame({
        "period": periods,
        "hurst": hurst_values,
    })


def _calculate_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """Calculate optimal hedge ratio using OLS.

    Args:
        y: Dependent variable (prices of first asset).
        x: Independent variable (prices of second asset).

    Returns:
        Hedge ratio (beta coefficient).
    """
    n = len(y)
    if n < 10:
        return 1.0

    # OLS: y = alpha + beta * x
    X = np.column_stack([np.ones(n), x])
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(coeffs[1])
    except np.linalg.LinAlgError:
        return 1.0


def find_mean_reverting_pairs(
    prices_df: pl.DataFrame,
    price_col: str = "close",
    asset_col: str = "asset_id",
    date_col: str = "date",
    max_half_life: float = 60.0,
    min_half_life: float = 1.0,
    min_observations: int = 100,
) -> pl.DataFrame:
    """Find mean-reverting pairs from a multi-asset price dataset.

    Tests all pairwise combinations of assets for mean reversion of their
    spread, useful for pairs trading strategy development.

    Args:
        prices_df: DataFrame with multiple assets' prices.
        price_col: Column name for price data.
        asset_col: Column name for asset identifier.
        date_col: Column name for date (used for alignment).
        max_half_life: Maximum acceptable half-life in periods.
        min_half_life: Minimum acceptable half-life (too fast = noise).
        min_observations: Minimum overlapping observations required.

    Returns:
        DataFrame with columns: asset_1, asset_2, half_life, hurst,
        is_mean_reverting, hedge_ratio. Sorted by half_life ascending.

    Example:
        >>> import polars as pl
        >>> import numpy as np
        >>> # Create sample multi-asset data
        >>> n = 200
        >>> dates = [f"2024-{i//30 + 1:02d}-{i%30 + 1:02d}" for i in range(n)]
        >>> df = pl.DataFrame({
        ...     "date": dates * 3,
        ...     "asset_id": ["AAPL"] * n + ["GOOGL"] * n + ["MSFT"] * n,
        ...     "close": list(np.cumsum(np.random.randn(n))) +
        ...              list(np.cumsum(np.random.randn(n))) +
        ...              list(np.cumsum(np.random.randn(n))),
        ... })
        >>> pairs = find_mean_reverting_pairs(df, max_half_life=30)
        >>> print(pairs)
    """
    # Get unique assets
    assets = prices_df[asset_col].unique().to_list()
    n_assets = len(assets)

    if n_assets < 2:
        return pl.DataFrame({
            "asset_1": pl.Series([], dtype=pl.Utf8),
            "asset_2": pl.Series([], dtype=pl.Utf8),
            "half_life": pl.Series([], dtype=pl.Float64),
            "hurst": pl.Series([], dtype=pl.Float64),
            "is_mean_reverting": pl.Series([], dtype=pl.Boolean),
            "hedge_ratio": pl.Series([], dtype=pl.Float64),
        })

    # Pivot to wide format for easier pair analysis
    pivot_df = prices_df.pivot(
        values=price_col,
        index=date_col,
        on=asset_col,
    ).sort(date_col)

    results: List[Dict] = []

    # Test all pairs
    for asset1, asset2 in combinations(assets, 2):
        if asset1 not in pivot_df.columns or asset2 not in pivot_df.columns:
            continue

        # Get aligned prices
        mask = pivot_df[asset1].is_not_null() & pivot_df[asset2].is_not_null()
        filtered = pivot_df.filter(mask)

        if filtered.height < min_observations:
            continue

        prices1 = filtered[asset1].to_numpy().astype(np.float64)
        prices2 = filtered[asset2].to_numpy().astype(np.float64)

        # Calculate hedge ratio
        hedge_ratio = _calculate_hedge_ratio(prices1, prices2)

        # Calculate spread
        spread = prices1 - hedge_ratio * prices2

        # Test spread for mean reversion
        mr_result = test_mean_reversion(spread)

        # Filter by half-life criteria
        if min_half_life <= mr_result.half_life <= max_half_life:
            results.append({
                "asset_1": asset1,
                "asset_2": asset2,
                "half_life": mr_result.half_life,
                "hurst": mr_result.hurst_exponent,
                "is_mean_reverting": mr_result.is_mean_reverting,
                "hedge_ratio": hedge_ratio,
                "adf_pvalue": mr_result.adf_pvalue,
                "variance_ratio": mr_result.variance_ratio,
                "confidence": mr_result.confidence,
            })

    if not results:
        return pl.DataFrame({
            "asset_1": pl.Series([], dtype=pl.Utf8),
            "asset_2": pl.Series([], dtype=pl.Utf8),
            "half_life": pl.Series([], dtype=pl.Float64),
            "hurst": pl.Series([], dtype=pl.Float64),
            "is_mean_reverting": pl.Series([], dtype=pl.Boolean),
            "hedge_ratio": pl.Series([], dtype=pl.Float64),
        })

    # Create DataFrame and sort by half_life
    result_df = pl.DataFrame(results).sort("half_life")

    return result_df


def rolling_variance_ratio(
    series: SeriesLike,
    window: int = 100,
    lag: int = 2,
) -> pl.DataFrame:
    """Calculate rolling variance ratio over time.

    Args:
        series: Price series to analyze.
        window: Rolling window size.
        lag: Lag for variance ratio calculation.

    Returns:
        DataFrame with columns: period, variance_ratio, z_stat.

    Example:
        >>> import numpy as np
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> rolling_vr = rolling_variance_ratio(prices, window=50)
        >>> print(rolling_vr.tail())
    """
    data = _to_numpy(series)
    n = len(data)

    if n < window:
        return pl.DataFrame({
            "period": pl.Series([], dtype=pl.Int64),
            "variance_ratio": pl.Series([], dtype=pl.Float64),
            "z_stat": pl.Series([], dtype=pl.Float64),
        })

    periods = []
    vr_values = []
    z_values = []

    for i in range(window, n + 1):
        window_data = data[i - window : i]
        vr_result = variance_ratio_test(window_data, lag=lag)
        periods.append(i - 1)
        vr_values.append(vr_result["vr"])
        z_values.append(vr_result["z_stat"])

    return pl.DataFrame({
        "period": periods,
        "variance_ratio": vr_values,
        "z_stat": z_values,
    })


def rolling_half_life(
    series: SeriesLike,
    window: int = 100,
) -> pl.DataFrame:
    """Calculate rolling half-life over time.

    Args:
        series: Price series to analyze.
        window: Rolling window size.

    Returns:
        DataFrame with columns: period, half_life.

    Example:
        >>> import numpy as np
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> rolling_hl = rolling_half_life(prices, window=50)
        >>> print(rolling_hl.tail())
    """
    data = _to_numpy(series)
    n = len(data)

    if n < window:
        return pl.DataFrame({
            "period": pl.Series([], dtype=pl.Int64),
            "half_life": pl.Series([], dtype=pl.Float64),
        })

    periods = []
    half_life_values = []

    for i in range(window, n + 1):
        window_data = data[i - window : i]
        hl = estimate_half_life(window_data)
        periods.append(i - 1)
        half_life_values.append(hl)

    return pl.DataFrame({
        "period": periods,
        "half_life": half_life_values,
    })


def generate_mean_reversion_report(
    prices: SeriesLike,
    asset_name: str = "Asset",
) -> str:
    """Generate a comprehensive mean reversion analysis report.

    Args:
        prices: Price series to analyze.
        asset_name: Name of the asset for the report.

    Returns:
        Formatted string with mean reversion analysis.

    Example:
        >>> import numpy as np
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> report = generate_mean_reversion_report(prices, "BTC/USD Spread")
        >>> print(report)
    """
    result = test_mean_reversion(prices)
    vr_result = variance_ratio_test(prices)
    adf_result = adf_test(prices)

    lines = [
        "=" * 60,
        f"MEAN REVERSION ANALYSIS: {asset_name}",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"Is Mean Reverting:   {'Yes' if result.is_mean_reverting else 'No'}",
        f"Confidence Level:    {result.confidence.upper()}",
        f"Half-Life:           {result.half_life:.1f} periods" if result.half_life < 1e6 else "Half-Life:           N/A (not mean reverting)",
        "",
        "HURST EXPONENT ANALYSIS",
        "-" * 40,
        f"Hurst Exponent:      {result.hurst_exponent:.4f}",
        f"Interpretation:      {'Mean-reverting (H < 0.5)' if result.hurst_exponent < 0.5 else 'Trending (H > 0.5)' if result.hurst_exponent > 0.5 else 'Random walk (H = 0.5)'}",
        "",
        "AUGMENTED DICKEY-FULLER TEST",
        "-" * 40,
        f"ADF Statistic:       {result.adf_statistic:.4f}",
        f"P-Value:             {result.adf_pvalue:.4f}",
        f"Critical (1%):       {adf_result['critical_1pct']:.4f}",
        f"Critical (5%):       {adf_result['critical_5pct']:.4f}",
        f"Critical (10%):      {adf_result['critical_10pct']:.4f}",
        f"Stationary at 5%:    {'Yes' if adf_result['is_stationary'] else 'No'}",
        "",
        "VARIANCE RATIO TEST",
        "-" * 40,
        f"Variance Ratio:      {result.variance_ratio:.4f}",
        f"Z-Statistic:         {vr_result['z_stat']:.4f}",
        f"P-Value:             {vr_result['p_value']:.4f}",
        f"Random Walk:         {'Yes' if vr_result['is_random_walk'] else 'No'}",
        "",
        "TRADING IMPLICATIONS",
        "-" * 40,
    ]

    if result.is_mean_reverting and result.confidence in ["high", "medium"]:
        lines.extend([
            f"- Suitable for mean reversion strategies",
            f"- Expected reversion time: ~{result.half_life:.0f} periods to 50%",
            f"- Entry: When price deviates >2 std from mean",
            f"- Exit: When price returns to mean",
        ])
    else:
        lines.extend([
            f"- NOT recommended for mean reversion strategies",
            f"- Consider momentum or trend-following approaches",
        ])

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
