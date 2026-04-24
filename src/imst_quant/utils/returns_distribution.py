"""Returns distribution analysis module for IMST-Quant.

This module provides statistical analysis of return distributions including:
- Descriptive statistics (skewness, kurtosis, moments)
- Normality testing (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)
- Tail risk analysis (tail ratios, extreme value statistics)
- Distribution fitting and comparison

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.Series(np.random.randn(252) * 0.01)
    >>>
    >>> # Full distribution analysis
    >>> stats = analyze_distribution(returns)
    >>> print(f"Skewness: {stats.skewness:.4f}")
    >>> print(f"Kurtosis: {stats.kurtosis:.4f}")
    >>> print(f"Normal: {stats.is_normal}")
    >>>
    >>> # Normality tests
    >>> tests = test_normality(returns)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class DistributionStats:
    """Comprehensive distribution statistics.

    Attributes:
        mean: Arithmetic mean of returns.
        std: Standard deviation of returns.
        skewness: Third moment (asymmetry).
        kurtosis: Fourth moment (tail heaviness), excess kurtosis.
        min_return: Minimum return observed.
        max_return: Maximum return observed.
        median: Median return.
        is_normal: Whether distribution passes normality test.
        jarque_bera_stat: Jarque-Bera test statistic.
        jarque_bera_pvalue: Jarque-Bera p-value.
        n_observations: Number of observations.
    """

    mean: float
    std: float
    skewness: float
    kurtosis: float
    min_return: float
    max_return: float
    median: float
    is_normal: bool
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    n_observations: int


@dataclass
class NormalityTests:
    """Results of multiple normality tests.

    Attributes:
        jarque_bera: Tuple of (statistic, p-value).
        shapiro_wilk: Tuple of (statistic, p-value) or None if n > 5000.
        anderson_darling: Tuple of (statistic, critical_values, significance_levels).
        kolmogorov_smirnov: Tuple of (statistic, p-value).
        is_normal_5pct: Whether distribution passes at 5% significance.
        is_normal_1pct: Whether distribution passes at 1% significance.
    """

    jarque_bera: tuple[float, float]
    shapiro_wilk: Optional[tuple[float, float]]
    anderson_darling: tuple[float, list[float], list[float]]
    kolmogorov_smirnov: tuple[float, float]
    is_normal_5pct: bool
    is_normal_1pct: bool


@dataclass
class TailAnalysis:
    """Tail risk analysis results.

    Attributes:
        left_tail_ratio: Ratio of observations below -2 std.
        right_tail_ratio: Ratio of observations above +2 std.
        tail_asymmetry: Difference between right and left tail ratios.
        var_95: 95% Value at Risk (empirical).
        var_99: 99% Value at Risk (empirical).
        expected_shortfall_95: 95% Expected Shortfall (CVaR).
        expected_shortfall_99: 99% Expected Shortfall (CVaR).
        max_loss_5_worst: Average of 5 worst returns.
        gain_loss_ratio: Ratio of average gains to average losses.
    """

    left_tail_ratio: float
    right_tail_ratio: float
    tail_asymmetry: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_loss_5_worst: float
    gain_loss_ratio: float


def calculate_moments(returns: pd.Series) -> dict[str, float]:
    """Calculate raw and central moments of returns.

    Args:
        returns: Series of returns.

    Returns:
        Dictionary with moment calculations.
    """
    n = len(returns)
    mean = returns.mean()
    std = returns.std()

    # Central moments
    m2 = ((returns - mean) ** 2).mean()
    m3 = ((returns - mean) ** 3).mean()
    m4 = ((returns - mean) ** 4).mean()

    # Standardized moments
    skewness = m3 / (std ** 3) if std > 0 else 0.0
    kurtosis = (m4 / (std ** 4)) - 3.0 if std > 0 else 0.0  # Excess kurtosis

    return {
        "mean": float(mean),
        "std": float(std),
        "variance": float(m2),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "n": n,
    }


def analyze_distribution(returns: pd.Series) -> DistributionStats:
    """Perform comprehensive distribution analysis.

    Args:
        returns: Series of returns.

    Returns:
        DistributionStats with all statistics.

    Example:
        >>> returns = pd.Series(np.random.randn(500) * 0.01)
        >>> stats = analyze_distribution(returns)
        >>> print(f"Distribution is {'normal' if stats.is_normal else 'non-normal'}")
    """
    moments = calculate_moments(returns)

    # Jarque-Bera test
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns.dropna())

    return DistributionStats(
        mean=moments["mean"],
        std=moments["std"],
        skewness=moments["skewness"],
        kurtosis=moments["kurtosis"],
        min_return=float(returns.min()),
        max_return=float(returns.max()),
        median=float(returns.median()),
        is_normal=jb_pvalue > 0.05,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_pvalue),
        n_observations=len(returns),
    )


def test_normality(
    returns: pd.Series,
    significance: float = 0.05,
) -> NormalityTests:
    """Run multiple normality tests on returns.

    Args:
        returns: Series of returns.
        significance: Significance level for testing.

    Returns:
        NormalityTests with results from multiple tests.
    """
    clean_returns = returns.dropna().values
    n = len(clean_returns)

    # Jarque-Bera test
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(clean_returns)

    # Shapiro-Wilk (only for n <= 5000)
    if n <= 5000:
        sw_stat, sw_pvalue = scipy_stats.shapiro(clean_returns)
        shapiro_wilk = (float(sw_stat), float(sw_pvalue))
    else:
        shapiro_wilk = None

    # Anderson-Darling
    ad_result = scipy_stats.anderson(clean_returns, dist="norm")
    anderson_darling = (
        float(ad_result.statistic),
        list(ad_result.critical_values),
        list(ad_result.significance_level),
    )

    # Kolmogorov-Smirnov
    # Standardize returns for KS test against standard normal
    standardized = (clean_returns - clean_returns.mean()) / clean_returns.std()
    ks_stat, ks_pvalue = scipy_stats.kstest(standardized, "norm")

    # Determine normality at different significance levels
    is_normal_5pct = jb_pvalue > 0.05 and ks_pvalue > 0.05
    is_normal_1pct = jb_pvalue > 0.01 and ks_pvalue > 0.01

    return NormalityTests(
        jarque_bera=(float(jb_stat), float(jb_pvalue)),
        shapiro_wilk=shapiro_wilk,
        anderson_darling=anderson_darling,
        kolmogorov_smirnov=(float(ks_stat), float(ks_pvalue)),
        is_normal_5pct=is_normal_5pct,
        is_normal_1pct=is_normal_1pct,
    )


def analyze_tails(returns: pd.Series) -> TailAnalysis:
    """Analyze tail behavior of returns distribution.

    Args:
        returns: Series of returns.

    Returns:
        TailAnalysis with tail risk metrics.

    Example:
        >>> returns = pd.Series(np.random.randn(1000) * 0.01)
        >>> tails = analyze_tails(returns)
        >>> print(f"VaR 95%: {tails.var_95:.4f}")
        >>> print(f"Expected Shortfall 95%: {tails.expected_shortfall_95:.4f}")
    """
    clean_returns = returns.dropna()
    mean = clean_returns.mean()
    std = clean_returns.std()

    # Tail ratios (beyond 2 standard deviations)
    left_tail = (clean_returns < mean - 2 * std).mean()
    right_tail = (clean_returns > mean + 2 * std).mean()

    # Empirical VaR
    var_95 = float(clean_returns.quantile(0.05))
    var_99 = float(clean_returns.quantile(0.01))

    # Expected Shortfall (CVaR)
    es_95 = float(clean_returns[clean_returns <= var_95].mean())
    es_99 = float(clean_returns[clean_returns <= var_99].mean())

    # Worst returns
    sorted_returns = clean_returns.sort_values()
    worst_5 = float(sorted_returns.iloc[:5].mean())

    # Gain/Loss ratio
    gains = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]
    if len(losses) > 0 and len(gains) > 0:
        gain_loss_ratio = abs(gains.mean() / losses.mean())
    else:
        gain_loss_ratio = np.nan

    return TailAnalysis(
        left_tail_ratio=float(left_tail),
        right_tail_ratio=float(right_tail),
        tail_asymmetry=float(right_tail - left_tail),
        var_95=var_95,
        var_99=var_99,
        expected_shortfall_95=es_95,
        expected_shortfall_99=es_99,
        max_loss_5_worst=worst_5,
        gain_loss_ratio=float(gain_loss_ratio),
    )


def rolling_skewness(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Calculate rolling skewness.

    Args:
        returns: Series of returns.
        window: Rolling window size.

    Returns:
        Series of rolling skewness values.
    """
    return returns.rolling(window=window).skew()


def rolling_kurtosis(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Calculate rolling excess kurtosis.

    Args:
        returns: Series of returns.
        window: Rolling window size.

    Returns:
        Series of rolling excess kurtosis values.
    """
    return returns.rolling(window=window).kurt()


def quantile_comparison(
    returns: pd.Series,
    quantiles: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Compare empirical vs theoretical (normal) quantiles.

    Args:
        returns: Series of returns.
        quantiles: List of quantiles to compare.

    Returns:
        DataFrame comparing empirical and theoretical quantiles.
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    clean_returns = returns.dropna()
    mean = clean_returns.mean()
    std = clean_returns.std()

    results = []
    for q in quantiles:
        empirical = clean_returns.quantile(q)
        theoretical = scipy_stats.norm.ppf(q, loc=mean, scale=std)
        diff = empirical - theoretical
        results.append({
            "quantile": q,
            "empirical": empirical,
            "theoretical": theoretical,
            "difference": diff,
            "ratio": empirical / theoretical if theoretical != 0 else np.nan,
        })

    return pd.DataFrame(results)


def distribution_summary(
    returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
) -> dict:
    """Generate comprehensive distribution summary.

    Args:
        returns: Series of returns.
        annualize: Whether to annualize return and volatility.
        trading_days: Number of trading days per year.

    Returns:
        Dictionary with all distribution metrics.
    """
    dist_stats = analyze_distribution(returns)
    tail_stats = analyze_tails(returns)
    normality = test_normality(returns)

    ann_factor = np.sqrt(trading_days) if annualize else 1.0
    ret_factor = trading_days if annualize else 1.0

    return {
        "return": {
            "mean": dist_stats.mean * ret_factor,
            "median": dist_stats.median * ret_factor,
            "std": dist_stats.std * ann_factor,
        },
        "distribution": {
            "skewness": dist_stats.skewness,
            "kurtosis": dist_stats.kurtosis,
            "min": dist_stats.min_return,
            "max": dist_stats.max_return,
        },
        "normality": {
            "is_normal": normality.is_normal_5pct,
            "jarque_bera_pvalue": normality.jarque_bera[1],
            "ks_pvalue": normality.kolmogorov_smirnov[1],
        },
        "tail_risk": {
            "var_95": tail_stats.var_95,
            "var_99": tail_stats.var_99,
            "es_95": tail_stats.expected_shortfall_95,
            "es_99": tail_stats.expected_shortfall_99,
            "gain_loss_ratio": tail_stats.gain_loss_ratio,
        },
        "observations": dist_stats.n_observations,
    }


def compare_periods(
    returns: pd.Series,
    split_date: Optional[str] = None,
) -> pd.DataFrame:
    """Compare distribution statistics across two time periods.

    Args:
        returns: Series of returns with datetime index.
        split_date: Date to split periods. If None, splits in half.

    Returns:
        DataFrame comparing statistics across periods.
    """
    if split_date is not None:
        period1 = returns[returns.index < split_date]
        period2 = returns[returns.index >= split_date]
    else:
        mid = len(returns) // 2
        period1 = returns.iloc[:mid]
        period2 = returns.iloc[mid:]

    stats1 = analyze_distribution(period1)
    stats2 = analyze_distribution(period2)

    return pd.DataFrame({
        "metric": ["mean", "std", "skewness", "kurtosis", "is_normal"],
        "period_1": [stats1.mean, stats1.std, stats1.skewness, stats1.kurtosis, stats1.is_normal],
        "period_2": [stats2.mean, stats2.std, stats2.skewness, stats2.kurtosis, stats2.is_normal],
    })
