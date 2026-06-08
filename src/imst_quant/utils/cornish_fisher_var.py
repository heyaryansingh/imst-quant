"""Cornish-Fisher VaR and modified risk metrics.

This module extends standard VaR with Cornish-Fisher expansion that adjusts
for skewness and excess kurtosis in return distributions. Most financial
returns are non-normal, so the CF expansion provides more accurate tail
risk estimates than parametric (Gaussian) VaR.

Functions:
    cornish_fisher_var: VaR adjusted for skewness and kurtosis
    cornish_fisher_cvar: Expected Shortfall using CF-adjusted quantile
    modified_sharpe_ratio: Sharpe ratio using CF-VaR as risk denominator
    jarque_bera_test: Test whether returns are normally distributed
    risk_summary: Comprehensive risk summary comparing standard vs CF metrics

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.cornish_fisher_var import cornish_fisher_var
    >>> returns = pl.Series("r", [-0.05, -0.03, 0.01, 0.02, -0.04, 0.015])
    >>> cf_var = cornish_fisher_var(returns, confidence_level=0.95)
    >>> print(f"95% CF-VaR: {cf_var:.4f}")
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class RiskComparison:
    """Comparison of standard vs Cornish-Fisher risk metrics."""

    parametric_var: float
    cornish_fisher_var: float
    historical_var: float
    parametric_cvar: float
    cornish_fisher_cvar: float
    skewness: float
    excess_kurtosis: float
    is_normal: bool
    jarque_bera_pvalue: float
    cf_adjustment_pct: float


def _extract_series(
    returns: Union[pl.Series, pl.DataFrame], return_col: str = "returns"
) -> pl.Series:
    """Extract a Polars Series from Series or DataFrame input."""
    if isinstance(returns, pl.DataFrame):
        return returns[return_col]
    return returns


def cornish_fisher_var(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    return_col: str = "returns",
) -> float:
    """Calculate Value at Risk using the Cornish-Fisher expansion.

    The Cornish-Fisher expansion adjusts the normal quantile for skewness
    and excess kurtosis, providing a more accurate VaR estimate for
    non-normal return distributions.

    The adjusted quantile is:
        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*K/24 - (2*z^3 - 5*z)*S^2/36

    where z is the standard normal quantile, S is skewness, and K is
    excess kurtosis.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.95 = 95%).
        return_col: Column name if returns is a DataFrame.

    Returns:
        CF-VaR as a positive decimal (estimated loss at the given confidence).

    Example:
        >>> returns = pl.Series([0.01, -0.03, 0.02, -0.05, 0.015, -0.02])
        >>> var_95 = cornish_fisher_var(returns, confidence_level=0.95)
        >>> print(f"95% CF-VaR: {var_95:.4f}")
    """
    ret = _extract_series(returns, return_col)
    arr = ret.drop_nulls().to_numpy().astype(np.float64)

    if len(arr) < 4:
        return 0.0

    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)

    if sigma == 0:
        return 0.0

    s = float(stats.skew(arr, bias=False))
    k = float(stats.kurtosis(arr, bias=False))  # excess kurtosis

    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )

    cf_var = -(mu + z_cf * sigma)
    return max(cf_var, 0.0)


def cornish_fisher_cvar(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    return_col: str = "returns",
) -> float:
    """Calculate Expected Shortfall using Cornish-Fisher adjusted threshold.

    Uses the CF-adjusted VaR as the threshold, then computes the mean of
    returns falling below that threshold. This gives a tail-risk measure
    that respects the non-normality of the return distribution.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.95).
        return_col: Column name if returns is a DataFrame.

    Returns:
        CF-CVaR as a positive decimal.

    Example:
        >>> returns = pl.Series([0.01, -0.03, 0.02, -0.05, 0.015])
        >>> cvar = cornish_fisher_cvar(returns, confidence_level=0.95)
    """
    ret = _extract_series(returns, return_col)
    arr = ret.drop_nulls().to_numpy().astype(np.float64)

    if len(arr) < 4:
        return 0.0

    cf_var_val = cornish_fisher_var(returns, confidence_level, return_col)
    threshold = -cf_var_val

    tail_losses = arr[arr <= threshold]
    if len(tail_losses) == 0:
        # If no observations beyond CF-VaR, fall back to worst observation
        return float(-np.min(arr))

    return float(-np.mean(tail_losses))


def modified_sharpe_ratio(
    returns: Union[pl.Series, pl.DataFrame],
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    return_col: str = "returns",
) -> float:
    """Calculate Modified Sharpe Ratio using CF-VaR as the risk denominator.

    The Modified Sharpe Ratio replaces standard deviation with CF-VaR,
    providing a risk-adjusted return metric that accounts for skewness
    and fat tails.

    MSR = (mean_return - risk_free_rate) / CF_VaR

    Args:
        returns: Series or DataFrame containing return data.
        risk_free_rate: Daily risk-free rate (default: 0).
        confidence_level: Confidence level for CF-VaR (default: 0.95).
        periods_per_year: For annualization (252=daily, 52=weekly, 12=monthly).
        return_col: Column name if returns is a DataFrame.

    Returns:
        Modified Sharpe ratio (annualized). Returns 0.0 if CF-VaR is zero.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> msr = modified_sharpe_ratio(returns, risk_free_rate=0.0001)
    """
    ret = _extract_series(returns, return_col)
    arr = ret.drop_nulls().to_numpy().astype(np.float64)

    if len(arr) < 4:
        return 0.0

    mean_excess = np.mean(arr) - risk_free_rate
    cf_var_val = cornish_fisher_var(returns, confidence_level, return_col)

    if cf_var_val == 0:
        return 0.0

    daily_msr = mean_excess / cf_var_val
    return float(daily_msr * np.sqrt(periods_per_year))


def jarque_bera_test(
    returns: Union[pl.Series, pl.DataFrame],
    return_col: str = "returns",
    significance: float = 0.05,
) -> Dict[str, float]:
    """Run the Jarque-Bera normality test on returns.

    Tests the null hypothesis that the return distribution has the
    skewness and kurtosis of a normal distribution.

    Args:
        returns: Series or DataFrame containing return data.
        return_col: Column name if returns is a DataFrame.
        significance: Significance level for normality decision (default: 0.05).

    Returns:
        Dictionary with keys:
        - statistic: JB test statistic
        - pvalue: p-value of the test
        - skewness: Sample skewness
        - excess_kurtosis: Sample excess kurtosis
        - is_normal: Whether we fail to reject normality at given significance

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> result = jarque_bera_test(returns)
        >>> print(f"Normal: {result['is_normal']}, p={result['pvalue']:.4f}")
    """
    ret = _extract_series(returns, return_col)
    arr = ret.drop_nulls().to_numpy().astype(np.float64)

    if len(arr) < 4:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "is_normal": True,
        }

    jb_stat, jb_pvalue = stats.jarque_bera(arr)
    s = float(stats.skew(arr, bias=False))
    k = float(stats.kurtosis(arr, bias=False))

    return {
        "statistic": float(jb_stat),
        "pvalue": float(jb_pvalue),
        "skewness": s,
        "excess_kurtosis": k,
        "is_normal": bool(jb_pvalue >= significance),
    }


def risk_summary(
    returns: Union[pl.Series, pl.DataFrame],
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    return_col: str = "returns",
) -> RiskComparison:
    """Generate comprehensive risk summary comparing standard vs CF metrics.

    Computes parametric (Gaussian) VaR, historical VaR, and Cornish-Fisher
    VaR side by side so users can see the impact of non-normality adjustments.

    Args:
        returns: Series or DataFrame containing return data.
        confidence_level: Confidence level (default: 0.95).
        risk_free_rate: Daily risk-free rate for Sharpe calculations.
        return_col: Column name if returns is a DataFrame.

    Returns:
        RiskComparison dataclass with all metrics.

    Example:
        >>> returns = pl.Series([0.01, -0.03, 0.02, -0.05, 0.015, -0.02])
        >>> summary = risk_summary(returns, confidence_level=0.99)
        >>> print(f"CF adjustment: {summary.cf_adjustment_pct:+.1f}%")
    """
    ret = _extract_series(returns, return_col)
    arr = ret.drop_nulls().to_numpy().astype(np.float64)

    if len(arr) < 4:
        return RiskComparison(
            parametric_var=0.0,
            cornish_fisher_var=0.0,
            historical_var=0.0,
            parametric_cvar=0.0,
            cornish_fisher_cvar=0.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            is_normal=True,
            jarque_bera_pvalue=1.0,
            cf_adjustment_pct=0.0,
        )

    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)

    # Parametric (Gaussian) VaR
    param_var = -(mu + z * sigma)
    param_var = max(param_var, 0.0)

    # Parametric CVaR (Gaussian assumption)
    phi_z = stats.norm.pdf(z)
    param_cvar = -(mu - sigma * phi_z / alpha)
    param_cvar = max(param_cvar, 0.0)

    # Historical VaR
    hist_var = float(-np.percentile(arr, alpha * 100))
    hist_var = max(hist_var, 0.0)

    # Cornish-Fisher VaR
    cf_var_val = cornish_fisher_var(returns, confidence_level, return_col)
    cf_cvar_val = cornish_fisher_cvar(returns, confidence_level, return_col)

    # Normality test
    jb = jarque_bera_test(returns, return_col)

    # CF adjustment percentage relative to parametric
    if param_var > 0:
        cf_adj_pct = ((cf_var_val - param_var) / param_var) * 100
    else:
        cf_adj_pct = 0.0

    return RiskComparison(
        parametric_var=param_var,
        cornish_fisher_var=cf_var_val,
        historical_var=hist_var,
        parametric_cvar=param_cvar,
        cornish_fisher_cvar=cf_cvar_val,
        skewness=jb["skewness"],
        excess_kurtosis=jb["excess_kurtosis"],
        is_normal=jb["is_normal"],
        jarque_bera_pvalue=jb["pvalue"],
        cf_adjustment_pct=cf_adj_pct,
    )
