"""VaR model backtesting via Kupiec and Christoffersen tests.

Validates whether a Value at Risk model produces violation rates consistent
with its stated confidence level (unconditional coverage) and whether
violations are independently distributed over time (no clustering).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def compute_violations(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
) -> np.ndarray:
    """Determine which periods breached (exceeded) the VaR forecast.

    Args:
        returns: Realized returns for each period.
        var_forecasts: VaR forecast for each period, expressed as a positive
            number representing the maximum expected loss (matching the
            convention used by ``VaRCalculator``).

    Returns:
        Boolean numpy array, True where the realized loss exceeded the VaR
        forecast (i.e. a violation occurred).
    """
    returns_arr = np.asarray(returns, dtype=float)
    var_arr = np.asarray(var_forecasts, dtype=float)
    if returns_arr.shape != var_arr.shape:
        raise ValueError("returns and var_forecasts must have the same shape")
    return returns_arr < -np.abs(var_arr)


def kupiec_pof_test(
    violations: ArrayLike,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Kupiec Proportion of Failures (unconditional coverage) test.

    Tests whether the observed violation rate matches the expected
    violation rate implied by the VaR confidence level.

    Args:
        violations: Boolean sequence, True where a VaR breach occurred.
        confidence_level: VaR confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        Dict with likelihood ratio statistic, p-value, violation counts,
        and a reject_null flag at the 95% test confidence level.
    """
    v = np.asarray(violations, dtype=bool)
    n = len(v)
    if n == 0:
        raise ValueError("violations must be non-empty")

    x = int(v.sum())
    p = 1 - confidence_level
    p_hat = x / n

    # Likelihood ratio statistic; guard against log(0) at the boundaries.
    def _log_likelihood(prob: float) -> float:
        if prob <= 0.0 or prob >= 1.0:
            return 0.0
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    ll_null = _log_likelihood(p)
    ll_alt = _log_likelihood(p_hat) if 0 < p_hat < 1 else 0.0
    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = max(lr_stat, 0.0)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "test": "kupiec_pof",
        "num_observations": n,
        "num_violations": x,
        "expected_violations": round(n * p, 2),
        "violation_rate": p_hat,
        "expected_rate": p,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_null": bool(p_value < 0.05),
    }


def christoffersen_independence_test(violations: ArrayLike) -> Dict[str, Any]:
    """Christoffersen independence test for clustering of VaR violations.

    A well-specified VaR model should produce violations that are
    independently distributed over time. Clustering (violations following
    violations) suggests the model fails to adapt to changing risk.

    Args:
        violations: Boolean sequence, True where a VaR breach occurred.

    Returns:
        Dict with the likelihood ratio statistic, p-value, transition
        counts, and a reject_null flag at the 95% test confidence level.
    """
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        raise ValueError("violations must contain at least 2 observations")

    prev = v[:-1]
    curr = v[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n0 = n00 + n01
    n1 = n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    def _log_lik(p01: float, p11: float) -> float:
        ll = 0.0
        if n0 > 0:
            if 0 < p01 < 1:
                ll += n00 * np.log(1 - p01) + n01 * np.log(p01)
        if n1 > 0:
            if 0 < p11 < 1:
                ll += n10 * np.log(1 - p11) + n11 * np.log(p11)
        return ll

    ll_unrestricted = _log_lik(pi01, pi11)
    ll_restricted = _log_lik(pi, pi)

    lr_stat = -2 * (ll_restricted - ll_unrestricted)
    lr_stat = max(lr_stat, 0.0)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "test": "christoffersen_independence",
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_null": bool(p_value < 0.05),
    }


def christoffersen_conditional_coverage_test(
    violations: ArrayLike,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Combined conditional coverage test (Kupiec + independence).

    Args:
        violations: Boolean sequence, True where a VaR breach occurred.
        confidence_level: VaR confidence level used for the coverage test.

    Returns:
        Dict with the combined likelihood ratio statistic (df=2), p-value,
        and a reject_null flag, plus the individual component results.
    """
    pof = kupiec_pof_test(violations, confidence_level)
    indep = christoffersen_independence_test(violations)

    lr_cc = pof["lr_statistic"] + indep["lr_statistic"]
    p_value = 1 - stats.chi2.cdf(lr_cc, df=2)

    return {
        "test": "christoffersen_conditional_coverage",
        "lr_statistic": lr_cc,
        "p_value": p_value,
        "reject_null": bool(p_value < 0.05),
        "unconditional_coverage": pof,
        "independence": indep,
    }


def var_backtest_summary(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Full VaR backtest report combining all diagnostics.

    Args:
        returns: Realized returns for each period.
        var_forecasts: VaR forecast for each period (positive loss numbers).
        confidence_level: VaR confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        Dict with violation series stats and the Kupiec, independence, and
        conditional coverage test results. Includes a plain-language
        ``model_adequate`` verdict (True if no test rejects the null).
    """
    violations = compute_violations(returns, var_forecasts)
    conditional = christoffersen_conditional_coverage_test(violations, confidence_level)

    model_adequate = not (
        conditional["unconditional_coverage"]["reject_null"]
        or conditional["independence"]["reject_null"]
    )

    return {
        "num_observations": len(violations),
        "num_violations": int(violations.sum()),
        "violation_rate": float(violations.mean()) if len(violations) else 0.0,
        "expected_rate": 1 - confidence_level,
        "confidence_level": confidence_level,
        "kupiec_pof": conditional["unconditional_coverage"],
        "christoffersen_independence": conditional["independence"],
        "conditional_coverage": conditional,
        "model_adequate": model_adequate,
    }
