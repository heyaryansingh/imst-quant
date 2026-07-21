"""Drawdown-at-Risk (DaR) and Conditional Drawdown-at-Risk (CDaR).

CDaR (Chekhlov, Uryasev & Zabarankin, 2005) applies the VaR/CVaR idea to
the drawdown distribution instead of the return distribution: DaR(alpha)
is the alpha-quantile of per-period drawdowns, and CDaR(alpha) is the mean
drawdown in the worst (1 - alpha) tail. Unlike max drawdown, CDaR uses the
whole underwater history, so it is far less noisy for a single path while
still penalizing deep, persistent losses.

Drawdowns here are reported as positive fractions (0.10 = 10% below peak).
"""

import numpy as np
from typing import Any, Dict, Sequence

Vector = Sequence[float]


def _validate_returns(returns: Vector, min_obs: int = 10) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if arr.size < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contains non-finite values")
    if np.any(arr <= -1):
        raise ValueError("returns contains values <= -100%")
    return arr


def drawdown_series(returns: Vector) -> np.ndarray:
    """Per-period drawdown from the running peak of the compounded curve.

    Args:
        returns: 1D simple-return series.

    Returns:
        Array of drawdowns as positive fractions (0 when at a new peak).

    Raises:
        ValueError: If input is malformed.
    """
    arr = _validate_returns(returns, min_obs=2)
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    return 1.0 - equity / peak


def drawdown_at_risk(returns: Vector, alpha: float = 0.95) -> float:
    """DaR(alpha): the alpha-quantile of the drawdown distribution.

    Args:
        returns: 1D simple-return series.
        alpha: Confidence level in (0, 1); 0.95 means the drawdown exceeded
            only 5% of the time.

    Returns:
        Drawdown-at-Risk as a positive fraction.

    Raises:
        ValueError: If input is malformed or alpha is out of range.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    arr = _validate_returns(returns)
    dd = drawdown_series(arr)
    return float(np.quantile(dd, alpha))


def conditional_drawdown_at_risk(returns: Vector, alpha: float = 0.95) -> float:
    """CDaR(alpha): mean drawdown in the worst (1 - alpha) tail.

    Args:
        returns: 1D simple-return series.
        alpha: Confidence level in (0, 1).

    Returns:
        Conditional Drawdown-at-Risk as a positive fraction. Always >= DaR.

    Raises:
        ValueError: If input is malformed or alpha is out of range.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    arr = _validate_returns(returns)
    dd = drawdown_series(arr)
    dar = np.quantile(dd, alpha)
    tail = dd[dd >= dar]
    return float(tail.mean()) if tail.size else float(dar)


def cdar_ratio(
    returns: Vector,
    alpha: float = 0.95,
    periods_per_year: int = 252,
) -> float:
    """Annualized return divided by CDaR(alpha) — a drawdown-aware Calmar.

    Args:
        returns: 1D simple-return series.
        alpha: Confidence level for CDaR.
        periods_per_year: Compounding periods per year (252 for daily).

    Returns:
        Annualized geometric return / CDaR. Positive infinity when CDaR is
        zero and returns are positive; 0.0 when both are zero.

    Raises:
        ValueError: If input is malformed or parameters are out of range.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    arr = _validate_returns(returns)
    total_growth = float(np.prod(1.0 + arr))
    ann_return = total_growth ** (periods_per_year / arr.size) - 1.0
    cdar = conditional_drawdown_at_risk(arr, alpha=alpha)
    if cdar == 0.0:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / cdar)


def analyze_drawdown_risk(
    returns: Vector,
    alphas: Sequence[float] = (0.90, 0.95, 0.99),
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """Summarize the drawdown distribution at several confidence levels.

    Args:
        returns: 1D simple-return series.
        alphas: Confidence levels to evaluate.
        periods_per_year: Compounding periods per year.

    Returns:
        Dict with max_drawdown, average_drawdown, time_underwater_pct
        (fraction of periods below the prior peak), current_drawdown,
        levels (list of {alpha, dar, cdar}), cdar_ratio (at the middle
        alpha), and n_observations.

    Raises:
        ValueError: If input is malformed or alphas are out of range.
    """
    if not alphas:
        raise ValueError("alphas must be non-empty")
    arr = _validate_returns(returns)
    dd = drawdown_series(arr)
    levels = [
        {
            "alpha": float(a),
            "dar": drawdown_at_risk(returns, alpha=a),
            "cdar": conditional_drawdown_at_risk(returns, alpha=a),
        }
        for a in alphas
    ]
    mid_alpha = sorted(alphas)[len(alphas) // 2]
    return {
        "max_drawdown": float(dd.max()),
        "average_drawdown": float(dd.mean()),
        "time_underwater_pct": float((dd > 0).mean()),
        "current_drawdown": float(dd[-1]),
        "levels": levels,
        "cdar_ratio": cdar_ratio(returns, alpha=mid_alpha, periods_per_year=periods_per_year),
        "n_observations": int(dd.size),
    }
