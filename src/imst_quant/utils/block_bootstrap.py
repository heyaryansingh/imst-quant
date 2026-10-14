"""Stationary block bootstrap confidence intervals for strategy metrics.

Resamples a return series in random-length contiguous blocks (Politis &
Romano, 1994) so volatility clustering and autocorrelation survive the
resampling, then reports percentile confidence intervals for Sharpe,
annualized return and max drawdown.

Example:
    >>> import numpy as np
    >>> r = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    >>> out = bootstrap_metrics(r, n_resamples=200, seed=1)
    >>> out["sharpe"]["lower"] <= out["sharpe"]["point"] <= out["sharpe"]["upper"]
    True
"""

from typing import Dict, Optional

import numpy as np


def stationary_bootstrap_indices(
    n: int, n_resamples: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Return an (n_resamples, n) array of stationary-bootstrap indices.

    Each step continues the current block with probability 1 - 1/mean_block,
    otherwise jumps to a uniformly random start; blocks wrap around the end.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if mean_block < 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")

    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_resamples)
    restart = rng.random((n_resamples, n)) < 1.0 / mean_block
    jumps = rng.integers(0, n, (n_resamples, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe(r: np.ndarray, periods_per_year: int) -> np.ndarray:
    std = r.std(axis=-1, ddof=1)
    mean = r.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0, mean / std * np.sqrt(periods_per_year), 0.0)
    return out


def _annual_return(r: np.ndarray, periods_per_year: int) -> np.ndarray:
    growth = np.prod(1.0 + r, axis=-1)
    years = r.shape[-1] / periods_per_year
    # A path that loses everything has no defined geometric rate; clamp to -100%
    return np.maximum(growth, 0.0) ** (1.0 / years) - 1.0


def _max_drawdown(r: np.ndarray) -> np.ndarray:
    # Start equity at 1.0 so a first-period loss counts as a drawdown
    equity = np.cumprod(1.0 + r, axis=-1)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=-1)
    return (equity / peak - 1.0).min(axis=-1)


def bootstrap_metrics(
    returns,
    n_resamples: int = 1000,
    mean_block: Optional[float] = None,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: Optional[int] = None,
) -> Dict:
    """Bootstrap confidence intervals for Sharpe, annual return and max drawdown.

    Args:
        returns: 1-D array-like of per-period simple returns. NaN/inf are dropped.
        n_resamples: Number of bootstrap paths.
        mean_block: Expected block length; defaults to n ** (1/3).
        confidence: Two-sided interval level in (0, 1).
        periods_per_year: Annualization factor.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with n_observations, mean_block, confidence, prob_positive_sharpe and
        one {point, lower, upper, std} dict per metric.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        raise ValueError(f"Need at least 10 finite returns, got {r.size}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 10:
        raise ValueError(f"n_resamples must be at least 10, got {n_resamples}")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")

    block = float(mean_block) if mean_block is not None else max(1.0, r.size ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    paths = r[stationary_bootstrap_indices(r.size, n_resamples, block, rng)]

    alpha = (1.0 - confidence) / 2.0
    metrics = {
        "sharpe": (_sharpe(r, periods_per_year), _sharpe(paths, periods_per_year)),
        "annual_return": (_annual_return(r, periods_per_year), _annual_return(paths, periods_per_year)),
        "max_drawdown": (_max_drawdown(r), _max_drawdown(paths)),
    }

    result: Dict = {
        "n_observations": int(r.size),
        "n_resamples": int(n_resamples),
        "mean_block": block,
        "confidence": confidence,
    }
    for name, (point, samples) in metrics.items():
        lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
        result[name] = {
            "point": float(point),
            "lower": float(lower),
            "upper": float(upper),
            "std": float(samples.std(ddof=1)),
        }
    result["prob_positive_sharpe"] = float((metrics["sharpe"][1] > 0).mean())
    return result
