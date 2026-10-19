"""Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

Given the return series of N strategy variants tried on the same dates,
CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2015) splits time into S
blocks, and for every way of choosing S/2 blocks as in-sample picks the
variant with the best in-sample Sharpe and records where it ranks
out-of-sample. PBO is the share of splits in which that "winner" lands in
the bottom half out-of-sample.

Example:
    >>> import numpy as np
    >>> noise = np.random.default_rng(0).normal(0, 0.01, (800, 20))
    >>> out = probability_of_backtest_overfitting(noise, n_blocks=8)
    >>> 0.3 < out["pbo"] < 0.7  # pure noise: selection is a coin flip
    True
"""

from itertools import combinations
from typing import Dict

import numpy as np


def probability_of_backtest_overfitting(
    returns: np.ndarray, n_blocks: int = 16
) -> Dict[str, float]:
    """Estimate PBO for a (T periods x N variants) return matrix.

    Args:
        returns: 2D array, one column per strategy variant, rows aligned by date.
            Rows with any non-finite value are dropped.
        n_blocks: Even number of contiguous time blocks S. Splits = C(S, S/2).

    Returns:
        Dict with pbo, logit_median, prob_oos_loss (best-IS variant has
        negative OOS Sharpe), degradation_slope (OLS slope of OOS on IS
        Sharpe of the selected variant), n_splits, n_observations, n_strategies.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("returns must be 2D with at least 2 strategy columns")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError(f"n_blocks must be an even number >= 2, got {n_blocks}")
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    t, n = arr.shape
    if t < 2 * n_blocks:
        raise ValueError(
            f"need at least {2 * n_blocks} complete rows for {n_blocks} blocks, got {t}"
        )

    # Per-block sufficient statistics let each split's Sharpe be assembled
    # without touching the raw rows again.
    blocks = np.array_split(arr, n_blocks)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.array([b.sum(axis=0) for b in blocks])
    s2 = np.array([(b * b).sum(axis=0) for b in blocks])

    def sharpe(mask: np.ndarray) -> np.ndarray:
        k = cnt[mask].sum()
        mean = s1[mask].sum(axis=0) / k
        var = (s2[mask].sum(axis=0) - k * mean * mean) / (k - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, mean / sd, 0.0)

    logits, is_best, oos_best = [], [], []
    # ponytail: enumerates all C(S, S/2) splits (12,870 at S=16); fine up to S~20.
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        mask = np.zeros(n_blocks, dtype=bool)
        mask[list(chosen)] = True
        sr_is, sr_oos = sharpe(mask), sharpe(~mask)
        best = int(np.argmax(sr_is))
        # Relative OOS rank in (0, 1); ties share the average rank.
        rank = (np.sum(sr_oos < sr_oos[best]) + 0.5 * (np.sum(sr_oos == sr_oos[best]) - 1) + 1)
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1 - omega)))
        is_best.append(sr_is[best])
        oos_best.append(sr_oos[best])

    logits = np.array(logits)
    is_best, oos_best = np.array(is_best), np.array(oos_best)
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if np.ptp(is_best) > 0 else 0.0
    return {
        "pbo": float(np.mean(logits <= 0)),
        "logit_median": float(np.median(logits)),
        "prob_oos_loss": float(np.mean(oos_best < 0)),
        "degradation_slope": slope,
        "n_splits": len(logits),
        "n_observations": t,
        "n_strategies": n,
    }
