"""Covariance matrix shrinkage estimators for robust portfolio optimization.

When asset returns have limited history or high dimensionality,
sample covariance matrices are noisy and singular. Shrinkage estimators
blend the sample covariance toward a structured target to produce
more stable, invertible estimates.

Implements:
    - Ledoit-Wolf linear shrinkage (constant-correlation target)
    - Oracle Approximating Shrinkage (OAS)
    - Identity shrinkage (diagonal target)
    - Custom target blending

Example:
    >>> import numpy as np
    >>> returns = np.random.randn(252, 10) * 0.02
    >>> cov = ledoit_wolf_shrinkage(returns)
    >>> print(f"Condition number: {np.linalg.cond(cov):.1f}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ShrinkageResult:
    """Result of covariance shrinkage estimation.

    Attributes:
        covariance: The shrunk covariance matrix.
        shrinkage_intensity: Optimal shrinkage intensity in [0, 1].
        target: The shrinkage target matrix used.
        sample_covariance: The original sample covariance.
        condition_number_before: Condition number of sample covariance.
        condition_number_after: Condition number of shrunk covariance.
    """

    covariance: np.ndarray
    shrinkage_intensity: float
    target: np.ndarray
    sample_covariance: np.ndarray
    condition_number_before: float
    condition_number_after: float


def _sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Compute de-meaned sample covariance matrix.

    Args:
        returns: T x N array of asset returns.

    Returns:
        N x N sample covariance matrix.
    """
    t, _n = returns.shape
    demeaned = returns - returns.mean(axis=0)
    return (demeaned.T @ demeaned) / t


def _constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """Build the constant-correlation shrinkage target.

    The target has the same variances as the sample but replaces
    all off-diagonal correlations with the average sample correlation.

    Args:
        sample_cov: N x N sample covariance matrix.

    Returns:
        N x N constant-correlation target matrix.
    """
    n = sample_cov.shape[0]
    std = np.sqrt(np.diag(sample_cov))

    # Avoid division by zero
    std_safe = np.where(std == 0, 1e-10, std)
    corr = sample_cov / np.outer(std_safe, std_safe)

    # Average off-diagonal correlation
    mask = ~np.eye(n, dtype=bool)
    avg_corr = corr[mask].mean() if n > 1 else 0.0

    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, np.diag(sample_cov))
    return target


def ledoit_wolf_shrinkage(
    returns: np.ndarray,
    assume_centered: bool = False,
) -> ShrinkageResult:
    """Ledoit-Wolf linear shrinkage with constant-correlation target.

    Implements the analytical formula from Ledoit & Wolf (2004)
    "A well-conditioned estimator for large-dimensional covariance matrices."

    Args:
        returns: T x N array of asset returns (rows = observations).
        assume_centered: If True, skip de-meaning.

    Returns:
        ShrinkageResult with optimally shrunk covariance.

    Raises:
        ValueError: If returns has fewer than 2 observations or assets.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.02, (100, 5))
        >>> result = ledoit_wolf_shrinkage(returns)
        >>> assert 0 <= result.shrinkage_intensity <= 1
        >>> assert result.covariance.shape == (5, 5)
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError("returns must be a T x N array with T >= 2 and N >= 1")

    t, n = returns.shape

    if not assume_centered:
        returns = returns - returns.mean(axis=0)

    sample_cov = (returns.T @ returns) / t
    target = _constant_correlation_target(sample_cov)

    # Frobenius norm terms for optimal shrinkage
    # delta = ||sample - target||^2_F / p  (scaled)
    delta = sample_cov - target

    # Estimate optimal shrinkage using Ledoit-Wolf formula
    # Sum of squared element-wise sample covariance estimation errors
    x2 = returns ** 2
    sample2 = (x2.T @ x2) / t  # E[r_i^2 * r_j^2]
    pi_hat = np.sum(sample2 - sample_cov ** 2)  # sum of asymptotic variances

    # Rho: cross-term for target
    rho_hat = 0.0
    std = np.sqrt(np.diag(sample_cov))
    std_safe = np.where(std == 0, 1e-10, std)

    for i in range(n):
        for j in range(n):
            if i == j:
                rho_hat += sample2[i, i] - sample_cov[i, i] ** 2
            else:
                # Scaling factors for constant-correlation target
                theta_ij = (
                    np.sum(returns[:, i] * returns[:, j] * returns[:, j] ** 2) / t
                    - sample_cov[i, j] * sample_cov[j, j]
                )
                theta_ji = (
                    np.sum(returns[:, j] * returns[:, i] * returns[:, i] ** 2) / t
                    - sample_cov[i, j] * sample_cov[i, i]
                )
                rho_hat += (
                    target[i, j]
                    / (std_safe[i] * std_safe[j])
                    * (theta_ij * std_safe[j] / std_safe[i]
                       + theta_ji * std_safe[i] / std_safe[j])
                    / 2.0
                )

    gamma_hat = np.sum(delta ** 2)

    # Kappa = (pi - rho) / gamma
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0

    # Optimal shrinkage intensity, clamped to [0, 1]
    shrinkage = max(0.0, min(1.0, kappa / t))

    # Shrunk covariance
    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

    cond_before = float(np.linalg.cond(sample_cov))
    cond_after = float(np.linalg.cond(shrunk_cov))

    return ShrinkageResult(
        covariance=shrunk_cov,
        shrinkage_intensity=shrinkage,
        target=target,
        sample_covariance=sample_cov,
        condition_number_before=cond_before,
        condition_number_after=cond_after,
    )


def oas_shrinkage(
    returns: np.ndarray,
    assume_centered: bool = False,
) -> ShrinkageResult:
    """Oracle Approximating Shrinkage (OAS) estimator.

    Shrinks toward a scaled identity matrix using the Chen, Wiesel,
    Eldar & Hero (2010) formula. Simpler and faster than Ledoit-Wolf;
    works well when no strong prior on correlation structure exists.

    Args:
        returns: T x N array of asset returns.
        assume_centered: If True, skip de-meaning.

    Returns:
        ShrinkageResult with OAS-shrunk covariance.

    Raises:
        ValueError: If returns has fewer than 2 observations.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.02, (100, 5))
        >>> result = oas_shrinkage(returns)
        >>> assert result.condition_number_after <= result.condition_number_before
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError("returns must be a T x N array with T >= 2 and N >= 1")

    t, n = returns.shape

    if not assume_centered:
        returns = returns - returns.mean(axis=0)

    sample_cov = (returns.T @ returns) / t

    # Target: scaled identity  mu * I  where mu = tr(S) / n
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    # OAS formula
    delta = sample_cov - target
    trace_s2 = np.sum(sample_cov ** 2)  # tr(S^2)
    trace_s_sq = np.trace(sample_cov) ** 2  # tr(S)^2

    # Numerator: (1 - 2/n) * tr(S^2) + tr(S)^2
    numerator = (1.0 - 2.0 / n) * trace_s2 + trace_s_sq
    # Denominator: (t + 1 - 2/n) * (tr(S^2) - tr(S)^2 / n)
    denominator = (t + 1.0 - 2.0 / n) * (trace_s2 - trace_s_sq / n)

    if denominator > 0:
        shrinkage = max(0.0, min(1.0, numerator / denominator))
    else:
        shrinkage = 1.0

    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

    cond_before = float(np.linalg.cond(sample_cov))
    cond_after = float(np.linalg.cond(shrunk_cov))

    return ShrinkageResult(
        covariance=shrunk_cov,
        shrinkage_intensity=shrinkage,
        target=target,
        sample_covariance=sample_cov,
        condition_number_before=cond_before,
        condition_number_after=cond_after,
    )


def identity_shrinkage(
    returns: np.ndarray,
    shrinkage_intensity: Optional[float] = None,
    assume_centered: bool = False,
) -> ShrinkageResult:
    """Shrink sample covariance toward the identity matrix.

    A simple shrinkage that regularizes the covariance by blending
    with the identity scaled to match the average variance.

    Args:
        returns: T x N array of asset returns.
        shrinkage_intensity: Fixed intensity in [0, 1]. If None,
            uses the Ledoit-Wolf optimal formula for identity target.
        assume_centered: If True, skip de-meaning.

    Returns:
        ShrinkageResult with identity-shrunk covariance.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.02, (100, 5))
        >>> result = identity_shrinkage(returns, shrinkage_intensity=0.3)
        >>> assert result.shrinkage_intensity == 0.3
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError("returns must be a T x N array with T >= 2 and N >= 1")

    t, n = returns.shape

    if not assume_centered:
        returns = returns - returns.mean(axis=0)

    sample_cov = (returns.T @ returns) / t
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    if shrinkage_intensity is not None:
        alpha = max(0.0, min(1.0, shrinkage_intensity))
    else:
        # Analytical optimal for identity target
        delta = sample_cov - target
        x2 = returns ** 2
        sample2 = (x2.T @ x2) / t
        pi_hat = np.sum(sample2 - sample_cov ** 2)
        gamma_hat = np.sum(delta ** 2)
        alpha = max(0.0, min(1.0, pi_hat / (t * gamma_hat))) if gamma_hat > 0 else 0.0

    shrunk_cov = alpha * target + (1 - alpha) * sample_cov

    cond_before = float(np.linalg.cond(sample_cov))
    cond_after = float(np.linalg.cond(shrunk_cov))

    return ShrinkageResult(
        covariance=shrunk_cov,
        shrinkage_intensity=alpha,
        target=target,
        sample_covariance=sample_cov,
        condition_number_before=cond_before,
        condition_number_after=cond_after,
    )


def custom_target_shrinkage(
    returns: np.ndarray,
    target: np.ndarray,
    shrinkage_intensity: float = 0.5,
    assume_centered: bool = False,
) -> ShrinkageResult:
    """Shrink sample covariance toward a user-specified target.

    Args:
        returns: T x N array of asset returns.
        target: N x N positive-semidefinite target matrix.
        shrinkage_intensity: Blending weight in [0, 1].
        assume_centered: If True, skip de-meaning.

    Returns:
        ShrinkageResult with custom-shrunk covariance.

    Raises:
        ValueError: If target dimensions don't match returns.
    """
    returns = np.asarray(returns, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if returns.ndim != 2 or returns.shape[0] < 2:
        raise ValueError("returns must be a T x N array with T >= 2")
    if target.shape != (returns.shape[1], returns.shape[1]):
        raise ValueError(
            f"target shape {target.shape} doesn't match "
            f"expected ({returns.shape[1]}, {returns.shape[1]})"
        )

    if not assume_centered:
        returns = returns - returns.mean(axis=0)

    sample_cov = (returns.T @ returns) / returns.shape[0]
    alpha = max(0.0, min(1.0, shrinkage_intensity))

    shrunk_cov = alpha * target + (1 - alpha) * sample_cov

    cond_before = float(np.linalg.cond(sample_cov))
    cond_after = float(np.linalg.cond(shrunk_cov))

    return ShrinkageResult(
        covariance=shrunk_cov,
        shrinkage_intensity=alpha,
        target=target,
        sample_covariance=sample_cov,
        condition_number_before=cond_before,
        condition_number_after=cond_after,
    )


def compare_shrinkage_methods(
    returns: np.ndarray,
) -> dict:
    """Compare all shrinkage methods on the same data.

    Returns a dict keyed by method name with ShrinkageResult values,
    useful for selecting the best estimator for a given dataset.

    Args:
        returns: T x N array of asset returns.

    Returns:
        Dict mapping method name to ShrinkageResult.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.02, (100, 5))
        >>> results = compare_shrinkage_methods(returns)
        >>> for name, res in results.items():
        ...     print(f"{name}: intensity={res.shrinkage_intensity:.3f}")
    """
    return {
        "ledoit_wolf": ledoit_wolf_shrinkage(returns),
        "oas": oas_shrinkage(returns),
        "identity": identity_shrinkage(returns),
    }
