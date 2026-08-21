"""Regression tests for the Ledoit-Wolf constant-correlation shrinkage intensity.

The existing suite only checks shape/symmetry/bounds, which cannot catch a wrong
shrinkage intensity. These tests pin ``ledoit_wolf_shrinkage`` against an
independent, vectorised transcription of the reference ``covCor`` estimator from
Ledoit & Wolf (2004), where

    rho = sum(diag(pi)) + rBar * sum_{i != j} sqrt(s_jj / s_ii) * theta_ii_ij

The scaling factors on ``theta_ii_ij`` and ``theta_jj_ij`` used to be swapped,
which only shows up when the assets have different variances.
"""

import numpy as np
import pytest

from imst_quant.utils.covariance_shrinkage import ledoit_wolf_shrinkage


def reference_shrinkage(returns: np.ndarray) -> float:
    """Vectorised transcription of Ledoit & Wolf's reference covCor code."""
    x = np.asarray(returns, dtype=np.float64)
    t, n = x.shape
    x = x - x.mean(axis=0)

    sample = (x.T @ x) / t
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)

    # Constant-correlation target.
    corr = sample / np.outer(sqrtvar, sqrtvar)
    off = ~np.eye(n, dtype=bool)
    r_bar = corr[off].mean()
    target = r_bar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(target, var)

    # pi: elementwise asymptotic variances of the sample covariance entries.
    y = x ** 2
    pi_mat = (y.T @ y) / t - sample ** 2
    pi_hat = pi_mat.sum()

    # rho: diagonal part plus the constant-correlation cross term.
    # theta_mat[i, j] = E[(r_i^2 - s_ii)(r_i r_j - s_ij)]
    theta_mat = ((x ** 3).T @ x) / t - var[:, None] * sample
    np.fill_diagonal(theta_mat, 0.0)
    rho_hat = np.trace(pi_mat) + r_bar * np.sum(
        np.outer(1.0 / sqrtvar, sqrtvar) * theta_mat
    )

    gamma_hat = np.sum((sample - target) ** 2)
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(max(0.0, min(1.0, kappa / t)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_reference_with_heterogeneous_volatilities(seed):
    rng = np.random.default_rng(seed)
    vols = np.array([0.002, 0.01, 0.05, 0.25, 0.9])
    returns = rng.normal(0, 1, (80, 5)) * vols

    result = ledoit_wolf_shrinkage(returns)
    assert result.shrinkage_intensity == pytest.approx(
        reference_shrinkage(returns), rel=1e-10, abs=1e-12
    )


def test_matches_reference_with_correlated_assets():
    rng = np.random.default_rng(7)
    factor = rng.normal(0, 1, (150, 1))
    idio = rng.normal(0, 1, (150, 4))
    loadings = np.array([0.3, 0.8, 1.5, 4.0])
    returns = (factor * loadings + idio) * np.array([0.005, 0.02, 0.06, 0.4])

    result = ledoit_wolf_shrinkage(returns)
    assert result.shrinkage_intensity == pytest.approx(
        reference_shrinkage(returns), rel=1e-10, abs=1e-12
    )


def test_equal_variance_case_still_matches():
    """The swapped scaling factors cancel when all variances are equal."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0, 0.02, (120, 6))

    result = ledoit_wolf_shrinkage(returns)
    assert result.shrinkage_intensity == pytest.approx(
        reference_shrinkage(returns), rel=1e-8, abs=1e-10
    )


def test_shrinkage_shrinks_toward_target():
    rng = np.random.default_rng(3)
    vols = np.array([0.001, 0.05, 0.5])
    returns = rng.normal(0, 1, (30, 3)) * vols

    result = ledoit_wolf_shrinkage(returns)
    blended = (
        result.shrinkage_intensity * result.target
        + (1 - result.shrinkage_intensity) * result.sample_covariance
    )
    np.testing.assert_allclose(result.covariance, blended, atol=1e-12)
    # Diagonal is identical in sample and target, so it must survive shrinkage.
    np.testing.assert_allclose(
        np.diag(result.covariance), np.diag(result.sample_covariance), atol=1e-15
    )
