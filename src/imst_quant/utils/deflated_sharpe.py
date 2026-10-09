"""Deflated Sharpe Ratio (DSR) for multiple-testing-aware strategy selection.

A strategy's observed Sharpe ratio tends to look better than its true skill
when many strategy variants were tried and the best one was reported (the
"backtest overfitting" problem). The Deflated Sharpe Ratio, introduced by
Bailey & Lopez de Prado (2014), corrects for this by comparing the observed
Sharpe ratio against the *expected maximum* Sharpe ratio one would see by
chance after ``n_trials`` independent backtests, then expresses the result
as a probability (via the Probabilistic Sharpe Ratio) that the true Sharpe
ratio exceeds that benchmark.
"""

import numpy as np
from scipy import stats
from typing import Any, Dict, Optional, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]

_EULER_MASCHERONI = 0.5772156649015329


def estimated_sharpe_ratio_stderr(
    n_observations: int,
    sharpe_ratio: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of an estimated Sharpe ratio.

    Uses the Mertens (2002) approximation, which accounts for
    non-normal returns via skewness and (non-excess) kurtosis.

    Args:
        n_observations: Number of return observations used to estimate the
            Sharpe ratio.
        sharpe_ratio: The (non-annualized) observed Sharpe ratio.
        skewness: Sample skewness of returns (0 for a normal distribution).
        kurtosis: Sample kurtosis of returns, non-excess convention
            (3 for a normal distribution).

    Returns:
        Standard error of the Sharpe ratio estimate.

    Raises:
        ValueError: If n_observations is less than 2.
    """
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")

    variance = (
        1
        - skewness * sharpe_ratio
        + ((kurtosis - 1) / 4) * sharpe_ratio**2
    ) / (n_observations - 1)
    return float(np.sqrt(max(variance, 0.0)))


def probabilistic_sharpe_ratio(
    sharpe_ratio: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds a benchmark.

    Args:
        sharpe_ratio: Observed (non-annualized) Sharpe ratio.
        benchmark_sharpe: Sharpe ratio to test against, e.g. 0 or the
            expected maximum Sharpe ratio from multiple trials.
        n_observations: Number of return observations.
        skewness: Sample skewness of returns.
        kurtosis: Sample kurtosis of returns, non-excess convention.

    Returns:
        PSR in [0, 1]. Values close to 1 indicate high confidence that the
        strategy's true skill exceeds the benchmark.
    """
    stderr = estimated_sharpe_ratio_stderr(
        n_observations, sharpe_ratio, skewness, kurtosis
    )
    if stderr == 0:
        return 1.0 if sharpe_ratio > benchmark_sharpe else 0.0

    z = (sharpe_ratio - benchmark_sharpe) / stderr
    return float(stats.norm.cdf(z))


def min_track_record_length(
    sharpe_ratio: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado, 2012).

    Number of observations needed before PSR(benchmark) reaches
    ``confidence`` given the observed per-period Sharpe ratio and return
    shape. Inverts the Mertens standard error used by
    probabilistic_sharpe_ratio, so the two agree exactly.

    Args:
        sharpe_ratio: Observed (non-annualized) Sharpe ratio.
        benchmark_sharpe: Sharpe ratio to beat, same units.
        skewness: Sample skewness of returns.
        kurtosis: Sample kurtosis of returns, non-excess convention.
        confidence: Target PSR, in (0, 1).

    Returns:
        Required number of observations, or ``inf`` when the observed
        Sharpe ratio does not exceed the benchmark.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    edge = sharpe_ratio - benchmark_sharpe
    if edge <= 0:
        return float("inf")
    shape = 1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2
    z = stats.norm.ppf(confidence)
    return float(1 + max(shape, 0.0) * (z / edge) ** 2)


def expected_max_sharpe_ratio(
    n_trials: int,
    sharpe_variance: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio observed across independent trials.

    Approximates E[max(SR_1, ..., SR_n)] assuming the trial Sharpe ratios
    are independent draws from a normal distribution with the given
    variance, using the extreme value approximation from Bailey & Lopez de
    Prado (2014).

    Args:
        n_trials: Number of independent strategy trials/backtests.
        sharpe_variance: Variance of the Sharpe ratios across trials.
            Defaults to 1.0 (unit variance) when unknown.

    Returns:
        Expected maximum Sharpe ratio under the null hypothesis of no
        skill (zero true Sharpe ratio across all trials).

    Raises:
        ValueError: If n_trials is less than 1.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0

    sharpe_std = np.sqrt(max(sharpe_variance, 0.0))
    return float(
        sharpe_std
        * (
            (1 - _EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n_trials)
            + _EULER_MASCHERONI * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_observations: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_variance: float = 1.0,
) -> Dict[str, Any]:
    """Compute the Deflated Sharpe Ratio for a strategy.

    Args:
        sharpe_ratio: Observed (non-annualized) Sharpe ratio of the
            selected strategy.
        n_observations: Number of return observations used to estimate the
            Sharpe ratio.
        n_trials: Number of independent strategy variants tried before this
            one was selected (use 1 if this was the only strategy tested).
        skewness: Sample skewness of returns.
        kurtosis: Sample kurtosis of returns, non-excess convention.
        sharpe_variance: Variance of Sharpe ratios across trials, used to
            estimate the expected maximum Sharpe ratio under multiple
            testing. Defaults to 1.0 when unknown.

    Returns:
        Dictionary containing:
            - deflated_sharpe_ratio: Probability the true Sharpe ratio
              exceeds the expected-by-chance maximum across n_trials
              (in [0, 1]; higher is better, >0.95 is a common threshold).
            - expected_max_sharpe: Benchmark Sharpe ratio expected by
              chance given n_trials.
            - sharpe_stderr: Standard error of the Sharpe ratio estimate.
            - n_trials: Number of trials used in the calculation.
            - n_observations: Number of observations used.
    """
    expected_max = expected_max_sharpe_ratio(n_trials, sharpe_variance)
    dsr = probabilistic_sharpe_ratio(
        sharpe_ratio, expected_max, n_observations, skewness, kurtosis
    )
    stderr = estimated_sharpe_ratio_stderr(
        n_observations, sharpe_ratio, skewness, kurtosis
    )

    return {
        "deflated_sharpe_ratio": dsr,
        "expected_max_sharpe": expected_max,
        "sharpe_stderr": stderr,
        "n_trials": n_trials,
        "n_observations": n_observations,
    }


def deflated_sharpe_ratio_from_returns(
    returns: ArrayLike,
    n_trials: int,
    periods_per_year: int = 1,
    sharpe_variance: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience wrapper computing DSR directly from a return series.

    The PSR/DSR probability is always computed on the per-period Sharpe
    ratio: the Mertens standard error is only valid in the units the
    returns were sampled in, so feeding it an annualized Sharpe ratio
    overstates confidence by roughly sqrt(periods_per_year).

    Args:
        returns: Array of periodic returns. NaN/inf values are dropped.
        n_trials: Number of independent strategy variants tried.
        periods_per_year: If greater than 1, the reported ``sharpe_ratio``,
            ``expected_max_sharpe`` and ``sharpe_stderr`` are annualized.
            The DSR probability itself is unit-free and unaffected.
        sharpe_variance: Variance of the per-period Sharpe ratios across
            trials. Defaults to the sampling variance of a zero-skill
            Sharpe estimate, 1 / (n - 1).

    Returns:
        Same dictionary as deflated_sharpe_ratio, plus:
            - sharpe_ratio: The (possibly annualized) Sharpe ratio used.
            - min_track_record_length: Observations needed for the DSR to
              reach 0.95 at the observed Sharpe (inf if it never will).

    Raises:
        ValueError: If fewer than 2 finite return observations are provided.
    """
    returns_arr = np.asarray(returns, dtype=float)
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size < 2:
        raise ValueError("returns must contain at least 2 observations")

    mean = returns_arr.mean()
    std = returns_arr.std(ddof=1)
    sharpe = 0.0 if std == 0 else float(mean / std)
    skewness = float(stats.skew(returns_arr))
    kurtosis = float(stats.kurtosis(returns_arr, fisher=False))

    if sharpe_variance is None:
        sharpe_variance = 1.0 / (returns_arr.size - 1)

    result = deflated_sharpe_ratio(
        sharpe_ratio=sharpe,
        n_observations=returns_arr.size,
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
        sharpe_variance=sharpe_variance,
    )

    result["min_track_record_length"] = min_track_record_length(
        sharpe, result["expected_max_sharpe"], skewness, kurtosis
    )

    annualization = float(np.sqrt(periods_per_year))
    result["sharpe_ratio"] = sharpe * annualization
    result["expected_max_sharpe"] *= annualization
    result["sharpe_stderr"] *= annualization
    return result
