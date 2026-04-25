"""Signal validation and quality assessment utilities.

Tools for evaluating trading signal quality before deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger()


def validate_signal_quality(
    signal: pd.Series,
    returns: pd.Series,
    min_sharpe: float = 1.0,
    min_hit_rate: float = 0.52,
    max_turnover: float = 2.0,
) -> Dict[str, any]:
    """Validate that a signal meets minimum quality thresholds.

    Args:
        signal: Series of trading signals (-1, 0, 1)
        returns: Series of forward returns
        min_sharpe: Minimum acceptable Sharpe ratio
        min_hit_rate: Minimum hit rate (% of profitable signals)
        max_turnover: Maximum acceptable annual turnover

    Returns:
        Dict with validation results and pass/fail status
    """
    # Align signal and returns
    aligned = pd.DataFrame({
        "signal": signal,
        "returns": returns,
    }).dropna()

    # Calculate strategy returns
    strategy_returns = aligned["signal"].shift(1) * aligned["returns"]
    strategy_returns = strategy_returns.dropna()

    # Sharpe ratio
    sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0

    # Hit rate (only for non-zero signals)
    non_zero_signals = aligned[aligned["signal"].shift(1) != 0]
    signal_returns = non_zero_signals["signal"].shift(1) * non_zero_signals["returns"]
    hit_rate = (signal_returns > 0).sum() / len(signal_returns) if len(signal_returns) > 0 else 0

    # Turnover (number of signal changes per year)
    signal_changes = (aligned["signal"] != aligned["signal"].shift(1)).sum()
    turnover = signal_changes / len(aligned) * 252

    # Validation checks
    validation = {
        "passed": True,
        "sharpe": float(sharpe),
        "hit_rate": float(hit_rate),
        "turnover": float(turnover),
        "checks": {
            "sharpe_check": sharpe >= min_sharpe,
            "hit_rate_check": hit_rate >= min_hit_rate,
            "turnover_check": turnover <= max_turnover,
        },
        "thresholds": {
            "min_sharpe": min_sharpe,
            "min_hit_rate": min_hit_rate,
            "max_turnover": max_turnover,
        },
    }

    # Overall pass/fail
    validation["passed"] = all(validation["checks"].values())

    return validation


def check_signal_consistency(
    signal: pd.Series,
    window: int = 20,
) -> Dict[str, float]:
    """Check signal consistency and stability over time.

    Args:
        signal: Series of trading signals
        window: Rolling window for consistency checks

    Returns:
        Dict with consistency metrics
    """
    # Signal flip rate (how often does it change?)
    flips = (signal != signal.shift(1)).astype(int)
    flip_rate = flips.rolling(window).mean()

    # Signal autocorrelation (persistence)
    autocorr_1 = signal.autocorr(lag=1)
    autocorr_5 = signal.autocorr(lag=5)

    # Percentage of time in each state
    pct_long = (signal == 1).sum() / len(signal)
    pct_short = (signal == -1).sum() / len(signal)
    pct_flat = (signal == 0).sum() / len(signal)

    return {
        "avg_flip_rate": float(flip_rate.mean()),
        "max_flip_rate": float(flip_rate.max()),
        "autocorr_lag1": float(autocorr_1),
        "autocorr_lag5": float(autocorr_5),
        "pct_long": float(pct_long),
        "pct_short": float(pct_short),
        "pct_flat": float(pct_flat),
    }


def detect_signal_lookahead(
    signal: pd.Series,
    future_returns: pd.Series,
    lags: List[int] = [1, 2, 3, 5, 10],
) -> Dict[str, float]:
    """Detect potential lookahead bias by checking correlation with future returns.

    Args:
        signal: Series of trading signals
        future_returns: Series of future returns
        lags: List of forward lags to check

    Returns:
        Dict with correlation at each lag (should be near zero for lag=0)
    """
    correlations = {}

    for lag in lags:
        shifted_returns = future_returns.shift(-lag)
        corr = signal.corr(shifted_returns)
        correlations[f"lag_{lag}"] = float(corr)

    # Lookahead warning if correlation at lag 0 is suspiciously high
    lookahead_warning = abs(correlations.get("lag_1", 0)) > 0.3

    return {
        "correlations": correlations,
        "lookahead_warning": lookahead_warning,
    }


def compare_signals(
    signal_a: pd.Series,
    signal_b: pd.Series,
    returns: pd.Series,
    name_a: str = "Signal A",
    name_b: str = "Signal B",
) -> Dict[str, any]:
    """Compare two signals head-to-head.

    Args:
        signal_a: First trading signal
        signal_b: Second trading signal
        returns: Forward returns for evaluation
        name_a: Name for first signal
        name_b: Name for second signal

    Returns:
        Dict with comparison metrics
    """
    # Align all series
    df = pd.DataFrame({
        "signal_a": signal_a,
        "signal_b": signal_b,
        "returns": returns,
    }).dropna()

    # Calculate returns for each signal
    returns_a = df["signal_a"].shift(1) * df["returns"]
    returns_b = df["signal_b"].shift(1) * df["returns"]

    # Sharpe ratios
    sharpe_a = (returns_a.mean() / returns_a.std() * np.sqrt(252)) if returns_a.std() > 0 else 0
    sharpe_b = (returns_b.mean() / returns_b.std() * np.sqrt(252)) if returns_b.std() > 0 else 0

    # Total returns
    total_a = (1 + returns_a).prod() - 1
    total_b = (1 + returns_b).prod() - 1

    # Agreement rate
    agreement = (df["signal_a"] == df["signal_b"]).sum() / len(df)

    # Correlation
    correlation = df["signal_a"].corr(df["signal_b"])

    return {
        name_a: {
            "sharpe": float(sharpe_a),
            "total_return": float(total_a),
            "volatility": float(returns_a.std() * np.sqrt(252)),
        },
        name_b: {
            "sharpe": float(sharpe_b),
            "total_return": float(total_b),
            "volatility": float(returns_b.std() * np.sqrt(252)),
        },
        "agreement_rate": float(agreement),
        "correlation": float(correlation),
        "winner": name_a if sharpe_a > sharpe_b else name_b,
    }


def signal_robustness_test(
    signal: pd.Series,
    returns: pd.Series,
    noise_levels: List[float] = [0.1, 0.2, 0.3],
    n_simulations: int = 100,
) -> Dict[str, any]:
    """Test signal robustness by adding noise to returns.

    Args:
        signal: Trading signal
        returns: Forward returns
        noise_levels: List of noise levels to test (as fraction of std dev)
        n_simulations: Number of Monte Carlo simulations per noise level

    Returns:
        Dict with robustness metrics at each noise level
    """
    # Align signal and returns
    aligned = pd.DataFrame({
        "signal": signal,
        "returns": returns,
    }).dropna()

    base_returns = aligned["signal"].shift(1) * aligned["returns"]
    base_sharpe = (base_returns.mean() / base_returns.std() * np.sqrt(252)) if base_returns.std() > 0 else 0

    results = {
        "base_sharpe": float(base_sharpe),
        "noise_tests": {},
    }

    returns_std = aligned["returns"].std()

    for noise_level in noise_levels:
        sharpes = []

        for _ in range(n_simulations):
            # Add noise to returns
            noise = np.random.normal(0, noise_level * returns_std, len(aligned))
            noisy_returns = aligned["returns"] + noise

            # Calculate Sharpe with noisy returns
            strategy_returns = aligned["signal"].shift(1) * noisy_returns
            sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
            sharpes.append(sharpe)

        results["noise_tests"][f"noise_{noise_level}"] = {
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "degradation": float(base_sharpe - np.mean(sharpes)),
        }

    return results


def generate_signal_report(
    signal: pd.Series,
    returns: pd.Series,
    signal_name: str = "Signal",
) -> str:
    """Generate a comprehensive signal validation report.

    Args:
        signal: Trading signal
        returns: Forward returns
        signal_name: Name of the signal

    Returns:
        Formatted text report
    """
    # Run all validation checks
    quality = validate_signal_quality(signal, returns)
    consistency = check_signal_consistency(signal)
    lookahead = detect_signal_lookahead(signal, returns)

    # Build report
    report = []
    report.append(f"=== {signal_name} Validation Report ===\n")

    # Quality metrics
    report.append("--- Quality Metrics ---")
    report.append(f"Sharpe Ratio: {quality['sharpe']:.2f} {'✓' if quality['checks']['sharpe_check'] else '✗'}")
    report.append(f"Hit Rate: {quality['hit_rate']:.2%} {'✓' if quality['checks']['hit_rate_check'] else '✗'}")
    report.append(f"Turnover: {quality['turnover']:.2f}x {'✓' if quality['checks']['turnover_check'] else '✗'}")
    report.append(f"\nOverall: {'PASSED' if quality['passed'] else 'FAILED'}\n")

    # Consistency
    report.append("--- Consistency Metrics ---")
    report.append(f"Avg Flip Rate: {consistency['avg_flip_rate']:.2%}")
    report.append(f"Autocorrelation (lag 1): {consistency['autocorr_lag1']:.3f}")
    report.append(f"Long/Short/Flat: {consistency['pct_long']:.1%} / {consistency['pct_short']:.1%} / {consistency['pct_flat']:.1%}\n")

    # Lookahead bias check
    report.append("--- Lookahead Bias Check ---")
    if lookahead['lookahead_warning']:
        report.append("⚠ WARNING: Potential lookahead bias detected!")
    else:
        report.append("✓ No obvious lookahead bias")

    for lag, corr in lookahead['correlations'].items():
        report.append(f"  {lag}: {corr:.3f}")

    return "\n".join(report)
