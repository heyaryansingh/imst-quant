"""Signal quality metrics and validation for trading signals.

This module provides comprehensive signal quality assessment tools including
signal-to-noise ratio, information coefficient, turnover analysis, and
consistency metrics for evaluating trading signal quality.

Typical usage:
    >>> from imst_quant.utils.signal_quality_metrics import (
    ...     calculate_signal_metrics,
    ...     analyze_signal_decay,
    ...     assess_signal_crowding
    ... )
    >>> metrics = calculate_signal_metrics(signals, returns, forward_periods=5)
    >>> decay = analyze_signal_decay(signals, returns, max_horizon=20)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import spearmanr


def calculate_information_coefficient(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    method: str = "spearman",
) -> float:
    """Calculate Information Coefficient between signals and forward returns.

    The IC measures the correlation between predicted signals and actual forward returns,
    indicating signal predictive power. Spearman IC is more robust to outliers.

    Args:
        signals: Array of signal values (predictions).
        forward_returns: Array of actual forward returns.
        method: Correlation method ("pearson" or "spearman").

    Returns:
        Information coefficient value. Typical good values:
        - IC > 0.05: Strong signal
        - 0.02 < IC < 0.05: Moderate signal
        - IC < 0.02: Weak signal

    Example:
        >>> ic = calculate_information_coefficient(signals, forward_returns)
        >>> print(f"Signal IC: {ic:.4f}")
    """
    # Remove NaN values
    mask = ~(np.isnan(signals) | np.isnan(forward_returns))
    signals_clean = signals[mask]
    returns_clean = forward_returns[mask]

    if len(signals_clean) < 2:
        return 0.0

    if method == "spearman":
        ic, _ = spearmanr(signals_clean, returns_clean)
    else:  # pearson
        ic, _ = stats.pearsonr(signals_clean, returns_clean)

    return float(ic) if not np.isnan(ic) else 0.0


def _forward_return_expr(
    return_col: str,
    periods: int,
    group_by: Optional[str] = None,
) -> pl.Expr:
    """Build the forward return over the next ``periods`` bars.

    Shifting a one-period return by ``-periods`` would skip everything in
    between and correlate the signal with a single bar far in the future, so
    the return is accumulated over the whole holding window instead. Row t
    receives the sum of returns t+1 .. t+periods, which uses no information
    available at t.

    Args:
        return_col: Column holding per-period returns.
        periods: Length of the forward window, in periods.
        group_by: Optional column to accumulate within (e.g. "asset_id"),
            so windows never straddle two assets.

    Returns:
        Expression aliased to "forward_return".
    """
    expr = pl.col(return_col).rolling_sum(window_size=periods).shift(-periods)
    if group_by:
        expr = expr.over(group_by)
    return expr.alias("forward_return")


def calculate_signal_metrics(
    df: pl.DataFrame,
    signal_col: str = "signal",
    return_col: str = "return_1d",
    forward_periods: int = 5,
    group_by: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate comprehensive signal quality metrics.

    Args:
        df: DataFrame containing signals and returns.
        signal_col: Name of column containing signal values.
        return_col: Name of column containing returns.
        forward_periods: Length of the forward return window, in periods.
            Returns are accumulated over the whole window, not sampled from
            its last bar.
        group_by: Optional column to group by (e.g., "asset_id").

    Returns:
        Dictionary containing:
        - ic: Information coefficient (Spearman correlation)
        - ic_std: Standard deviation of rolling IC
        - ic_ir: Information ratio (mean rolling IC / rolling IC std)
        - signal_mean: Mean signal value
        - signal_std: Signal standard deviation
        - signal_autocorr: Signal autocorrelation at lag 1
        - hit_rate: Fraction of directional calls that matched the return
          direction; bars where the signal or the return is flat are excluded
        - sharpe: Sharpe ratio of signal-weighted returns
        - turnover: Average signal turnover (change magnitude)

    Example:
        >>> metrics = calculate_signal_metrics(df, signal_col="momentum_score")
        >>> print(f"IC: {metrics['ic']:.4f}, Hit Rate: {metrics['hit_rate']:.2%}")
    """
    if signal_col not in df.columns or return_col not in df.columns:
        raise ValueError(f"Required columns '{signal_col}' and '{return_col}' not found")

    if forward_periods < 1:
        raise ValueError(f"forward_periods must be at least 1, got {forward_periods}")

    # Calculate forward returns
    df = df.sort("date") if "date" in df.columns else df
    df = df.with_columns(_forward_return_expr(return_col, forward_periods, group_by))

    # Drop nulls
    df_clean = df.drop_nulls([signal_col, "forward_return"])

    if df_clean.height == 0:
        return {
            "ic": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "signal_mean": 0.0,
            "signal_std": 0.0,
            "signal_autocorr": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
        }

    signals = df_clean[signal_col].to_numpy()
    forward_returns = df_clean["forward_return"].to_numpy()

    # Information coefficient
    ic = calculate_information_coefficient(signals, forward_returns)

    # Rolling IC for stability
    window = 252  # One year of daily data
    if len(signals) >= window:
        rolling_ics = []
        for i in range(len(signals) - window + 1):
            window_ic = calculate_information_coefficient(
                signals[i : i + window],
                forward_returns[i : i + window],
            )
            rolling_ics.append(window_ic)
        # A single window has no dispersion, and np.std(ddof=1) would return
        # NaN for it rather than a usable number.
        if len(rolling_ics) > 1:
            ic_std = float(np.std(rolling_ics, ddof=1))
            ic_ir = float(np.mean(rolling_ics)) / ic_std if ic_std > 0 else 0.0
        else:
            ic_std = 0.0
            ic_ir = 0.0
    else:
        ic_std = 0.0
        ic_ir = 0.0

    # Signal statistics
    signal_mean = float(np.mean(signals))
    # ddof=1 is NaN for a single observation.
    signal_std = float(np.std(signals, ddof=1)) if len(signals) > 1 else 0.0

    # Signal autocorrelation
    if len(signals) > 1:
        signal_autocorr = float(np.corrcoef(signals[:-1], signals[1:])[0, 1])
        if np.isnan(signal_autocorr):
            signal_autocorr = 0.0
    else:
        signal_autocorr = 0.0

    # Hit rate (directional accuracy). A flat signal or a flat return makes no
    # directional call, and counting those 0 == 0 matches as hits inflated the
    # rate towards 1 for sparse signals.
    signal_direction = np.sign(signals)
    return_direction = np.sign(forward_returns)
    directional = (signal_direction != 0) & (return_direction != 0)
    hit_rate = (
        float(np.mean(signal_direction[directional] == return_direction[directional]))
        if directional.any()
        else 0.0
    )

    # Signal-weighted returns Sharpe
    weighted_returns = signals * forward_returns
    weighted_std = float(np.std(weighted_returns, ddof=1)) if len(weighted_returns) > 1 else 0.0
    sharpe = (
        (float(np.mean(weighted_returns)) / weighted_std) * np.sqrt(252)
        if weighted_std > 0
        else 0.0
    )

    # Turnover
    signal_changes = np.abs(np.diff(signals))
    turnover = float(np.mean(signal_changes)) if len(signal_changes) > 0 else 0.0

    return {
        "ic": float(ic),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "signal_mean": float(signal_mean),
        "signal_std": float(signal_std),
        "signal_autocorr": float(signal_autocorr),
        "hit_rate": float(hit_rate),
        "sharpe": float(sharpe),
        "turnover": float(turnover),
    }


def analyze_signal_decay(
    df: pl.DataFrame,
    signal_col: str = "signal",
    return_col: str = "return_1d",
    max_horizon: int = 20,
    group_by: Optional[str] = None,
) -> Dict[int, float]:
    """Analyze how signal predictive power decays over time horizons.

    Args:
        df: DataFrame containing signals and returns.
        signal_col: Name of column containing signal values.
        return_col: Name of column containing returns.
        max_horizon: Maximum forward horizon to test (in periods). Horizon h
            uses the return accumulated over the next h periods.
        group_by: Optional column to group by (e.g., "asset_id").

    Returns:
        Dictionary mapping forward horizon to the IC against the cumulative
        return over that horizon, showing signal decay.

    Example:
        >>> decay = analyze_signal_decay(df, max_horizon=10)
        >>> print(f"1-day IC: {decay[1]:.4f}, 10-day IC: {decay[10]:.4f}")
    """
    if signal_col not in df.columns or return_col not in df.columns:
        raise ValueError(f"Required columns not found")

    decay_curve = {}

    df = df.sort("date") if "date" in df.columns else df

    for horizon in range(1, max_horizon + 1):
        df_temp = df.with_columns(
            _forward_return_expr(return_col, horizon, group_by)
        )

        df_clean = df_temp.drop_nulls([signal_col, "forward_return"])

        if df_clean.height == 0:
            decay_curve[horizon] = 0.0
            continue

        signals = df_clean[signal_col].to_numpy()
        forward_returns = df_clean["forward_return"].to_numpy()

        ic = calculate_information_coefficient(signals, forward_returns)
        decay_curve[horizon] = ic

    return decay_curve


def assess_signal_crowding(
    signals_dict: Dict[str, np.ndarray],
    threshold: float = 0.7,
) -> Dict[str, any]:
    """Assess signal crowding by analyzing correlation between multiple signals.

    High correlation between signals indicates crowding, which can lead to
    reduced alpha and increased risk during liquidations.

    Args:
        signals_dict: Dictionary mapping signal names to signal value arrays.
        threshold: Correlation threshold above which signals are considered crowded.

    Returns:
        Dictionary containing:
        - avg_correlation: Average pairwise correlation between all signals
        - max_correlation: Maximum correlation between any two signals
        - crowded_pairs: List of (signal1, signal2, correlation) tuples above threshold
        - correlation_matrix: Full correlation matrix as dict

    Example:
        >>> signals = {"momentum": mom_sig, "mean_rev": mr_sig}
        >>> crowding = assess_signal_crowding(signals)
        >>> print(f"Avg correlation: {crowding['avg_correlation']:.3f}")
    """
    signal_names = list(signals_dict.keys())
    n_signals = len(signal_names)

    if n_signals < 2:
        return {
            "avg_correlation": 0.0,
            "max_correlation": 0.0,
            "crowded_pairs": [],
            "correlation_matrix": {},
        }

    # Build correlation matrix
    correlations = []
    correlation_matrix = {}

    for i, name1 in enumerate(signal_names):
        correlation_matrix[name1] = {}
        for j, name2 in enumerate(signal_names):
            sig1 = signals_dict[name1]
            sig2 = signals_dict[name2]

            # Align lengths
            min_len = min(len(sig1), len(sig2))
            sig1 = sig1[:min_len]
            sig2 = sig2[:min_len]

            # Remove NaNs
            mask = ~(np.isnan(sig1) | np.isnan(sig2))
            sig1_clean = sig1[mask]
            sig2_clean = sig2[mask]

            if len(sig1_clean) < 2:
                corr = 0.0
            else:
                corr, _ = spearmanr(sig1_clean, sig2_clean)
                if np.isnan(corr):
                    corr = 0.0

            correlation_matrix[name1][name2] = float(corr)

            if i < j:  # Only upper triangle for pairwise stats
                correlations.append((name1, name2, corr))

    # Statistics
    corr_values = [abs(c[2]) for c in correlations]
    avg_correlation = float(np.mean(corr_values)) if corr_values else 0.0
    max_correlation = float(np.max(corr_values)) if corr_values else 0.0

    # Crowded pairs
    crowded_pairs = [
        (name1, name2, corr) for name1, name2, corr in correlations if abs(corr) > threshold
    ]

    return {
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "crowded_pairs": crowded_pairs,
        "correlation_matrix": correlation_matrix,
    }


def calculate_signal_to_noise_ratio(
    signals: np.ndarray,
    returns: np.ndarray,
    noise_window: int = 20,
) -> float:
    """Calculate signal-to-noise ratio for a trading signal.

    SNR is computed as the ratio of the signal's predictive power to its variability.
    Higher SNR indicates more reliable signals with less noise.

    Args:
        signals: Array of signal values.
        returns: Array of corresponding returns.
        noise_window: Window size for estimating noise via rolling std.

    Returns:
        Signal-to-noise ratio. Values > 1.0 indicate signal dominates noise.

    Example:
        >>> snr = calculate_signal_to_noise_ratio(signals, returns)
        >>> print(f"SNR: {snr:.2f}")
    """
    # Remove NaNs
    mask = ~(np.isnan(signals) | np.isnan(returns))
    signals_clean = signals[mask]
    returns_clean = returns[mask]

    if len(signals_clean) < noise_window:
        return 0.0

    # Signal strength: IC
    ic = calculate_information_coefficient(signals_clean, returns_clean)
    signal_strength = abs(ic)

    # Noise: rolling standard deviation of signal changes
    signal_changes = np.diff(signals_clean)

    if len(signal_changes) < noise_window:
        return 0.0

    rolling_stds = []
    for i in range(len(signal_changes) - noise_window + 1):
        window_std = np.std(signal_changes[i : i + noise_window], ddof=1)
        rolling_stds.append(window_std)

    avg_noise = np.mean(rolling_stds) if rolling_stds else 1.0

    # SNR
    snr = signal_strength / avg_noise if avg_noise > 0 else 0.0

    return float(snr)


def rank_signals_by_quality(
    signals_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    signal_cols: List[str],
    return_col: str = "return_1d",
    forward_periods: int = 5,
) -> List[Tuple[str, Dict[str, float]]]:
    """Rank multiple signals by their quality metrics.

    Args:
        signals_df: DataFrame containing signal columns.
        returns_df: DataFrame containing returns (must be joinable with signals_df).
        signal_cols: List of signal column names to evaluate.
        return_col: Name of return column.
        forward_periods: Forward periods for IC calculation.

    Returns:
        List of (signal_name, metrics_dict) tuples, sorted by IC descending.

    Example:
        >>> signal_cols = ["momentum", "mean_reversion", "sentiment"]
        >>> ranking = rank_signals_by_quality(df, df, signal_cols)
        >>> print(f"Best signal: {ranking[0][0]} with IC {ranking[0][1]['ic']:.4f}")
    """
    results = []

    for signal_col in signal_cols:
        if signal_col not in signals_df.columns:
            continue

        # Create combined df
        df = signals_df.select([signal_col])
        if "date" in signals_df.columns and "date" in returns_df.columns:
            df = df.with_columns(signals_df["date"])
            df = df.join(returns_df.select(["date", return_col]), on="date", how="inner")
        else:
            df = df.with_columns(returns_df[return_col])

        # Calculate metrics
        try:
            metrics = calculate_signal_metrics(
                df,
                signal_col=signal_col,
                return_col=return_col,
                forward_periods=forward_periods,
            )
            results.append((signal_col, metrics))
        except Exception:
            continue

    # Sort by IC descending
    results.sort(key=lambda x: abs(x[1]["ic"]), reverse=True)

    return results
