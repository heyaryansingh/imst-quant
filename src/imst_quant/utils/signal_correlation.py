"""Signal correlation analysis utilities for trading strategy evaluation.

Provides functions to analyze correlations between trading signals, returns,
and sentiment scores to identify multicollinearity and redundant signals.

Example:
    >>> import polars as pl
    >>> signals = pl.DataFrame({
    ...     "signal_a": [1, -1, 1, 0, 1],
    ...     "signal_b": [1, 1, -1, 0, 1],
    ...     "returns": [0.01, -0.02, 0.03, 0.0, 0.02]
    ... })
    >>> corr_matrix = calculate_signal_correlation_matrix(signals, ["signal_a", "signal_b"])
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def calculate_signal_correlation_matrix(
    df: pl.DataFrame,
    signal_cols: List[str],
    method: str = "pearson"
) -> pl.DataFrame:
    """Calculate correlation matrix between multiple trading signals.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names to analyze.
        method: Correlation method - "pearson" or "spearman".

    Returns:
        DataFrame with correlation matrix (signal_cols x signal_cols).

    Example:
        >>> signals = pl.DataFrame({
        ...     "ma_cross": [1, -1, 1, 0],
        ...     "rsi_signal": [1, 1, -1, 0],
        ...     "momentum": [0, -1, 1, 1]
        ... })
        >>> corr = calculate_signal_correlation_matrix(
        ...     signals, ["ma_cross", "rsi_signal", "momentum"]
        ... )
    """
    n_signals = len(signal_cols)
    corr_matrix = np.zeros((n_signals, n_signals))

    for i, col1 in enumerate(signal_cols):
        for j, col2 in enumerate(signal_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                if method == "pearson":
                    corr = df.select([
                        pl.corr(col1, col2).alias("correlation")
                    ])["correlation"][0]
                elif method == "spearman":
                    # Spearman = Pearson on ranks
                    rank1 = df.select(pl.col(col1).rank().alias("rank1"))["rank1"]
                    rank2 = df.select(pl.col(col2).rank().alias("rank2"))["rank2"]
                    temp_df = pl.DataFrame({"r1": rank1, "r2": rank2})
                    corr = temp_df.select([
                        pl.corr("r1", "r2").alias("correlation")
                    ])["correlation"][0]
                else:
                    raise ValueError(f"Unknown method: {method}")

                corr_matrix[i, j] = corr if corr is not None else 0.0

    # Create DataFrame with signal names as columns
    result = {}
    for i, col in enumerate(signal_cols):
        result[col] = corr_matrix[:, i].tolist()

    return pl.DataFrame(result).with_columns([
        pl.Series("signal", signal_cols)
    ]).select(["signal"] + signal_cols)


def identify_redundant_signals(
    df: pl.DataFrame,
    signal_cols: List[str],
    threshold: float = 0.8
) -> List[Tuple[str, str, float]]:
    """Identify pairs of highly correlated (redundant) signals.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names.
        threshold: Correlation threshold above which signals are redundant.

    Returns:
        List of tuples (signal1, signal2, correlation) for redundant pairs.

    Example:
        >>> signals = pl.DataFrame({
        ...     "signal_a": [1, -1, 1, 0, 1],
        ...     "signal_b": [1, -1, 1, 0, 1],  # Nearly identical
        ...     "signal_c": [0, 1, -1, 1, 0]
        ... })
        >>> redundant = identify_redundant_signals(
        ...     signals, ["signal_a", "signal_b", "signal_c"], threshold=0.9
        ... )
    """
    corr_matrix = calculate_signal_correlation_matrix(df, signal_cols)
    redundant_pairs = []

    for i, col1 in enumerate(signal_cols):
        for j, col2 in enumerate(signal_cols):
            if i < j:  # Avoid duplicates and self-correlation
                corr_val = float(corr_matrix[col2][i])
                if abs(corr_val) >= threshold:
                    redundant_pairs.append((col1, col2, corr_val))

    # Sort by absolute correlation descending
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return redundant_pairs


def calculate_signal_to_returns_correlation(
    df: pl.DataFrame,
    signal_col: str,
    returns_col: str = "returns",
    lag: int = 0
) -> float:
    """Calculate correlation between signal and future returns.

    Args:
        df: DataFrame with signal and returns columns.
        signal_col: Name of signal column.
        returns_col: Name of returns column.
        lag: Number of periods to lag returns (positive = future returns).

    Returns:
        Correlation coefficient.

    Example:
        >>> data = pl.DataFrame({
        ...     "signal": [1, -1, 1, 0, 1],
        ...     "returns": [0.01, -0.02, 0.03, 0.0, 0.02]
        ... })
        >>> corr = calculate_signal_to_returns_correlation(
        ...     data, "signal", "returns", lag=1
        ... )
    """
    if lag > 0:
        # Shift returns backward (to align with past signals)
        df_aligned = df.with_columns([
            pl.col(returns_col).shift(-lag).alias("shifted_returns")
        ])
        returns_to_use = "shifted_returns"
    elif lag < 0:
        # Shift signals forward
        df_aligned = df.with_columns([
            pl.col(signal_col).shift(-lag).alias("shifted_signal")
        ])
        signal_to_use = "shifted_signal"
        returns_to_use = returns_col
    else:
        df_aligned = df
        signal_to_use = signal_col
        returns_to_use = returns_col

    # Drop nulls from shifting
    df_aligned = df_aligned.drop_nulls([signal_to_use if lag < 0 else signal_col,
                                        returns_to_use if lag >= 0 else returns_col])

    corr = df_aligned.select([
        pl.corr(signal_col if lag >= 0 else signal_to_use,
                returns_to_use if lag >= 0 else returns_col).alias("correlation")
    ])["correlation"][0]

    return float(corr) if corr is not None else 0.0


def analyze_signal_lead_lag(
    df: pl.DataFrame,
    signal_col: str,
    returns_col: str = "returns",
    max_lag: int = 5
) -> Dict[int, float]:
    """Analyze signal-return correlation across multiple time lags.

    Useful for determining optimal signal timing and whether signals
    are predictive of future returns or lagging indicators.

    Args:
        df: DataFrame with signal and returns.
        signal_col: Name of signal column.
        returns_col: Name of returns column.
        max_lag: Maximum lag periods to test (both positive and negative).

    Returns:
        Dictionary mapping lag to correlation (positive lag = future returns).

    Example:
        >>> data = pl.DataFrame({
        ...     "signal": [1, -1, 1, 0, 1, -1, 1],
        ...     "returns": [0.01, -0.02, 0.03, 0.0, 0.02, -0.01, 0.01]
        ... })
        >>> lag_corr = analyze_signal_lead_lag(data, "signal", max_lag=3)
        >>> for lag, corr in lag_corr.items():
        ...     print(f"Lag {lag}: {corr:.3f}")
    """
    lag_correlations = {}

    for lag in range(-max_lag, max_lag + 1):
        corr = calculate_signal_to_returns_correlation(
            df, signal_col, returns_col, lag=lag
        )
        lag_correlations[lag] = corr

    return lag_correlations


def calculate_signal_diversification(
    df: pl.DataFrame,
    signal_cols: List[str]
) -> float:
    """Calculate diversification score for a set of signals.

    Higher scores indicate more diverse (less correlated) signals,
    which is desirable for robust multi-signal strategies.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names.

    Returns:
        Diversification score between 0 and 1 (higher = more diverse).

    Example:
        >>> signals = pl.DataFrame({
        ...     "signal_a": [1, -1, 1, 0],
        ...     "signal_b": [0, 1, -1, 1],
        ...     "signal_c": [-1, 0, 1, -1]
        ... })
        >>> div_score = calculate_signal_diversification(
        ...     signals, ["signal_a", "signal_b", "signal_c"]
        ... )
    """
    if len(signal_cols) < 2:
        return 1.0

    corr_matrix = calculate_signal_correlation_matrix(df, signal_cols)

    # Calculate average absolute correlation (excluding diagonal)
    n = len(signal_cols)
    total_corr = 0.0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                corr_val = float(corr_matrix[signal_cols[j]][i])
                total_corr += abs(corr_val)
                count += 1

    avg_abs_corr = total_corr / count if count > 0 else 0.0

    # Diversification score: 1 - average absolute correlation
    # Score of 1 = completely uncorrelated, 0 = perfectly correlated
    return 1.0 - avg_abs_corr


def calculate_signal_agreement(
    df: pl.DataFrame,
    signal_cols: List[str]
) -> pl.DataFrame:
    """Calculate signal agreement statistics across multiple signals.

    Provides insights into how often signals agree on direction,
    which can inform ensemble strategy design.

    Args:
        df: DataFrame containing signal columns.
        signal_cols: List of signal column names.

    Returns:
        DataFrame with agreement statistics: total_agree, bullish_agree,
        bearish_agree, neutral_agree, disagreement_rate.

    Example:
        >>> signals = pl.DataFrame({
        ...     "ma_cross": [1, -1, 1, 0, 1],
        ...     "rsi_signal": [1, -1, -1, 0, 1],
        ...     "momentum": [1, 1, -1, 1, 1]
        ... })
        >>> agreement = calculate_signal_agreement(
        ...     signals, ["ma_cross", "rsi_signal", "momentum"]
        ... )
    """
    n_rows = df.height
    n_signals = len(signal_cols)

    # Count agreement types row-by-row
    total_agree = 0
    bullish_agree = 0
    bearish_agree = 0
    neutral_agree = 0

    for i in range(n_rows):
        row_signals = [df[col][i] for col in signal_cols]

        # Check if all non-zero signals agree
        non_zero = [s for s in row_signals if s != 0]

        if len(non_zero) == 0:
            neutral_agree += 1
            total_agree += 1
        elif len(non_zero) == 1:
            total_agree += 1
            if non_zero[0] > 0:
                bullish_agree += 1
            else:
                bearish_agree += 1
        else:
            # Check agreement
            all_positive = all(s > 0 for s in non_zero)
            all_negative = all(s < 0 for s in non_zero)

            if all_positive:
                bullish_agree += 1
                total_agree += 1
            elif all_negative:
                bearish_agree += 1
                total_agree += 1

    disagreement_rate = 1.0 - (total_agree / n_rows) if n_rows > 0 else 0.0

    return pl.DataFrame({
        "total_agreement": [total_agree],
        "bullish_agreement": [bullish_agree],
        "bearish_agreement": [bearish_agree],
        "neutral_agreement": [neutral_agree],
        "disagreement_rate": [disagreement_rate],
        "agreement_rate": [total_agree / n_rows if n_rows > 0 else 0.0]
    })
