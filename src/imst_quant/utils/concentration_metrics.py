"""Portfolio concentration and diversification metrics.

This module provides metrics to measure portfolio concentration, diversification,
and asset allocation quality. All metrics work with Polars DataFrames.

Functions:
    herfindahl_index: Calculate Herfindahl-Hirschman Index (HHI)
    effective_n: Calculate effective number of positions
    concentration_ratio: Calculate top-N concentration ratio
    gini_coefficient: Calculate Gini coefficient for position sizes
    shannon_entropy: Calculate Shannon entropy for diversification
    calculate_all_concentration: Compute all concentration metrics

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.concentration_metrics import calculate_all_concentration
    >>> weights = pl.Series("weights", [0.4, 0.3, 0.2, 0.1])
    >>> metrics = calculate_all_concentration(weights)
    >>> print(f"HHI: {metrics['hhi']:.4f}")
"""

from typing import Dict, Union

import numpy as np
import polars as pl


def herfindahl_index(
    weights: Union[pl.Series, pl.DataFrame],
    weight_col: str = "weight",
) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration.

    HHI measures concentration by summing squared weights. Ranges from 1/N
    (perfectly diversified) to 1.0 (fully concentrated).

    Args:
        weights: Series or DataFrame containing position weights.
        weight_col: Column name if weights is a DataFrame.

    Returns:
        HHI value between 0 and 1.

    Example:
        >>> weights = pl.Series([0.25, 0.25, 0.25, 0.25])
        >>> hhi = herfindahl_index(weights)
        >>> print(f"HHI: {hhi:.4f}")  # Should be 0.25 for equal weights
    """
    if isinstance(weights, pl.DataFrame):
        w = weights[weight_col]
    else:
        w = weights

    # Normalize weights to sum to 1
    total = w.sum()
    if total == 0:
        return 0.0

    normalized = w / total
    hhi = (normalized ** 2).sum()

    return float(hhi)


def effective_n(
    weights: Union[pl.Series, pl.DataFrame],
    weight_col: str = "weight",
) -> float:
    """Calculate effective number of positions (1 / HHI).

    This metric represents the equivalent number of equal-weighted positions
    that would produce the same concentration as the current portfolio.

    Args:
        weights: Series or DataFrame containing position weights.
        weight_col: Column name if weights is a DataFrame.

    Returns:
        Effective number of positions.

    Example:
        >>> weights = pl.Series([0.5, 0.3, 0.2])
        >>> n = effective_n(weights)
        >>> print(f"Effective N: {n:.2f}")
    """
    hhi = herfindahl_index(weights, weight_col)
    if hhi == 0:
        return 0.0

    return 1.0 / hhi


def concentration_ratio(
    weights: Union[pl.Series, pl.DataFrame],
    top_n: int = 5,
    weight_col: str = "weight",
) -> float:
    """Calculate top-N concentration ratio.

    Measures the proportion of total portfolio value held in the top N positions.

    Args:
        weights: Series or DataFrame containing position weights.
        top_n: Number of top positions to include (default: 5).
        weight_col: Column name if weights is a DataFrame.

    Returns:
        Concentration ratio (0 to 1).

    Example:
        >>> weights = pl.Series([0.4, 0.3, 0.2, 0.1])
        >>> cr = concentration_ratio(weights, top_n=2)
        >>> print(f"Top 2 Concentration: {cr:.2%}")
    """
    if isinstance(weights, pl.DataFrame):
        w = weights[weight_col]
    else:
        w = weights

    # Normalize and sort descending
    total = w.sum()
    if total == 0:
        return 0.0

    normalized = w / total
    top_weights = normalized.top_k(min(top_n, len(normalized)))

    return float(top_weights.sum())


def gini_coefficient(
    weights: Union[pl.Series, pl.DataFrame],
    weight_col: str = "weight",
) -> float:
    """Calculate Gini coefficient for position size inequality.

    Gini coefficient measures inequality in portfolio allocation.
    0 = perfect equality, 1 = maximum inequality.

    Args:
        weights: Series or DataFrame containing position weights.
        weight_col: Column name if weights is a DataFrame.

    Returns:
        Gini coefficient (0 to 1).

    Example:
        >>> weights = pl.Series([0.7, 0.2, 0.1])
        >>> gini = gini_coefficient(weights)
        >>> print(f"Gini: {gini:.4f}")
    """
    if isinstance(weights, pl.DataFrame):
        w = weights[weight_col]
    else:
        w = weights

    # Convert to numpy for calculation
    sorted_weights = np.sort(w.to_numpy())
    n = len(sorted_weights)

    if n == 0 or sorted_weights.sum() == 0:
        return 0.0

    # Normalized cumulative sum
    cumsum = np.cumsum(sorted_weights)
    cumsum_norm = cumsum / cumsum[-1]

    # Calculate Gini using trapezoid rule
    gini = 1 - 2 * np.trapz(cumsum_norm, dx=1/n)

    return float(gini)


def shannon_entropy(
    weights: Union[pl.Series, pl.DataFrame],
    weight_col: str = "weight",
) -> float:
    """Calculate Shannon entropy for portfolio diversification.

    Higher entropy indicates better diversification. Maximum entropy
    is log(N) for N equal-weighted positions.

    Args:
        weights: Series or DataFrame containing position weights.
        weight_col: Column name if weights is a DataFrame.

    Returns:
        Shannon entropy value.

    Example:
        >>> weights = pl.Series([0.25, 0.25, 0.25, 0.25])
        >>> entropy = shannon_entropy(weights)
        >>> print(f"Entropy: {entropy:.4f}")
    """
    if isinstance(weights, pl.DataFrame):
        w = weights[weight_col]
    else:
        w = weights

    # Normalize weights
    total = w.sum()
    if total == 0:
        return 0.0

    normalized = w / total

    # Filter out zero weights and calculate entropy
    nonzero = normalized.filter(normalized > 0)
    entropy = -(nonzero * nonzero.log()).sum()

    return float(entropy)


def calculate_all_concentration(
    weights: Union[pl.Series, pl.DataFrame],
    weight_col: str = "weight",
    top_n: int = 5,
) -> Dict[str, float]:
    """Calculate all concentration metrics at once.

    Args:
        weights: Series or DataFrame containing position weights.
        weight_col: Column name if weights is a DataFrame.
        top_n: Number of top positions for concentration ratio.

    Returns:
        Dictionary with all concentration metrics.

    Example:
        >>> weights = pl.Series([0.4, 0.3, 0.2, 0.1])
        >>> metrics = calculate_all_concentration(weights, top_n=2)
        >>> for name, value in metrics.items():
        ...     print(f"{name}: {value:.4f}")
    """
    if isinstance(weights, pl.DataFrame):
        w = weights[weight_col]
    else:
        w = weights

    n_positions = len(w.filter(w > 0))
    max_entropy = np.log(n_positions) if n_positions > 0 else 0.0
    entropy = shannon_entropy(w, weight_col)

    return {
        "hhi": herfindahl_index(w, weight_col),
        "effective_n": effective_n(w, weight_col),
        f"top_{top_n}_concentration": concentration_ratio(w, top_n, weight_col),
        "gini": gini_coefficient(w, weight_col),
        "shannon_entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "n_positions": n_positions,
    }
