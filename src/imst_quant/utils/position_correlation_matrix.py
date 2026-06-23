"""Position correlation matrix calculator for analyzing portfolio diversification.

This module provides tools to calculate and visualize correlation matrices
between portfolio positions to assess diversification quality.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger()


def calculate_position_correlation_matrix(
    returns_df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate correlation matrix between position returns.

    Args:
        returns_df: DataFrame with datetime index and columns for each position's returns.
        method: Correlation method ('pearson', 'kendall', or 'spearman').

    Returns:
        Correlation matrix as DataFrame.

    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03],
        ...     'MSFT': [0.02, -0.01, 0.02],
        ...     'GOOGL': [-0.01, 0.02, 0.01],
        ... })
        >>> corr_matrix = calculate_position_correlation_matrix(returns)
    """
    if returns_df.empty:
        raise ValueError("returns_df cannot be empty")

    correlation_matrix = returns_df.corr(method=method)

    logger.info(
        "correlation_matrix_calculated",
        num_positions=len(correlation_matrix),
        method=method,
        avg_correlation=correlation_matrix.values[
            ~pd.np.eye(len(correlation_matrix), dtype=bool)
        ]
        .mean()
        if len(correlation_matrix) > 1
        else 0,
    )

    return correlation_matrix


def identify_highly_correlated_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> List[tuple]:
    """Identify pairs of positions with correlation above threshold.

    Args:
        correlation_matrix: Correlation matrix from calculate_position_correlation_matrix.
        threshold: Minimum correlation to flag (0 to 1).

    Returns:
        List of (position1, position2, correlation) tuples.

    Example:
        >>> corr = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=['A', 'B'], columns=['A', 'B'])
        >>> identify_highly_correlated_pairs(corr, threshold=0.8)
        [('A', 'B', 0.9)]
    """
    high_corr_pairs = []

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                pos1 = correlation_matrix.index[i]
                pos2 = correlation_matrix.columns[j]
                high_corr_pairs.append((pos1, pos2, corr_value))

    # Sort by absolute correlation descending
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    logger.info(
        "highly_correlated_pairs_identified",
        num_pairs=len(high_corr_pairs),
        threshold=threshold,
    )

    return high_corr_pairs


def calculate_diversification_ratio(
    weights: pd.Series,
    correlation_matrix: pd.DataFrame,
    volatilities: pd.Series,
) -> float:
    """Calculate portfolio diversification ratio.

    Diversification ratio = (weighted avg volatility) / (portfolio volatility)
    Higher values indicate better diversification.

    Args:
        weights: Series of position weights (must sum to 1).
        correlation_matrix: Position correlation matrix.
        volatilities: Series of position volatilities (standard deviations).

    Returns:
        Diversification ratio (typically between 1 and sqrt(N)).

    Example:
        >>> weights = pd.Series({'A': 0.5, 'B': 0.5})
        >>> corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=['A', 'B'], columns=['A', 'B'])
        >>> vols = pd.Series({'A': 0.2, 'B': 0.3})
        >>> calculate_diversification_ratio(weights, corr, vols)
    """
    if not abs(weights.sum() - 1.0) < 1e-6:
        raise ValueError(f"Weights must sum to 1, got {weights.sum()}")

    # Weighted average volatility
    weighted_avg_vol = (weights * volatilities).sum()

    # Portfolio volatility
    covariance_matrix = correlation_matrix * pd.np.outer(volatilities, volatilities)
    portfolio_variance = weights @ covariance_matrix @ weights
    portfolio_vol = pd.np.sqrt(portfolio_variance)

    if portfolio_vol == 0:
        return 0.0

    diversification_ratio = weighted_avg_vol / portfolio_vol

    logger.info(
        "diversification_ratio_calculated",
        ratio=diversification_ratio,
        portfolio_vol=portfolio_vol,
        weighted_avg_vol=weighted_avg_vol,
    )

    return diversification_ratio


def suggest_diversification_improvements(
    correlation_matrix: pd.DataFrame,
    weights: pd.Series,
    high_correlation_threshold: float = 0.8,
) -> Dict[str, List[str]]:
    """Suggest ways to improve portfolio diversification.

    Args:
        correlation_matrix: Position correlation matrix.
        weights: Current position weights.
        high_correlation_threshold: Threshold to flag high correlation.

    Returns:
        Dictionary with 'reduce_overlap' and 'increase_exposure' suggestions.

    Example:
        >>> corr = pd.DataFrame([[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]],
        ...                     index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
        >>> weights = pd.Series({'A': 0.4, 'B': 0.4, 'C': 0.2})
        >>> suggest_diversification_improvements(corr, weights)
    """
    suggestions = {
        "reduce_overlap": [],
        "increase_exposure": [],
    }

    # Find highly correlated pairs
    high_corr_pairs = identify_highly_correlated_pairs(
        correlation_matrix, high_correlation_threshold
    )

    for pos1, pos2, corr_val in high_corr_pairs:
        weight1 = weights.get(pos1, 0)
        weight2 = weights.get(pos2, 0)

        if weight1 > 0 and weight2 > 0:
            suggestions["reduce_overlap"].append(
                f"Consider reducing exposure to {pos1} or {pos2} "
                f"(correlation: {corr_val:.2f}, weights: {weight1:.2%}/{weight2:.2%})"
            )

    # Find low-correlation positions with small weights
    avg_correlations = correlation_matrix.abs().mean()
    for pos in correlation_matrix.index:
        avg_corr = avg_correlations[pos]
        weight = weights.get(pos, 0)

        if avg_corr < 0.5 and weight < 0.1:
            suggestions["increase_exposure"].append(
                f"Consider increasing exposure to {pos} "
                f"(low avg correlation: {avg_corr:.2f}, current weight: {weight:.2%})"
            )

    logger.info(
        "diversification_suggestions_generated",
        reduce_overlap_count=len(suggestions["reduce_overlap"]),
        increase_exposure_count=len(suggestions["increase_exposure"]),
    )

    return suggestions


def export_correlation_report(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
    weights: Optional[pd.Series] = None,
    volatilities: Optional[pd.Series] = None,
) -> None:
    """Export a comprehensive correlation analysis report.

    Args:
        correlation_matrix: Position correlation matrix.
        output_path: Path to save the report (CSV or Excel).
        weights: Optional position weights for diversification ratio.
        volatilities: Optional position volatilities for diversification ratio.

    Example:
        >>> corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=['A', 'B'], columns=['A', 'B'])
        >>> export_correlation_report(corr, Path('correlation_report.csv'))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Correlation matrix
            correlation_matrix.to_excel(writer, sheet_name="Correlation Matrix")

            # Highly correlated pairs
            high_corr = identify_highly_correlated_pairs(correlation_matrix)
            if high_corr:
                pd.DataFrame(
                    high_corr, columns=["Position 1", "Position 2", "Correlation"]
                ).to_excel(writer, sheet_name="High Correlations", index=False)

            # Diversification metrics
            if weights is not None and volatilities is not None:
                div_ratio = calculate_diversification_ratio(
                    weights, correlation_matrix, volatilities
                )
                pd.DataFrame(
                    {"Metric": ["Diversification Ratio"], "Value": [div_ratio]}
                ).to_excel(writer, sheet_name="Diversification", index=False)

    else:
        # CSV fallback - just save correlation matrix
        correlation_matrix.to_csv(output_path)

    logger.info("correlation_report_exported", path=str(output_path))
