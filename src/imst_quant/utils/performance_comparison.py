"""Performance comparison utilities for multi-asset and multi-strategy analysis.

This module provides tools for comparing the performance of multiple assets or trading
strategies side-by-side, with statistical significance testing and relative performance metrics.

Typical usage:
    >>> import polars as pl
    >>> from imst_quant.utils.performance_comparison import compare_assets, compare_strategies
    >>>
    >>> # Compare multiple assets
    >>> df = pl.read_parquet("features.parquet")
    >>> comparison = compare_assets(df, asset_col="ticker", return_col="return_1d")
    >>>
    >>> # Compare strategies with statistical tests
    >>> comparison = compare_strategies(strategy_returns, test_significance=True)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from scipy import stats


def compare_assets(
    df: pl.DataFrame,
    asset_col: str = "asset_id",
    return_col: str = "return_1d",
    date_col: str = "date",
    risk_free_rate: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Compare performance metrics across multiple assets.

    Args:
        df: DataFrame containing asset returns with asset identifier column.
        asset_col: Name of column containing asset identifiers.
        return_col: Name of column containing return values.
        date_col: Name of column containing date/timestamp.
        risk_free_rate: Daily risk-free rate for Sharpe calculation.

    Returns:
        Dictionary mapping asset IDs to their performance metrics including:
        - total_return: Cumulative return over period
        - annualized_return: Annualized return
        - volatility: Annualized volatility
        - sharpe: Sharpe ratio
        - sortino: Sortino ratio
        - max_drawdown: Maximum drawdown
        - win_rate: Fraction of positive return periods
        - avg_win: Average return on winning periods
        - avg_loss: Average return on losing periods
        - profit_factor: Ratio of gross profits to gross losses

    Example:
        >>> comparison = compare_assets(df, asset_col="ticker")
        >>> print(f"Best Sharpe: {max(comparison.items(), key=lambda x: x[1]['sharpe'])}")
    """
    if asset_col not in df.columns or return_col not in df.columns:
        raise ValueError(f"Required columns '{asset_col}' and '{return_col}' not found")

    results = {}

    for asset_id in df[asset_col].unique().sort():
        asset_df = df.filter(pl.col(asset_col) == asset_id)
        returns = asset_df[return_col].drop_nulls()

        if returns.len() == 0:
            continue

        returns_np = returns.to_numpy()

        # Basic returns
        total_return = np.prod(1 + returns_np) - 1
        mean_return = np.mean(returns_np)
        annualized_return = (1 + mean_return) ** 252 - 1

        # Risk metrics
        volatility = np.std(returns_np, ddof=1) * np.sqrt(252)

        # Sharpe ratio
        excess_returns = returns_np - risk_free_rate
        sharpe = (
            (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)
            if np.std(excess_returns, ddof=1) > 0
            else 0.0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns_np[returns_np < risk_free_rate]
        downside_std = (
            np.std(downside_returns, ddof=1)
            if len(downside_returns) > 1
            else np.std(returns_np, ddof=1)
        )
        sortino = (
            (np.mean(excess_returns) / downside_std) * np.sqrt(252)
            if downside_std > 0
            else 0.0
        )

        # Max drawdown
        cumulative = np.cumprod(1 + returns_np)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Win/loss statistics
        wins = returns_np[returns_np > 0]
        losses = returns_np[returns_np < 0]

        win_rate = len(wins) / len(returns_np) if len(returns_np) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        results[str(asset_id)] = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "num_periods": int(len(returns_np)),
        }

    return results


def compare_strategies(
    returns_dict: Dict[str, np.ndarray],
    risk_free_rate: float = 0.0,
    test_significance: bool = True,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Compare performance of multiple strategies with statistical testing.

    Args:
        returns_dict: Dictionary mapping strategy names to return arrays.
        risk_free_rate: Daily risk-free rate for Sharpe calculation.
        test_significance: Whether to perform pairwise significance tests.
        confidence_level: Confidence level for statistical tests.

    Returns:
        Dictionary containing:
        - metrics: Per-strategy performance metrics
        - rankings: Strategies ranked by various metrics
        - pairwise_tests: Statistical test results (if test_significance=True)

    Example:
        >>> strategies = {"momentum": returns1, "mean_reversion": returns2}
        >>> comparison = compare_strategies(strategies, test_significance=True)
        >>> print(comparison["rankings"]["sharpe"])
    """
    metrics = {}

    # Calculate metrics for each strategy
    for name, returns in returns_dict.items():
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            continue

        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        mean_return = np.mean(returns)
        annualized_return = (1 + mean_return) ** 252 - 1
        volatility = np.std(returns, ddof=1) * np.sqrt(252)

        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate
        sharpe = (
            (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)
            if np.std(excess_returns, ddof=1) > 0
            else 0.0
        )

        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        calmar = (
            annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        )

        metrics[name] = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar": float(calmar),
            "num_periods": len(returns),
        }

    # Rankings
    rankings = {
        "sharpe": sorted(metrics.keys(), key=lambda k: metrics[k]["sharpe"], reverse=True),
        "total_return": sorted(metrics.keys(), key=lambda k: metrics[k]["total_return"], reverse=True),
        "calmar": sorted(metrics.keys(), key=lambda k: metrics[k]["calmar"], reverse=True),
        "volatility": sorted(metrics.keys(), key=lambda k: metrics[k]["volatility"]),
    }

    result = {"metrics": metrics, "rankings": rankings}

    # Statistical significance testing
    if test_significance and len(returns_dict) > 1:
        pairwise_tests = {}
        strategy_names = list(returns_dict.keys())

        for i, name1 in enumerate(strategy_names):
            for name2 in strategy_names[i + 1 :]:
                returns1 = returns_dict[name1]
                returns2 = returns_dict[name2]

                # Align lengths
                min_len = min(len(returns1), len(returns2))
                returns1 = returns1[:min_len]
                returns2 = returns2[:min_len]

                # T-test for mean difference
                t_stat, p_value = stats.ttest_rel(returns1, returns2)

                # Mann-Whitney U test (non-parametric)
                u_stat, u_pvalue = stats.mannwhitneyu(
                    returns1, returns2, alternative="two-sided"
                )

                is_significant = p_value < (1 - confidence_level)

                pairwise_tests[f"{name1}_vs_{name2}"] = {
                    "t_statistic": float(t_stat),
                    "t_pvalue": float(p_value),
                    "u_statistic": float(u_stat),
                    "u_pvalue": float(u_pvalue),
                    "significant": bool(is_significant),
                    "winner": name1 if np.mean(returns1) > np.mean(returns2) else name2,
                }

        result["pairwise_tests"] = pairwise_tests

    return result


def rank_by_multiple_criteria(
    comparison: Dict[str, Dict[str, float]],
    criteria: List[str] = ["sharpe", "sortino", "calmar"],
    weights: Optional[List[float]] = None,
) -> List[tuple[str, float]]:
    """Rank assets/strategies by multiple weighted criteria.

    Args:
        comparison: Dictionary of asset/strategy metrics from compare_assets or compare_strategies.
        criteria: List of metric names to use for ranking.
        weights: Optional weights for each criterion (defaults to equal weights).

    Returns:
        List of (asset_id, composite_score) tuples, sorted by composite score descending.

    Example:
        >>> comparison = compare_assets(df)
        >>> ranking = rank_by_multiple_criteria(
        ...     comparison, criteria=["sharpe", "calmar"], weights=[0.6, 0.4]
        ... )
        >>> print(f"Top asset: {ranking[0][0]} with score {ranking[0][1]:.3f}")
    """
    if weights is None:
        weights = [1.0 / len(criteria)] * len(criteria)

    if len(weights) != len(criteria):
        raise ValueError("Number of weights must match number of criteria")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    scores = {}

    for asset_id, metrics in comparison.items():
        # Normalize each criterion to [0, 1] range
        composite_score = 0.0

        for criterion, weight in zip(criteria, weights):
            if criterion not in metrics:
                continue

            # Get all values for this criterion to normalize
            all_values = [m[criterion] for m in comparison.values() if criterion in m]

            if len(all_values) == 0:
                continue

            min_val = min(all_values)
            max_val = max(all_values)

            # Normalize (handle case where all values are the same)
            if max_val - min_val > 0:
                normalized = (metrics[criterion] - min_val) / (max_val - min_val)
            else:
                normalized = 1.0

            composite_score += weight * normalized

        scores[asset_id] = composite_score

    # Sort by composite score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def format_comparison_table(
    comparison: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format comparison results as a readable ASCII table.

    Args:
        comparison: Dictionary of asset/strategy metrics.
        metrics: List of metrics to include (defaults to all).

    Returns:
        Formatted ASCII table string.

    Example:
        >>> comparison = compare_assets(df)
        >>> print(format_comparison_table(comparison, ["sharpe", "max_drawdown"]))
    """
    if not comparison:
        return "No data to display"

    if metrics is None:
        # Use all metrics from first entry
        first_entry = next(iter(comparison.values()))
        metrics = [k for k in first_entry.keys() if k != "num_periods"]

    # Build header
    col_width = 12
    header = f"{'Asset':<15}"
    for metric in metrics:
        header += f"{metric:>{col_width}}"

    separator = "-" * len(header)

    lines = [header, separator]

    # Add rows
    for asset_id, asset_metrics in comparison.items():
        row = f"{asset_id:<15}"
        for metric in metrics:
            value = asset_metrics.get(metric, 0.0)
            if isinstance(value, float):
                if abs(value) < 10:
                    row += f"{value:>{col_width}.4f}"
                else:
                    row += f"{value:>{col_width}.2f}"
            else:
                row += f"{value:>{col_width}}"
        lines.append(row)

    return "\n".join(lines)
