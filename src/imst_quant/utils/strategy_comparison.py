"""Strategy comparison utility for comparing multiple trading strategies side-by-side.

This module provides tools to compare different trading strategies based on their
backtest results, performance metrics, and risk-adjusted returns.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger()


def compare_strategies(
    strategy_results: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compare multiple trading strategies side-by-side.

    Args:
        strategy_results: Dictionary mapping strategy names to their backtest results.
            Each result dict should contain metrics like 'sharpe', 'total_return',
            'max_drawdown', 'win_rate', etc.
        output_path: Optional path to save comparison table as CSV.

    Returns:
        DataFrame with strategies as rows and metrics as columns.

    Example:
        >>> results = {
        ...     'LSTM': {'sharpe': 1.5, 'total_return': 0.25, 'max_drawdown': -0.12},
        ...     'CNN': {'sharpe': 1.3, 'total_return': 0.20, 'max_drawdown': -0.10},
        ... }
        >>> df = compare_strategies(results)
        >>> print(df)
    """
    if not strategy_results:
        raise ValueError("strategy_results cannot be empty")

    comparison_data = []

    for strategy_name, metrics in strategy_results.items():
        row = {"strategy": strategy_name}
        row.update(metrics)
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.set_index("strategy")

    # Sort by Sharpe ratio descending if available
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False)

    logger.info(
        "strategy_comparison_complete",
        num_strategies=len(strategy_results),
        best_strategy=df.index[0] if not df.empty else None,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("comparison_saved", path=str(output_path))

    return df


def rank_strategies(
    strategy_results: Dict[str, Dict],
    metric: str = "sharpe",
    ascending: bool = False,
) -> List[tuple]:
    """Rank strategies by a specific metric.

    Args:
        strategy_results: Dictionary mapping strategy names to their metrics.
        metric: Metric to rank by (e.g., 'sharpe', 'total_return', 'max_drawdown').
        ascending: If True, rank from lowest to highest. If False, highest to lowest.

    Returns:
        List of (strategy_name, metric_value) tuples, sorted by the metric.

    Example:
        >>> results = {'LSTM': {'sharpe': 1.5}, 'CNN': {'sharpe': 1.3}}
        >>> rank_strategies(results, metric='sharpe')
        [('LSTM', 1.5), ('CNN', 1.3)]
    """
    rankings = [
        (name, metrics.get(metric))
        for name, metrics in strategy_results.items()
        if metric in metrics
    ]

    # Filter out None values
    rankings = [(name, value) for name, value in rankings if value is not None]

    rankings.sort(key=lambda x: x[1], reverse=not ascending)

    logger.info(
        "strategies_ranked",
        metric=metric,
        num_strategies=len(rankings),
        best=rankings[0][0] if rankings else None,
    )

    return rankings


def calculate_relative_performance(
    strategy_results: Dict[str, Dict],
    benchmark_strategy: str,
    metric: str = "total_return",
) -> Dict[str, float]:
    """Calculate relative performance of strategies compared to a benchmark.

    Args:
        strategy_results: Dictionary mapping strategy names to their metrics.
        benchmark_strategy: Name of the strategy to use as benchmark.
        metric: Metric to compare (e.g., 'total_return', 'sharpe').

    Returns:
        Dictionary mapping strategy names to their relative performance
        (strategy_metric / benchmark_metric - 1).

    Example:
        >>> results = {'LSTM': {'total_return': 0.25}, 'Benchmark': {'total_return': 0.15}}
        >>> calculate_relative_performance(results, 'Benchmark', 'total_return')
        {'LSTM': 0.6667, 'Benchmark': 0.0}
    """
    if benchmark_strategy not in strategy_results:
        raise ValueError(f"Benchmark strategy '{benchmark_strategy}' not found")

    benchmark_value = strategy_results[benchmark_strategy].get(metric)
    if benchmark_value is None:
        raise ValueError(
            f"Metric '{metric}' not found in benchmark strategy '{benchmark_strategy}'"
        )

    if benchmark_value == 0:
        logger.warning(
            "benchmark_value_zero",
            metric=metric,
            msg="Cannot calculate relative performance with zero benchmark",
        )
        return {}

    relative_perf = {}
    for name, metrics in strategy_results.items():
        value = metrics.get(metric)
        if value is not None:
            relative_perf[name] = (value / benchmark_value) - 1

    logger.info(
        "relative_performance_calculated",
        benchmark=benchmark_strategy,
        metric=metric,
        num_strategies=len(relative_perf),
    )

    return relative_perf


def find_best_strategy(
    strategy_results: Dict[str, Dict],
    primary_metric: str = "sharpe",
    secondary_metric: Optional[str] = "max_drawdown",
    secondary_ascending: bool = True,
) -> str:
    """Find the best strategy based on primary and optional secondary metric.

    Args:
        strategy_results: Dictionary mapping strategy names to their metrics.
        primary_metric: Primary metric to optimize (e.g., 'sharpe').
        secondary_metric: Optional tiebreaker metric (e.g., 'max_drawdown').
        secondary_ascending: Direction for secondary metric (True for minimize).

    Returns:
        Name of the best strategy.

    Example:
        >>> results = {
        ...     'LSTM': {'sharpe': 1.5, 'max_drawdown': -0.12},
        ...     'CNN': {'sharpe': 1.5, 'max_drawdown': -0.10},
        ... }
        >>> find_best_strategy(results)
        'CNN'  # Same Sharpe, but better (smaller) max drawdown
    """
    if not strategy_results:
        raise ValueError("strategy_results cannot be empty")

    # Filter strategies that have the primary metric
    valid_strategies = {
        name: metrics
        for name, metrics in strategy_results.items()
        if primary_metric in metrics
    }

    if not valid_strategies:
        raise ValueError(f"No strategies have the primary metric '{primary_metric}'")

    # Sort by primary metric (descending)
    sorted_strategies = sorted(
        valid_strategies.items(),
        key=lambda x: x[1][primary_metric],
        reverse=True,
    )

    # Get top performers (those with max primary metric value)
    max_primary = sorted_strategies[0][1][primary_metric]
    top_performers = [
        (name, metrics)
        for name, metrics in sorted_strategies
        if metrics[primary_metric] == max_primary
    ]

    # If tie and secondary metric provided, use it
    if len(top_performers) > 1 and secondary_metric:
        top_performers_with_secondary = [
            (name, metrics)
            for name, metrics in top_performers
            if secondary_metric in metrics
        ]

        if top_performers_with_secondary:
            top_performers_with_secondary.sort(
                key=lambda x: x[1][secondary_metric],
                reverse=not secondary_ascending,
            )
            best = top_performers_with_secondary[0][0]
        else:
            best = top_performers[0][0]
    else:
        best = top_performers[0][0]

    logger.info(
        "best_strategy_found",
        strategy=best,
        primary_metric=primary_metric,
        primary_value=strategy_results[best][primary_metric],
    )

    return best
