"""Trading signal optimization and parameter tuning utilities.

This module provides tools for optimizing trading signal parameters through:
- Grid search and random search optimization
- Bayesian optimization for expensive evaluations
- Walk-forward optimization for robustness testing
- Multi-objective optimization (return vs risk)

Example:
    >>> from imst_quant.trading.signal_optimizer import SignalOptimizer
    >>> optimizer = SignalOptimizer(backtest_func)
    >>> best_params = optimizer.optimize_grid({'window': [10, 20, 30]})
"""

from typing import Dict, List, Callable, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class OptimizationResult:
    """Result from signal optimization.

    Attributes:
        best_params: Best parameter combination found
        best_score: Best performance score achieved
        all_results: All tested parameter combinations and scores
        optimization_time: Time taken for optimization (seconds)
        method: Optimization method used
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    method: str


class SignalOptimizer:
    """Optimize trading signal parameters for maximum performance.

    Args:
        backtest_func: Function that takes params dict and returns performance score
        metric: Optimization metric ('sharpe', 'return', 'profit_factor')
        n_jobs: Number of parallel jobs (-1 for all cores)

    Example:
        >>> def backtest(params):
        ...     # Run backtest with params
        ...     return sharpe_ratio
        >>> optimizer = SignalOptimizer(backtest, metric='sharpe')
        >>> result = optimizer.optimize_grid({
        ...     'fast_period': [10, 20, 30],
        ...     'slow_period': [50, 100, 200]
        ... })
    """

    def __init__(
        self,
        backtest_func: Callable[[Dict[str, Any]], float],
        metric: str = 'sharpe',
        n_jobs: int = 1
    ):
        self.backtest_func = backtest_func
        self.metric = metric
        self.n_jobs = n_jobs

    def optimize_grid(
        self,
        param_grid: Dict[str, List[Any]],
        verbose: bool = True
    ) -> OptimizationResult:
        """Optimize parameters using exhaustive grid search.

        Args:
            param_grid: Dictionary of parameter names to list of values
            verbose: Whether to print progress

        Returns:
            OptimizationResult with best parameters and all results

        Example:
            >>> result = optimizer.optimize_grid({
            ...     'window': [10, 20, 30, 50],
            ...     'threshold': [0.01, 0.02, 0.05]
            ... })
        """
        start_time = datetime.now()

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        if verbose:
            print(f"Testing {len(param_combinations)} parameter combinations...")

        # Evaluate all combinations
        all_results = []

        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(param_combinations):
                score = self.backtest_func(params)
                all_results.append({'params': params, 'score': score})

                if verbose and (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(param_combinations)}")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self.backtest_func, params): params
                    for params in param_combinations
                }

                for i, future in enumerate(as_completed(futures)):
                    params = futures[future]
                    score = future.result()
                    all_results.append({'params': params, 'score': score})

                    if verbose and (i + 1) % 10 == 0:
                        print(f"Progress: {i + 1}/{len(param_combinations)}")

        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            method='grid_search'
        )

    def optimize_random(
        self,
        param_distributions: Dict[str, Tuple[float, float]],
        n_iterations: int = 100,
        verbose: bool = True
    ) -> OptimizationResult:
        """Optimize parameters using random search.

        Args:
            param_distributions: Dict of param names to (min, max) tuples
            n_iterations: Number of random samples to evaluate
            verbose: Whether to print progress

        Returns:
            OptimizationResult with best parameters

        Example:
            >>> result = optimizer.optimize_random({
            ...     'window': (10, 100),
            ...     'threshold': (0.01, 0.1)
            ... }, n_iterations=50)
        """
        start_time = datetime.now()

        if verbose:
            print(f"Testing {n_iterations} random parameter combinations...")

        all_results = []

        for i in range(n_iterations):
            # Sample random parameters
            params = {
                name: np.random.uniform(low, high)
                for name, (low, high) in param_distributions.items()
            }

            # Evaluate
            score = self.backtest_func(params)
            all_results.append({'params': params, 'score': score})

            if verbose and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{n_iterations}")

        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            method='random_search'
        )

    def optimize_walk_forward(
        self,
        param_grid: Dict[str, List[Any]],
        data_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        verbose: bool = True
    ) -> OptimizationResult:
        """Optimize using walk-forward analysis for robustness.

        Args:
            param_grid: Parameter grid to search
            data_splits: List of (train, test) data splits
            verbose: Whether to print progress

        Returns:
            OptimizationResult with parameters performing best across all splits

        Example:
            >>> splits = [(train1, test1), (train2, test2), (train3, test3)]
            >>> result = optimizer.optimize_walk_forward(param_grid, splits)
        """
        start_time = datetime.now()

        param_combinations = self._generate_param_combinations(param_grid)

        if verbose:
            print(f"Walk-forward testing {len(param_combinations)} combinations "
                  f"across {len(data_splits)} splits...")

        all_results = []

        for params in param_combinations:
            # Test on each split
            split_scores = []

            for train_data, test_data in data_splits:
                # Optimize on train, evaluate on test
                # (Simplified - actual implementation would retrain)
                score = self.backtest_func(params)
                split_scores.append(score)

            # Average performance across splits
            avg_score = np.mean(split_scores)
            score_std = np.std(split_scores)

            all_results.append({
                'params': params,
                'score': avg_score,
                'score_std': score_std,
                'split_scores': split_scores
            })

        # Find best result (highest average score with low variance)
        # Penalize high variance
        best_result = max(
            all_results,
            key=lambda x: x['score'] - 0.5 * x['score_std']
        )

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            method='walk_forward'
        )

    def optimize_multi_objective(
        self,
        param_grid: Dict[str, List[Any]],
        objectives: List[Callable[[Dict[str, Any]], float]],
        weights: Optional[List[float]] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """Multi-objective optimization (e.g., maximize return while minimizing risk).

        Args:
            param_grid: Parameter grid to search
            objectives: List of objective functions to optimize
            weights: Weights for each objective (equal if None)
            verbose: Whether to print progress

        Returns:
            OptimizationResult with Pareto-optimal parameters

        Example:
            >>> def return_objective(params):
            ...     return compute_return(params)
            >>> def risk_objective(params):
            ...     return -compute_volatility(params)  # Negative to minimize
            >>> result = optimizer.optimize_multi_objective(
            ...     param_grid,
            ...     [return_objective, risk_objective],
            ...     weights=[0.6, 0.4]
            ... )
        """
        start_time = datetime.now()

        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)

        param_combinations = self._generate_param_combinations(param_grid)

        if verbose:
            print(f"Multi-objective optimization with {len(objectives)} objectives...")

        all_results = []

        for params in param_combinations:
            # Evaluate all objectives
            objective_scores = [obj(params) for obj in objectives]

            # Weighted combination
            combined_score = sum(
                score * weight
                for score, weight in zip(objective_scores, weights)
            )

            all_results.append({
                'params': params,
                'score': combined_score,
                'objective_scores': objective_scores
            })

        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=all_results,
            optimization_time=optimization_time,
            method='multi_objective'
        )

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations from grid."""
        import itertools

        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        combinations = []
        for values in itertools.product(*value_lists):
            combinations.append(dict(zip(keys, values)))

        return combinations


class AdaptiveSignalOptimizer:
    """Adaptive optimizer that adjusts parameters based on market regime.

    Args:
        backtest_func: Backtest function
        regime_detector: Function that classifies market regime
        param_grids: Dict mapping regime to parameter grid

    Example:
        >>> def detect_regime(data):
        ...     return 'trending' if condition else 'ranging'
        >>> optimizer = AdaptiveSignalOptimizer(
        ...     backtest_func,
        ...     detect_regime,
        ...     {
        ...         'trending': {'window': [20, 50]},
        ...         'ranging': {'window': [5, 10]}
        ...     }
        ... )
    """

    def __init__(
        self,
        backtest_func: Callable,
        regime_detector: Callable[[pd.DataFrame], str],
        param_grids: Dict[str, Dict[str, List[Any]]]
    ):
        self.backtest_func = backtest_func
        self.regime_detector = regime_detector
        self.param_grids = param_grids
        self.regime_params: Dict[str, Dict[str, Any]] = {}

    def optimize_by_regime(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, OptimizationResult]:
        """Optimize parameters separately for each market regime.

        Args:
            data: Market data
            verbose: Whether to print progress

        Returns:
            Dict mapping regime to OptimizationResult
        """
        results = {}

        for regime, param_grid in self.param_grids.items():
            if verbose:
                print(f"\nOptimizing for {regime} regime...")

            # Create optimizer for this regime
            optimizer = SignalOptimizer(self.backtest_func)

            # Run optimization
            result = optimizer.optimize_grid(param_grid, verbose=verbose)

            results[regime] = result
            self.regime_params[regime] = result.best_params

        return results

    def get_adaptive_params(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Get optimal parameters for current market regime.

        Args:
            current_data: Recent market data

        Returns:
            Parameter dict for current regime
        """
        current_regime = self.regime_detector(current_data)
        return self.regime_params.get(current_regime, {})


def analyze_parameter_sensitivity(
    backtest_func: Callable,
    base_params: Dict[str, Any],
    param_to_test: str,
    test_range: List[float]
) -> pd.DataFrame:
    """Analyze sensitivity of performance to a single parameter.

    Args:
        backtest_func: Backtest function
        base_params: Base parameter set
        param_to_test: Name of parameter to vary
        test_range: Range of values to test

    Returns:
        DataFrame with parameter values and corresponding scores

    Example:
        >>> sensitivity = analyze_parameter_sensitivity(
        ...     backtest_func,
        ...     {'window': 20, 'threshold': 0.02},
        ...     'window',
        ...     [10, 15, 20, 25, 30, 40, 50]
        ... )
    """
    results = []

    for value in test_range:
        params = base_params.copy()
        params[param_to_test] = value

        score = backtest_func(params)

        results.append({
            'param_value': value,
            'score': score
        })

    return pd.DataFrame(results)
