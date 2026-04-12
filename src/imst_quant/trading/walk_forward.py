"""Walk-forward validation for trading strategy optimization.

This module implements walk-forward analysis (WFA), a robust validation
technique that splits data into multiple train/test windows and evaluates
out-of-sample performance. This helps detect overfitting and provides
more realistic performance estimates.

Features:
    - Configurable window sizes and step sizes
    - Anchored and rolling window modes
    - Multiple performance metrics
    - Parameter stability analysis
    - Aggregated out-of-sample results

Example:
    >>> from imst_quant.trading.walk_forward import WalkForwardValidator
    >>> validator = WalkForwardValidator(
    ...     train_size=252,  # 1 year training
    ...     test_size=63,    # 3 months testing
    ...     step_size=63,    # Step forward 3 months
    ... )
    >>> results = validator.run(features_df, strategy_fn, param_grid)
    >>> print(f"OOS Sharpe: {results['oos_sharpe']:.2f}")

Note:
    Walk-forward validation is computationally expensive but provides
    much more reliable performance estimates than simple train/test splits.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class WindowResult:
    """Results from a single walk-forward window.

    Attributes:
        window_id: Sequential window identifier.
        train_start: Training period start date.
        train_end: Training period end date.
        test_start: Test period start date.
        test_end: Test period end date.
        best_params: Optimized parameters from training.
        train_metrics: Performance metrics on training data.
        test_metrics: Performance metrics on test data (out-of-sample).
    """

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        train_size: Number of periods for training window.
        test_size: Number of periods for test window.
        step_size: Number of periods to step forward.
        anchored: If True, training always starts from beginning.
        min_train_size: Minimum training periods required.
        optimization_metric: Metric to optimize during training.
        higher_is_better: Whether higher metric values are better.
    """

    train_size: int = 252
    test_size: int = 63
    step_size: int = 63
    anchored: bool = False
    min_train_size: int = 126
    optimization_metric: str = "sharpe"
    higher_is_better: bool = True


class WalkForwardValidator:
    """Walk-forward analysis for strategy validation.

    This class implements rolling or anchored walk-forward validation
    to evaluate trading strategies on out-of-sample data. It helps
    detect overfitting by testing optimized parameters on future data.

    Attributes:
        config: Walk-forward configuration.
        windows: List of window results after running validation.

    Example:
        >>> validator = WalkForwardValidator(train_size=252, test_size=63)
        >>> results = validator.run(
        ...     data=features_df,
        ...     strategy_fn=my_strategy,
        ...     param_grid={"lookback": [5, 10, 20]},
        ... )
        >>> for window in results["windows"]:
        ...     print(f"Window {window.window_id}: OOS Sharpe = {window.test_metrics['sharpe']:.2f}")
    """

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 63,
        config: Optional[WalkForwardConfig] = None,
    ) -> None:
        """Initialize the walk-forward validator.

        Args:
            train_size: Number of periods for training window.
            test_size: Number of periods for test window.
            step_size: Number of periods to step forward.
            config: Optional full configuration object.
        """
        if config is not None:
            self.config = config
        else:
            self.config = WalkForwardConfig(
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )

        self.windows: List[WindowResult] = []

        logger.info(
            "walk_forward_validator_initialized",
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            step_size=self.config.step_size,
            anchored=self.config.anchored,
        )

    def _generate_windows(
        self,
        n_periods: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate train/test window indices.

        Args:
            n_periods: Total number of periods in the data.

        Returns:
            List of tuples (train_start, train_end, test_start, test_end).
        """
        windows = []
        min_required = self.config.train_size + self.config.test_size

        if n_periods < min_required:
            logger.warning(
                "insufficient_data_for_walk_forward",
                n_periods=n_periods,
                min_required=min_required,
            )
            return windows

        train_start = 0
        while True:
            if self.config.anchored:
                train_start_idx = 0
            else:
                train_start_idx = train_start

            train_end_idx = train_start + self.config.train_size
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.config.test_size

            if test_end_idx > n_periods:
                break

            # Check minimum training size for anchored mode
            actual_train_size = train_end_idx - train_start_idx
            if actual_train_size >= self.config.min_train_size:
                windows.append((
                    train_start_idx,
                    train_end_idx,
                    test_start_idx,
                    test_end_idx,
                ))

            train_start += self.config.step_size

        logger.info("walk_forward_windows_generated", n_windows=len(windows))
        return windows

    def _optimize_params(
        self,
        train_data: pl.DataFrame,
        strategy_fn: Callable,
        param_grid: Dict[str, List],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Find optimal parameters on training data.

        Args:
            train_data: Training period data.
            strategy_fn: Strategy function to evaluate.
            param_grid: Parameter grid to search.

        Returns:
            Tuple of (best_params, best_metrics).
        """
        from itertools import product

        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        best_params: Dict[str, Any] = {}
        best_metric_value = float("-inf") if self.config.higher_is_better else float("inf")
        best_metrics: Dict[str, float] = {}

        for combo in product(*values):
            params = dict(zip(keys, combo))
            try:
                metrics = strategy_fn(train_data, **params)
                metric_value = metrics.get(self.config.optimization_metric, 0.0)

                is_better = (
                    metric_value > best_metric_value
                    if self.config.higher_is_better
                    else metric_value < best_metric_value
                )

                if is_better:
                    best_metric_value = metric_value
                    best_params = params.copy()
                    best_metrics = metrics.copy()

            except Exception as e:
                logger.warning(
                    "param_combination_failed",
                    params=params,
                    error=str(e),
                )
                continue

        return best_params, best_metrics

    def run(
        self,
        data: pl.DataFrame,
        strategy_fn: Callable[[pl.DataFrame], Dict[str, float]],
        param_grid: Optional[Dict[str, List]] = None,
        date_column: str = "date",
    ) -> Dict:
        """Run walk-forward validation.

        Args:
            data: DataFrame with features and returns, sorted by date.
            strategy_fn: Function that takes data + params and returns metrics.
                Signature: (data: pl.DataFrame, **params) -> Dict[str, float]
            param_grid: Optional parameter grid for optimization.
                If None, uses default parameters.
            date_column: Name of the date column.

        Returns:
            Dictionary containing:
                - windows: List of WindowResult objects
                - summary: Aggregated out-of-sample statistics
                - param_stability: Parameter stability analysis
        """
        self.windows = []

        # Sort data by date
        data = data.sort(date_column)
        dates = data[date_column].to_list()
        n_periods = len(data)

        # Generate windows
        window_indices = self._generate_windows(n_periods)

        if not window_indices:
            return {
                "windows": [],
                "summary": {},
                "param_stability": {},
                "error": "Insufficient data for walk-forward validation",
            }

        param_grid = param_grid or {}

        for i, (train_start, train_end, test_start, test_end) in enumerate(window_indices):
            logger.info(
                "processing_walk_forward_window",
                window=i + 1,
                total=len(window_indices),
            )

            train_data = data.slice(train_start, train_end - train_start)
            test_data = data.slice(test_start, test_end - test_start)

            # Optimize on training data
            if param_grid:
                best_params, train_metrics = self._optimize_params(
                    train_data,
                    strategy_fn,
                    param_grid,
                )
            else:
                best_params = {}
                train_metrics = strategy_fn(train_data)

            # Evaluate on test data with optimized params
            try:
                test_metrics = strategy_fn(test_data, **best_params)
            except Exception as e:
                logger.error(
                    "test_evaluation_failed",
                    window=i + 1,
                    error=str(e),
                )
                test_metrics = {"sharpe": 0.0, "total_return": 0.0}

            window_result = WindowResult(
                window_id=i + 1,
                train_start=str(dates[train_start]),
                train_end=str(dates[train_end - 1]),
                test_start=str(dates[test_start]),
                test_end=str(dates[test_end - 1]),
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            self.windows.append(window_result)

        # Calculate summary statistics
        summary = self._calculate_summary()
        param_stability = self._analyze_param_stability(param_grid)

        return {
            "windows": self.windows,
            "summary": summary,
            "param_stability": param_stability,
        }

    def _calculate_summary(self) -> Dict[str, float]:
        """Calculate aggregate statistics from all windows.

        Returns:
            Dictionary of summary statistics.
        """
        if not self.windows:
            return {}

        # Collect test metrics
        oos_sharpes = [w.test_metrics.get("sharpe", 0.0) for w in self.windows]
        oos_returns = [w.test_metrics.get("total_return", 0.0) for w in self.windows]
        is_sharpes = [w.train_metrics.get("sharpe", 0.0) for w in self.windows]

        # Calculate statistics
        avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0.0
        avg_is_sharpe = sum(is_sharpes) / len(is_sharpes) if is_sharpes else 0.0

        # Sharpe decay (overfitting indicator)
        sharpe_decay = (
            (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe
            if avg_is_sharpe != 0
            else 0.0
        )

        # Win rate of OOS periods
        positive_oos = sum(1 for r in oos_returns if r > 0)
        oos_win_rate = positive_oos / len(oos_returns) if oos_returns else 0.0

        # Std dev of OOS Sharpe (consistency)
        if len(oos_sharpes) > 1:
            mean_sharpe = avg_oos_sharpe
            variance = sum((s - mean_sharpe) ** 2 for s in oos_sharpes) / len(oos_sharpes)
            sharpe_std = variance ** 0.5
        else:
            sharpe_std = 0.0

        return {
            "n_windows": len(self.windows),
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_is_sharpe": avg_is_sharpe,
            "sharpe_decay_pct": sharpe_decay * 100,
            "oos_sharpe_std": sharpe_std,
            "oos_win_rate": oos_win_rate,
            "total_oos_return": sum(oos_returns),
            "avg_oos_return": sum(oos_returns) / len(oos_returns) if oos_returns else 0.0,
        }

    def _analyze_param_stability(
        self,
        param_grid: Dict[str, List],
    ) -> Dict[str, Dict]:
        """Analyze parameter stability across windows.

        Args:
            param_grid: Original parameter grid.

        Returns:
            Dictionary with stability metrics per parameter.
        """
        if not self.windows or not param_grid:
            return {}

        stability = {}
        for param_name in param_grid.keys():
            values = [
                w.best_params.get(param_name)
                for w in self.windows
                if param_name in w.best_params
            ]

            if not values:
                continue

            # Check if values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                    std_val = variance ** 0.5
                else:
                    std_val = 0.0

                stability[param_name] = {
                    "values": values,
                    "mean": mean_val,
                    "std": std_val,
                    "cv": std_val / abs(mean_val) if mean_val != 0 else 0.0,
                    "unique_count": len(set(values)),
                }
            else:
                # Categorical parameter
                from collections import Counter
                counts = Counter(values)
                most_common = counts.most_common(1)[0] if counts else (None, 0)

                stability[param_name] = {
                    "values": values,
                    "value_counts": dict(counts),
                    "most_common": most_common[0],
                    "most_common_pct": most_common[1] / len(values) if values else 0.0,
                }

        return stability

    def get_oos_equity_curve(self) -> List[float]:
        """Construct cumulative equity curve from OOS returns.

        Returns:
            List of cumulative returns across all OOS periods.
        """
        equity = [1.0]
        for window in self.windows:
            ret = window.test_metrics.get("total_return", 0.0)
            equity.append(equity[-1] * (1 + ret))
        return equity

    def to_dict(self) -> Dict:
        """Convert results to dictionary format.

        Returns:
            Dictionary representation of all results.
        """
        return {
            "config": {
                "train_size": self.config.train_size,
                "test_size": self.config.test_size,
                "step_size": self.config.step_size,
                "anchored": self.config.anchored,
            },
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_period": f"{w.train_start} to {w.train_end}",
                    "test_period": f"{w.test_start} to {w.test_end}",
                    "best_params": w.best_params,
                    "train_metrics": w.train_metrics,
                    "test_metrics": w.test_metrics,
                }
                for w in self.windows
            ],
            "summary": self._calculate_summary(),
            "oos_equity_curve": self.get_oos_equity_curve(),
        }
