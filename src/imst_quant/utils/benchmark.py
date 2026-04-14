"""Benchmark comparison and relative performance analysis.

This module provides utilities for comparing strategy performance against
common market benchmarks and calculating relative metrics like alpha,
beta, tracking error, and information ratio.

Features:
- Benchmark return calculation (buy-and-hold, equal-weight)
- Alpha and beta estimation via regression
- Information ratio and tracking error
- Up/down capture ratios
- Rolling relative performance
- Benchmark-adjusted metrics

Example:
    >>> from imst_quant.utils.benchmark import BenchmarkAnalyzer
    >>> analyzer = BenchmarkAnalyzer(strategy_returns, benchmark_returns)
    >>> metrics = analyzer.calculate_all_metrics()
    >>> print(f"Alpha: {metrics['alpha']:.2%}, Beta: {metrics['beta']:.2f}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class BenchmarkMetrics:
    """Container for benchmark comparison metrics.

    Attributes:
        alpha: Jensen's alpha (annualized excess return above CAPM).
        beta: Market sensitivity (CAPM beta).
        r_squared: Coefficient of determination from regression.
        tracking_error: Annualized standard deviation of excess returns.
        information_ratio: Active return per unit of active risk.
        up_capture: Percentage of benchmark gains captured.
        down_capture: Percentage of benchmark losses captured.
        capture_ratio: Up capture / down capture (higher is better).
        excess_return: Total strategy return minus benchmark return.
        correlation: Correlation between strategy and benchmark.
    """

    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    capture_ratio: float
    excess_return: float
    correlation: float


class BenchmarkAnalyzer:
    """Analyzer for comparing strategy performance to a benchmark.

    Calculates relative performance metrics including alpha, beta,
    tracking error, and capture ratios.

    Attributes:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        risk_free_rate: Daily risk-free rate for alpha calculation.

    Example:
        >>> strategy = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        >>> benchmark = pl.Series([0.005, -0.01, 0.01, -0.003, 0.015])
        >>> analyzer = BenchmarkAnalyzer(strategy, benchmark)
        >>> print(f"Beta: {analyzer.beta:.2f}")
    """

    def __init__(
        self,
        strategy_returns: pl.Series | np.ndarray | List[float],
        benchmark_returns: pl.Series | np.ndarray | List[float],
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
    ):
        """Initialize benchmark analyzer.

        Args:
            strategy_returns: Strategy daily returns.
            benchmark_returns: Benchmark daily returns.
            risk_free_rate: Daily risk-free rate. Default 0.
            annualization_factor: Periods per year. Default 252 (daily).
        """
        self.strategy_returns = self._to_numpy(strategy_returns)
        self.benchmark_returns = self._to_numpy(benchmark_returns)
        self.risk_free_rate = risk_free_rate
        self.ann_factor = annualization_factor

        # Align arrays if needed
        min_len = min(len(self.strategy_returns), len(self.benchmark_returns))
        self.strategy_returns = self.strategy_returns[:min_len]
        self.benchmark_returns = self.benchmark_returns[:min_len]

        # Calculate key metrics on init
        self._calculate_regression()

    def _to_numpy(self, data) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, pl.Series):
            return data.drop_nulls().to_numpy()
        elif isinstance(data, list):
            return np.array(data)
        return np.asarray(data)

    def _calculate_regression(self) -> None:
        """Calculate alpha and beta via OLS regression."""
        # Excess returns over risk-free rate
        excess_strategy = self.strategy_returns - self.risk_free_rate
        excess_benchmark = self.benchmark_returns - self.risk_free_rate

        # OLS regression: strategy_excess = alpha + beta * benchmark_excess + epsilon
        if len(excess_benchmark) < 2:
            self.alpha = 0.0
            self.beta = 1.0
            self.r_squared = 0.0
            return

        X = np.vstack([excess_benchmark, np.ones(len(excess_benchmark))]).T
        result = np.linalg.lstsq(X, excess_strategy, rcond=None)
        coefficients = result[0]

        self.beta = float(coefficients[0])
        daily_alpha = float(coefficients[1])
        self.alpha = daily_alpha * self.ann_factor  # Annualize

        # R-squared
        predicted = self.beta * excess_benchmark + daily_alpha
        ss_res = np.sum((excess_strategy - predicted) ** 2)
        ss_tot = np.sum((excess_strategy - np.mean(excess_strategy)) ** 2)
        self.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def tracking_error(self) -> float:
        """Calculate annualized tracking error.

        Tracking error is the standard deviation of the difference
        between strategy and benchmark returns.

        Returns:
            Annualized tracking error.
        """
        excess = self.strategy_returns - self.benchmark_returns
        te = float(np.std(excess, ddof=1)) * np.sqrt(self.ann_factor)
        return te

    @property
    def information_ratio(self) -> float:
        """Calculate information ratio.

        Information ratio = (Active Return) / (Tracking Error)
        Measures risk-adjusted relative performance.

        Returns:
            Information ratio (higher is better).
        """
        excess = self.strategy_returns - self.benchmark_returns
        active_return = float(np.mean(excess)) * self.ann_factor
        te = self.tracking_error
        return active_return / te if te > 0 else 0.0

    @property
    def correlation(self) -> float:
        """Calculate correlation with benchmark.

        Returns:
            Pearson correlation coefficient.
        """
        if len(self.strategy_returns) < 2:
            return 0.0
        return float(np.corrcoef(self.strategy_returns, self.benchmark_returns)[0, 1])

    def capture_ratios(self) -> Tuple[float, float, float]:
        """Calculate up and down capture ratios.

        Up capture measures how much of benchmark gains are captured.
        Down capture measures how much of benchmark losses are captured.

        Returns:
            Tuple of (up_capture, down_capture, capture_ratio).
        """
        # Up capture: strategy return when benchmark > 0
        up_mask = self.benchmark_returns > 0
        if np.any(up_mask):
            up_strategy = np.prod(1 + self.strategy_returns[up_mask]) - 1
            up_benchmark = np.prod(1 + self.benchmark_returns[up_mask]) - 1
            up_capture = (up_strategy / up_benchmark * 100) if up_benchmark != 0 else 100
        else:
            up_capture = 100.0

        # Down capture: strategy return when benchmark < 0
        down_mask = self.benchmark_returns < 0
        if np.any(down_mask):
            down_strategy = np.prod(1 + self.strategy_returns[down_mask]) - 1
            down_benchmark = np.prod(1 + self.benchmark_returns[down_mask]) - 1
            down_capture = (down_strategy / down_benchmark * 100) if down_benchmark != 0 else 100
        else:
            down_capture = 100.0

        # Capture ratio (higher is better: more upside, less downside)
        capture_ratio = up_capture / down_capture if down_capture != 0 else float('inf')

        return float(up_capture), float(down_capture), float(capture_ratio)

    def excess_return(self) -> float:
        """Calculate total excess return over benchmark.

        Returns:
            Cumulative strategy return minus benchmark return.
        """
        strategy_total = np.prod(1 + self.strategy_returns) - 1
        benchmark_total = np.prod(1 + self.benchmark_returns) - 1
        return float(strategy_total - benchmark_total)

    def calculate_all_metrics(self) -> BenchmarkMetrics:
        """Calculate all benchmark comparison metrics.

        Returns:
            BenchmarkMetrics dataclass with all metrics.

        Example:
            >>> metrics = analyzer.calculate_all_metrics()
            >>> print(f"IR: {metrics.information_ratio:.2f}")
        """
        up_cap, down_cap, cap_ratio = self.capture_ratios()

        return BenchmarkMetrics(
            alpha=self.alpha,
            beta=self.beta,
            r_squared=self.r_squared,
            tracking_error=self.tracking_error,
            information_ratio=self.information_ratio,
            up_capture=up_cap,
            down_capture=down_cap,
            capture_ratio=cap_ratio,
            excess_return=self.excess_return(),
            correlation=self.correlation,
        )

    def rolling_alpha_beta(
        self,
        window: int = 60,
    ) -> pl.DataFrame:
        """Calculate rolling alpha and beta.

        Args:
            window: Rolling window size. Default 60.

        Returns:
            DataFrame with rolling_alpha and rolling_beta columns.

        Example:
            >>> rolling_df = analyzer.rolling_alpha_beta(window=60)
        """
        n = len(self.strategy_returns)
        alphas = []
        betas = []

        for i in range(n):
            if i < window - 1:
                alphas.append(None)
                betas.append(None)
            else:
                strat_window = self.strategy_returns[i - window + 1 : i + 1]
                bench_window = self.benchmark_returns[i - window + 1 : i + 1]

                excess_s = strat_window - self.risk_free_rate
                excess_b = bench_window - self.risk_free_rate

                X = np.vstack([excess_b, np.ones(window)]).T
                result = np.linalg.lstsq(X, excess_s, rcond=None)
                coeffs = result[0]

                betas.append(float(coeffs[0]))
                alphas.append(float(coeffs[1]) * self.ann_factor)

        return pl.DataFrame({
            "rolling_alpha": alphas,
            "rolling_beta": betas,
        })

    def rolling_tracking_error(self, window: int = 60) -> pl.Series:
        """Calculate rolling tracking error.

        Args:
            window: Rolling window size. Default 60.

        Returns:
            Series with rolling tracking error values.
        """
        excess = self.strategy_returns - self.benchmark_returns
        excess_series = pl.Series(excess)
        rolling_te = excess_series.rolling_std(window_size=window) * np.sqrt(self.ann_factor)
        return rolling_te


def create_benchmark_from_prices(
    prices_df: pl.DataFrame,
    benchmark_type: str = "equal_weight",
    price_col: str = "close",
    asset_col: str = "asset_id",
    date_col: str = "date",
) -> pl.DataFrame:
    """Create benchmark returns from price data.

    Args:
        prices_df: DataFrame with price data.
        benchmark_type: "equal_weight" or "market_cap" (if available).
        price_col: Column name for prices. Default "close".
        asset_col: Column name for asset IDs. Default "asset_id".
        date_col: Column name for dates. Default "date".

    Returns:
        DataFrame with date and benchmark_return columns.

    Example:
        >>> benchmark_df = create_benchmark_from_prices(prices, "equal_weight")
    """
    # Calculate returns for each asset
    returns_df = prices_df.sort([date_col, asset_col]).with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1).over(asset_col) - 1)
        .alias("return")
    )

    # Aggregate based on benchmark type
    if benchmark_type == "equal_weight":
        benchmark = (
            returns_df
            .group_by(date_col)
            .agg(pl.col("return").mean().alias("benchmark_return"))
            .sort(date_col)
        )
    else:
        # For market_cap weighted, would need market cap data
        benchmark = (
            returns_df
            .group_by(date_col)
            .agg(pl.col("return").mean().alias("benchmark_return"))
            .sort(date_col)
        )

    return benchmark


def compare_to_benchmark(
    strategy_returns: pl.Series,
    benchmark_returns: pl.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Quick comparison of strategy vs benchmark.

    Convenience function for getting key comparison metrics.

    Args:
        strategy_returns: Strategy daily returns.
        benchmark_returns: Benchmark daily returns.
        risk_free_rate: Daily risk-free rate. Default 0.

    Returns:
        Dict with alpha, beta, info_ratio, tracking_error, excess_return.

    Example:
        >>> metrics = compare_to_benchmark(strategy, benchmark)
        >>> print(f"Alpha: {metrics['alpha']:.2%}")
    """
    analyzer = BenchmarkAnalyzer(
        strategy_returns,
        benchmark_returns,
        risk_free_rate=risk_free_rate,
    )
    metrics = analyzer.calculate_all_metrics()

    return {
        "alpha": metrics.alpha,
        "beta": metrics.beta,
        "information_ratio": metrics.information_ratio,
        "tracking_error": metrics.tracking_error,
        "excess_return": metrics.excess_return,
        "correlation": metrics.correlation,
        "r_squared": metrics.r_squared,
        "up_capture": metrics.up_capture,
        "down_capture": metrics.down_capture,
        "capture_ratio": metrics.capture_ratio,
    }
