"""Monte Carlo simulation for portfolio risk assessment and scenario analysis.

This module provides Monte Carlo simulation capabilities for estimating portfolio
risk metrics, generating return distributions, and analyzing worst-case scenarios.

Features:
- Historical simulation using bootstrap resampling
- Parametric simulation assuming normal/t-distributions
- Geometric Brownian Motion (GBM) price path generation
- Confidence interval estimation for returns
- Value-at-Risk (VaR) and Expected Shortfall (ES) via simulation
- Stress testing with tail scenarios

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.monte_carlo import MonteCarloSimulator
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
    >>> simulator = MonteCarloSimulator(returns, n_simulations=10000)
    >>> var_95 = simulator.var_simulation(confidence=0.95)
    >>> print(f"VaR (95%): {var_95:.2%}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results.

    Attributes:
        simulated_returns: Array of shape (n_simulations, horizon) with simulated paths.
        terminal_values: Final portfolio values for each simulation.
        percentiles: Dict mapping percentile levels to values.
        var: Value-at-Risk at specified confidence level.
        expected_shortfall: Expected Shortfall (CVaR) at specified confidence.
        mean_return: Average simulated return.
        std_return: Standard deviation of simulated returns.
    """

    simulated_returns: np.ndarray
    terminal_values: np.ndarray
    percentiles: Dict[int, float]
    var: float
    expected_shortfall: float
    mean_return: float
    std_return: float


class MonteCarloSimulator:
    """Monte Carlo simulator for portfolio risk analysis.

    Performs Monte Carlo simulations using historical or parametric methods
    to estimate risk metrics and generate return distributions.

    Attributes:
        returns: Historical return series used for simulation.
        n_simulations: Number of simulation paths to generate.
        seed: Random seed for reproducibility.

    Example:
        >>> returns = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02, 0.01])
        >>> sim = MonteCarloSimulator(returns, n_simulations=5000, seed=42)
        >>> result = sim.run_historical_simulation(horizon=252)
        >>> print(f"Expected annual return: {result.mean_return:.2%}")
    """

    def __init__(
        self,
        returns: pl.Series | np.ndarray | List[float],
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ):
        """Initialize Monte Carlo simulator.

        Args:
            returns: Historical return series (daily returns recommended).
            n_simulations: Number of simulation paths. Default 10000.
            seed: Random seed for reproducibility. Default None.
        """
        if isinstance(returns, pl.Series):
            self.returns = returns.drop_nulls().to_numpy()
        elif isinstance(returns, list):
            self.returns = np.array(returns)
        else:
            self.returns = returns

        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Calculate historical statistics
        self._mean = float(np.mean(self.returns))
        self._std = float(np.std(self.returns, ddof=1))

    def run_historical_simulation(
        self,
        horizon: int = 252,
        initial_value: float = 1.0,
    ) -> SimulationResult:
        """Run historical bootstrap simulation.

        Generates simulated return paths by randomly sampling (with replacement)
        from historical returns. This preserves the empirical distribution
        including fat tails and skewness.

        Args:
            horizon: Number of periods to simulate. Default 252 (1 year daily).
            initial_value: Starting portfolio value. Default 1.0.

        Returns:
            SimulationResult with simulated paths and risk metrics.

        Example:
            >>> result = simulator.run_historical_simulation(horizon=252)
            >>> print(f"5th percentile: {result.percentiles[5]:.2%}")
        """
        # Bootstrap sample returns
        simulated_returns = self._rng.choice(
            self.returns,
            size=(self.n_simulations, horizon),
            replace=True,
        )

        # Calculate terminal values via compound returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
        terminal_values = initial_value * cumulative_returns[:, -1]

        return self._create_result(simulated_returns, terminal_values)

    def run_parametric_simulation(
        self,
        horizon: int = 252,
        initial_value: float = 1.0,
        distribution: str = "normal",
        df: int = 5,
    ) -> SimulationResult:
        """Run parametric Monte Carlo simulation.

        Generates simulated return paths assuming a parametric distribution
        (normal or Student's t) with parameters estimated from historical data.

        Args:
            horizon: Number of periods to simulate. Default 252.
            initial_value: Starting portfolio value. Default 1.0.
            distribution: Distribution type, "normal" or "t". Default "normal".
            df: Degrees of freedom for t-distribution. Default 5.

        Returns:
            SimulationResult with simulated paths and risk metrics.

        Example:
            >>> result = simulator.run_parametric_simulation(
            ...     horizon=252, distribution="t", df=5
            ... )
        """
        if distribution == "normal":
            simulated_returns = self._rng.normal(
                loc=self._mean,
                scale=self._std,
                size=(self.n_simulations, horizon),
            )
        elif distribution == "t":
            # Scale t-distribution to match historical volatility
            t_samples = self._rng.standard_t(df=df, size=(self.n_simulations, horizon))
            scale_factor = self._std * np.sqrt((df - 2) / df) if df > 2 else self._std
            simulated_returns = self._mean + scale_factor * t_samples
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
        terminal_values = initial_value * cumulative_returns[:, -1]

        return self._create_result(simulated_returns, terminal_values)

    def run_gbm_simulation(
        self,
        horizon: int = 252,
        initial_price: float = 100.0,
        dt: float = 1 / 252,
    ) -> SimulationResult:
        """Run Geometric Brownian Motion price simulation.

        Simulates price paths using the GBM model:
        dS = mu*S*dt + sigma*S*dW

        Useful for option pricing and long-term projections where
        log-normality is assumed.

        Args:
            horizon: Number of time steps. Default 252.
            initial_price: Starting price. Default 100.0.
            dt: Time step size. Default 1/252 (daily).

        Returns:
            SimulationResult with simulated price paths.

        Example:
            >>> result = simulator.run_gbm_simulation(
            ...     horizon=252, initial_price=100.0
            ... )
            >>> final_prices = result.terminal_values
        """
        # Annualized parameters
        mu = self._mean / dt  # Drift
        sigma = self._std / np.sqrt(dt)  # Volatility

        # Generate Brownian increments
        dW = self._rng.normal(0, np.sqrt(dt), size=(self.n_simulations, horizon))

        # Build price paths
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        simulated_returns = np.exp(log_returns) - 1

        # Calculate prices
        cumulative = np.cumprod(1 + simulated_returns, axis=1)
        terminal_values = initial_price * cumulative[:, -1]

        return self._create_result(simulated_returns, terminal_values, is_price=True)

    def var_simulation(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        method: str = "historical",
    ) -> float:
        """Calculate Value-at-Risk via Monte Carlo simulation.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR). Default 0.95.
            horizon: Holding period in days. Default 1.
            method: Simulation method, "historical" or "parametric". Default "historical".

        Returns:
            VaR as a positive number representing potential loss.

        Example:
            >>> var_95 = simulator.var_simulation(confidence=0.95, horizon=10)
            >>> print(f"10-day 95% VaR: {var_95:.2%}")
        """
        if method == "historical":
            result = self.run_historical_simulation(horizon=horizon)
        else:
            result = self.run_parametric_simulation(horizon=horizon)

        # VaR is the negative of the (1-confidence) percentile of returns
        total_returns = result.terminal_values - 1.0  # Convert to returns
        return -float(np.percentile(total_returns, (1 - confidence) * 100))

    def expected_shortfall_simulation(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        method: str = "historical",
    ) -> float:
        """Calculate Expected Shortfall (CVaR) via Monte Carlo simulation.

        Expected Shortfall is the average loss given that losses exceed VaR.
        It's a coherent risk measure that captures tail risk.

        Args:
            confidence: Confidence level. Default 0.95.
            horizon: Holding period in days. Default 1.
            method: Simulation method. Default "historical".

        Returns:
            Expected Shortfall as a positive number.

        Example:
            >>> es_95 = simulator.expected_shortfall_simulation(confidence=0.95)
            >>> print(f"95% ES: {es_95:.2%}")
        """
        if method == "historical":
            result = self.run_historical_simulation(horizon=horizon)
        else:
            result = self.run_parametric_simulation(horizon=horizon)

        total_returns = result.terminal_values - 1.0
        var_threshold = np.percentile(total_returns, (1 - confidence) * 100)
        tail_losses = total_returns[total_returns <= var_threshold]

        return -float(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0

    def stress_test(
        self,
        scenarios: Dict[str, float],
        horizon: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Run stress tests with specified return scenarios.

        Simulates portfolio behavior under stressed conditions by
        shifting the return distribution.

        Args:
            scenarios: Dict mapping scenario names to return shocks.
                E.g., {"crash": -0.10, "correction": -0.05}
            horizon: Holding period. Default 1.

        Returns:
            Dict mapping scenario names to results with var, es, and mean_loss.

        Example:
            >>> scenarios = {"market_crash": -0.15, "correction": -0.05}
            >>> stress_results = simulator.stress_test(scenarios)
            >>> print(stress_results["market_crash"]["mean_loss"])
        """
        results = {}

        for name, shock in scenarios.items():
            # Shift historical returns by the shock
            stressed_returns = self.returns + shock

            # Run simulation with stressed returns
            stressed_sim = self._rng.choice(
                stressed_returns,
                size=(self.n_simulations, horizon),
                replace=True,
            )

            terminal_returns = np.prod(1 + stressed_sim, axis=1) - 1

            var_95 = -float(np.percentile(terminal_returns, 5))
            es_95 = -float(np.mean(terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)]))
            mean_loss = -float(np.mean(terminal_returns))

            results[name] = {
                "var_95": var_95,
                "es_95": es_95,
                "mean_loss": mean_loss,
                "worst_case": -float(np.min(terminal_returns)),
                "prob_loss": float(np.mean(terminal_returns < 0)),
            }

        return results

    def confidence_interval(
        self,
        horizon: int = 252,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for portfolio returns.

        Args:
            horizon: Investment horizon in periods. Default 252.
            confidence: Confidence level. Default 0.95.

        Returns:
            Tuple of (lower_bound, upper_bound) for returns.

        Example:
            >>> lower, upper = simulator.confidence_interval(horizon=252, confidence=0.95)
            >>> print(f"95% CI: [{lower:.2%}, {upper:.2%}]")
        """
        result = self.run_historical_simulation(horizon=horizon)
        total_returns = result.terminal_values - 1.0

        alpha = (1 - confidence) / 2
        lower = float(np.percentile(total_returns, alpha * 100))
        upper = float(np.percentile(total_returns, (1 - alpha) * 100))

        return lower, upper

    def _create_result(
        self,
        simulated_returns: np.ndarray,
        terminal_values: np.ndarray,
        is_price: bool = False,
    ) -> SimulationResult:
        """Create SimulationResult from simulation outputs."""
        if is_price:
            total_returns = terminal_values / terminal_values.mean() - 1
        else:
            total_returns = terminal_values - 1.0

        percentiles = {
            p: float(np.percentile(total_returns, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        var_95 = -float(np.percentile(total_returns, 5))
        tail_mask = total_returns <= np.percentile(total_returns, 5)
        es_95 = -float(np.mean(total_returns[tail_mask])) if tail_mask.any() else var_95

        return SimulationResult(
            simulated_returns=simulated_returns,
            terminal_values=terminal_values,
            percentiles=percentiles,
            var=var_95,
            expected_shortfall=es_95,
            mean_return=float(np.mean(total_returns)),
            std_return=float(np.std(total_returns)),
        )


def portfolio_monte_carlo(
    returns_df: pl.DataFrame,
    weights: Dict[str, float],
    n_simulations: int = 10000,
    horizon: int = 252,
    asset_col: str = "asset_id",
    return_col: str = "return_1d",
    seed: Optional[int] = None,
) -> SimulationResult:
    """Run Monte Carlo simulation for a weighted portfolio.

    Simulates correlated asset returns using historical covariance
    and generates portfolio return paths.

    Args:
        returns_df: DataFrame with asset returns.
        weights: Dict mapping asset IDs to portfolio weights.
        n_simulations: Number of simulation paths. Default 10000.
        horizon: Simulation horizon in periods. Default 252.
        asset_col: Column name for asset identifier. Default "asset_id".
        return_col: Column name for returns. Default "return_1d".
        seed: Random seed. Default None.

    Returns:
        SimulationResult for the portfolio.

    Example:
        >>> weights = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        >>> result = portfolio_monte_carlo(df, weights, horizon=252)
    """
    rng = np.random.default_rng(seed)

    # Build returns matrix
    assets = list(weights.keys())
    returns_list = []

    for asset in assets:
        asset_df = returns_df.filter(pl.col(asset_col) == asset)
        if asset_df.height > 0:
            returns_list.append(asset_df[return_col].drop_nulls().to_numpy())
        else:
            returns_list.append(np.array([0.0]))

    # Find minimum length and truncate
    min_len = min(len(r) for r in returns_list)
    returns_matrix = np.column_stack([r[:min_len] for r in returns_list])

    # Calculate covariance and means
    means = np.mean(returns_matrix, axis=0)
    cov = np.cov(returns_matrix.T)

    # Ensure covariance is positive semi-definite
    if len(assets) > 1:
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 0):
            cov += np.eye(len(assets)) * (-min(eigvals) + 1e-8)

    # Generate correlated returns
    weight_arr = np.array([weights.get(a, 0) for a in assets])

    simulated_asset_returns = rng.multivariate_normal(
        means, cov, size=(n_simulations, horizon)
    )

    # Calculate portfolio returns (weighted sum)
    portfolio_returns = np.sum(simulated_asset_returns * weight_arr, axis=2)

    # Calculate terminal values
    terminal_values = np.prod(1 + portfolio_returns, axis=1)

    # Create result
    total_returns = terminal_values - 1.0
    percentiles = {
        p: float(np.percentile(total_returns, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

    var_95 = -float(np.percentile(total_returns, 5))
    tail_mask = total_returns <= np.percentile(total_returns, 5)
    es_95 = -float(np.mean(total_returns[tail_mask])) if tail_mask.any() else var_95

    return SimulationResult(
        simulated_returns=portfolio_returns,
        terminal_values=terminal_values,
        percentiles=percentiles,
        var=var_95,
        expected_shortfall=es_95,
        mean_return=float(np.mean(total_returns)),
        std_return=float(np.std(total_returns)),
    )
