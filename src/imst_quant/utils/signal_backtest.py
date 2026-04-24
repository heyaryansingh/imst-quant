"""Quick signal backtesting framework for IMST-Quant.

This module provides a lightweight framework for rapid signal validation:
- Single-pass signal backtesting
- Signal performance metrics (hit rate, return per signal, etc.)
- Signal correlation and redundancy analysis
- Signal combination optimization

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.Series(np.random.randn(252) * 0.01)
    >>> signal = pd.Series(np.sign(np.random.randn(252)), index=returns.index)
    >>>
    >>> # Backtest a signal
    >>> result = backtest_signal(signal, returns)
    >>> print(f"Hit Rate: {result.hit_rate:.2%}")
    >>> print(f"Signal Sharpe: {result.sharpe_ratio:.2f}")
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class SignalBacktestResult:
    """Results from signal backtesting.

    Attributes:
        total_return: Total cumulative return.
        annualized_return: Annualized return.
        volatility: Annualized volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        max_drawdown: Maximum drawdown.
        hit_rate: Percentage of correct directional predictions.
        avg_win: Average winning trade return.
        avg_loss: Average losing trade return.
        profit_factor: Ratio of gross profits to gross losses.
        n_trades: Number of position changes.
        n_long: Number of long periods.
        n_short: Number of short periods.
        avg_holding_period: Average holding period in days.
        signal_correlation: Correlation between signal and forward returns.
    """

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    n_long: int
    n_short: int
    avg_holding_period: float
    signal_correlation: float


@dataclass
class SignalComparison:
    """Comparison of multiple signals.

    Attributes:
        signal_names: List of signal names.
        sharpe_ratios: Sharpe ratio for each signal.
        hit_rates: Hit rate for each signal.
        correlations: Correlation matrix between signals.
        information_ratios: Information ratio vs benchmark for each.
    """

    signal_names: list[str]
    sharpe_ratios: list[float]
    hit_rates: list[float]
    correlations: pd.DataFrame
    information_ratios: list[float]


def backtest_signal(
    signal: pd.Series,
    returns: pd.Series,
    transaction_cost: float = 0.0,
    lag: int = 1,
    trading_days: int = 252,
) -> SignalBacktestResult:
    """Backtest a trading signal against returns.

    Args:
        signal: Series of signals (-1, 0, 1) for short/flat/long.
        returns: Series of forward returns.
        transaction_cost: Cost per trade as decimal (e.g., 0.001 = 10bps).
        lag: Number of periods to lag signal (default 1 for tradeable signal).
        trading_days: Trading days per year for annualization.

    Returns:
        SignalBacktestResult with performance metrics.

    Example:
        >>> signal = pd.Series([1, 1, -1, -1, 1], index=pd.date_range("2024-01-01", periods=5))
        >>> returns = pd.Series([0.01, -0.02, -0.01, 0.02, 0.01], index=signal.index)
        >>> result = backtest_signal(signal, returns)
    """
    # Align signal and returns
    aligned = pd.DataFrame({"signal": signal, "returns": returns}).dropna()

    # Lag the signal (to avoid lookahead bias)
    lagged_signal = aligned["signal"].shift(lag)
    aligned = aligned.iloc[lag:]
    lagged_signal = lagged_signal.iloc[lag:]

    # Calculate strategy returns
    strategy_returns = lagged_signal * aligned["returns"]

    # Transaction costs
    trades = lagged_signal.diff().abs()
    costs = trades * transaction_cost
    strategy_returns = strategy_returns - costs

    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    n_periods = len(strategy_returns)
    ann_factor = trading_days / n_periods if n_periods > 0 else 0
    annualized_return = (1 + total_return) ** ann_factor - 1 if total_return > -1 else -1

    volatility = strategy_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

    # Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Hit rate (correct direction predictions)
    correct_direction = (lagged_signal * aligned["returns"]) > 0
    hit_rate = correct_direction.mean()

    # Win/Loss statistics
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Trade statistics
    n_trades = int(trades.sum() / 2)  # Each full trade is 2 position changes
    n_long = int((lagged_signal > 0).sum())
    n_short = int((lagged_signal < 0).sum())

    # Average holding period
    position_changes = (lagged_signal != lagged_signal.shift(1))
    holding_periods = []
    current_hold = 0
    for changed in position_changes:
        if changed:
            if current_hold > 0:
                holding_periods.append(current_hold)
            current_hold = 1
        else:
            current_hold += 1
    if current_hold > 0:
        holding_periods.append(current_hold)
    avg_holding = np.mean(holding_periods) if holding_periods else 0.0

    # Signal correlation
    signal_corr = lagged_signal.corr(aligned["returns"])

    return SignalBacktestResult(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        volatility=float(volatility),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
        hit_rate=float(hit_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        avg_holding_period=float(avg_holding),
        signal_correlation=float(signal_corr) if not np.isnan(signal_corr) else 0.0,
    )


def compare_signals(
    signals: dict[str, pd.Series],
    returns: pd.Series,
    transaction_cost: float = 0.0,
) -> SignalComparison:
    """Compare multiple signals against the same returns.

    Args:
        signals: Dictionary of signal name -> signal series.
        returns: Series of returns.
        transaction_cost: Cost per trade as decimal.

    Returns:
        SignalComparison with comparative metrics.
    """
    signal_names = list(signals.keys())
    sharpe_ratios = []
    hit_rates = []
    info_ratios = []

    # Backtest each signal
    for name, signal in signals.items():
        result = backtest_signal(signal, returns, transaction_cost)
        sharpe_ratios.append(result.sharpe_ratio)
        hit_rates.append(result.hit_rate)

        # Information ratio vs buy-and-hold
        benchmark_return = returns.mean() * 252
        benchmark_vol = returns.std() * np.sqrt(252)
        excess_return = result.annualized_return - benchmark_return
        tracking_error = (result.volatility - benchmark_vol) if result.volatility > 0 else 1.0
        info_ratio = excess_return / abs(tracking_error) if tracking_error != 0 else 0.0
        info_ratios.append(info_ratio)

    # Signal correlations
    signal_df = pd.DataFrame(signals)
    correlations = signal_df.corr()

    return SignalComparison(
        signal_names=signal_names,
        sharpe_ratios=sharpe_ratios,
        hit_rates=hit_rates,
        correlations=correlations,
        information_ratios=info_ratios,
    )


def signal_decay_analysis(
    signal: pd.Series,
    returns: pd.Series,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Analyze signal predictive power decay over different horizons.

    Args:
        signal: Series of signals.
        returns: Series of returns.
        max_lag: Maximum lag to test.

    Returns:
        DataFrame with IC (Information Coefficient) at each lag.
    """
    results = []
    for lag in range(1, max_lag + 1):
        # Forward returns over lag period
        fwd_returns = returns.rolling(window=lag).sum().shift(-lag)

        # Align
        aligned = pd.DataFrame({
            "signal": signal,
            "fwd_returns": fwd_returns
        }).dropna()

        if len(aligned) > 10:
            ic = aligned["signal"].corr(aligned["fwd_returns"])
            hit_rate = ((aligned["signal"] > 0) == (aligned["fwd_returns"] > 0)).mean()
        else:
            ic = np.nan
            hit_rate = np.nan

        results.append({
            "lag": lag,
            "information_coefficient": ic,
            "hit_rate": hit_rate,
        })

    return pd.DataFrame(results)


def turnover_analysis(signal: pd.Series) -> dict:
    """Analyze signal turnover characteristics.

    Args:
        signal: Series of signals.

    Returns:
        Dictionary with turnover metrics.
    """
    changes = signal.diff().abs()

    # Count position changes
    full_changes = (changes == 2).sum()  # -1 to 1 or vice versa
    partial_changes = (changes == 1).sum()  # 0 to +/-1 or vice versa

    # Daily turnover rate
    daily_turnover = changes.mean() / 2  # Divide by 2 since max change is 2

    # Average holding period
    position_changes = signal != signal.shift(1)
    holding_periods = []
    current = 0
    for changed in position_changes:
        if changed:
            if current > 0:
                holding_periods.append(current)
            current = 1
        else:
            current += 1

    return {
        "full_reversals": int(full_changes),
        "partial_changes": int(partial_changes),
        "daily_turnover_rate": float(daily_turnover),
        "avg_holding_period": float(np.mean(holding_periods)) if holding_periods else 0.0,
        "max_holding_period": int(max(holding_periods)) if holding_periods else 0,
        "min_holding_period": int(min(holding_periods)) if holding_periods else 0,
    }


def combine_signals(
    signals: dict[str, pd.Series],
    method: str = "equal",
    weights: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Combine multiple signals into a single signal.

    Args:
        signals: Dictionary of signal name -> signal series.
        method: Combination method - "equal", "weighted", "vote".
        weights: Weights for "weighted" method.

    Returns:
        Combined signal series.
    """
    signal_df = pd.DataFrame(signals)

    if method == "equal":
        combined = signal_df.mean(axis=1)
    elif method == "weighted":
        if weights is None:
            weights = {k: 1.0 / len(signals) for k in signals}
        combined = sum(signal_df[k] * w for k, w in weights.items())
    elif method == "vote":
        # Majority vote with sign
        combined = np.sign(signal_df.sum(axis=1))
    else:
        combined = signal_df.mean(axis=1)

    return combined


def signal_statistics(signal: pd.Series) -> dict:
    """Calculate descriptive statistics for a signal.

    Args:
        signal: Series of signals.

    Returns:
        Dictionary with signal statistics.
    """
    return {
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "min": float(signal.min()),
        "max": float(signal.max()),
        "pct_long": float((signal > 0).mean()),
        "pct_short": float((signal < 0).mean()),
        "pct_flat": float((signal == 0).mean()),
        "autocorrelation_1": float(signal.autocorr(lag=1)) if len(signal) > 1 else 0.0,
        "autocorrelation_5": float(signal.autocorr(lag=5)) if len(signal) > 5 else 0.0,
    }


def rolling_signal_performance(
    signal: pd.Series,
    returns: pd.Series,
    window: int = 63,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate rolling signal performance metrics.

    Args:
        signal: Series of signals.
        returns: Series of returns.
        window: Rolling window size.
        trading_days: Trading days per year.

    Returns:
        DataFrame with rolling performance metrics.
    """
    aligned = pd.DataFrame({"signal": signal.shift(1), "returns": returns}).dropna()
    strategy_returns = aligned["signal"] * aligned["returns"]

    rolling_return = strategy_returns.rolling(window=window).mean() * trading_days
    rolling_vol = strategy_returns.rolling(window=window).std() * np.sqrt(trading_days)
    rolling_sharpe = rolling_return / rolling_vol

    correct = (aligned["signal"] * aligned["returns"]) > 0
    rolling_hit_rate = correct.rolling(window=window).mean()

    return pd.DataFrame({
        "rolling_return": rolling_return,
        "rolling_volatility": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
        "rolling_hit_rate": rolling_hit_rate,
    })


def generate_random_signal(
    index: pd.DatetimeIndex,
    seed: Optional[int] = None,
) -> pd.Series:
    """Generate a random signal for benchmarking.

    Args:
        index: DatetimeIndex for the signal.
        seed: Random seed for reproducibility.

    Returns:
        Random signal series.
    """
    if seed is not None:
        np.random.seed(seed)

    signal = pd.Series(
        np.random.choice([-1, 0, 1], size=len(index)),
        index=index,
    )
    return signal


def bootstrap_signal(
    signal: pd.Series,
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 20,
) -> dict:
    """Bootstrap confidence intervals for signal performance.

    Args:
        signal: Series of signals.
        returns: Series of returns.
        n_simulations: Number of bootstrap simulations.
        block_size: Block size for block bootstrap.

    Returns:
        Dictionary with bootstrapped confidence intervals.
    """
    original_result = backtest_signal(signal, returns)

    sharpe_samples = []
    hit_rate_samples = []

    n = len(signal)
    n_blocks = n // block_size

    for _ in range(n_simulations):
        # Block bootstrap
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
        sampled_indices = []
        for bi in block_indices:
            start = bi * block_size
            end = min(start + block_size, n)
            sampled_indices.extend(range(start, end))

        sampled_indices = sampled_indices[:n]
        sampled_signal = signal.iloc[sampled_indices].reset_index(drop=True)
        sampled_returns = returns.iloc[sampled_indices].reset_index(drop=True)

        result = backtest_signal(sampled_signal, sampled_returns)
        sharpe_samples.append(result.sharpe_ratio)
        hit_rate_samples.append(result.hit_rate)

    return {
        "sharpe_mean": float(np.mean(sharpe_samples)),
        "sharpe_std": float(np.std(sharpe_samples)),
        "sharpe_ci_lower": float(np.percentile(sharpe_samples, 2.5)),
        "sharpe_ci_upper": float(np.percentile(sharpe_samples, 97.5)),
        "hit_rate_mean": float(np.mean(hit_rate_samples)),
        "hit_rate_ci_lower": float(np.percentile(hit_rate_samples, 2.5)),
        "hit_rate_ci_upper": float(np.percentile(hit_rate_samples, 97.5)),
        "original_sharpe": original_result.sharpe_ratio,
        "original_hit_rate": original_result.hit_rate,
    }
