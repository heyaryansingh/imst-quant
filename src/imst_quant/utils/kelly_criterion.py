"""Kelly Criterion position sizing with advanced variants.

This module implements the Kelly Criterion for optimal position sizing,
including fractional Kelly, Kelly with win rate, and practical adjustments.

The Kelly Criterion is a formula that determines the optimal bet size to
maximize long-term capital growth while minimizing risk of ruin.

Functions:
    kelly_formula: Classic Kelly Criterion calculation
    kelly_win_rate: Kelly based on win rate and win/loss ratio
    fractional_kelly: Conservative fractional Kelly (recommended)
    kelly_from_sharpe: Estimate Kelly fraction from Sharpe ratio
    optimal_f: Ralph Vince's Optimal F calculation
    kelly_portfolio: Multi-asset Kelly allocation

Example:
    >>> from imst_quant.utils.kelly_criterion import fractional_kelly
    >>> # Win rate = 55%, avg win = $100, avg loss = $50
    >>> kelly_pct = fractional_kelly(win_rate=0.55, win_loss_ratio=2.0, fraction=0.25)
    >>> print(f"Optimal position size: {kelly_pct:.2%}")
    Optimal position size: 2.50%
"""

from typing import Dict, List, Union

import numpy as np
import polars as pl


def kelly_formula(
    probability: float,
    win_amount: float,
    loss_amount: float,
) -> float:
    """Calculate Kelly Criterion percentage for a single bet.

    Classic Kelly formula: K = (p*W - q*L) / (W*L)
    where:
        p = probability of winning
        q = probability of losing (1-p)
        W = win amount per dollar risked
        L = loss amount per dollar risked (usually 1)

    Args:
        probability: Probability of winning (0 to 1).
        win_amount: Expected win amount per unit risked.
        loss_amount: Expected loss amount per unit risked.

    Returns:
        Kelly percentage (can be negative if expected value is negative).

    Example:
        >>> # 60% chance to win $2 for every $1 risked
        >>> kelly = kelly_formula(probability=0.6, win_amount=2.0, loss_amount=1.0)
        >>> print(f"Kelly %: {kelly:.2%}")
        Kelly %: 10.00%
    """
    if not (0 <= probability <= 1):
        raise ValueError("Probability must be between 0 and 1")

    if loss_amount <= 0 or win_amount <= 0:
        raise ValueError("Win and loss amounts must be positive")

    q = 1 - probability
    kelly = (probability * win_amount - q * loss_amount) / (win_amount * loss_amount)

    return float(kelly)


def kelly_win_rate(
    win_rate: float,
    win_loss_ratio: float,
) -> float:
    """Calculate Kelly percentage from win rate and win/loss ratio.

    Simplified Kelly: K = W - (1-W)/R
    where:
        W = win rate
        R = ratio of average win to average loss

    Args:
        win_rate: Percentage of winning trades (0 to 1).
        win_loss_ratio: Ratio of average win to average loss.

    Returns:
        Kelly percentage.

    Example:
        >>> # 55% win rate, average win is 2x average loss
        >>> kelly = kelly_win_rate(win_rate=0.55, win_loss_ratio=2.0)
        >>> print(f"Kelly %: {kelly:.2%}")
        Kelly %: 32.50%
    """
    if not (0 <= win_rate <= 1):
        raise ValueError("Win rate must be between 0 and 1")

    if win_loss_ratio <= 0:
        raise ValueError("Win/loss ratio must be positive")

    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    return float(kelly)


def fractional_kelly(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
) -> float:
    """Calculate fractional Kelly (conservative Kelly).

    Full Kelly can lead to large drawdowns in practice due to:
    - Estimation errors in win rate and win/loss ratio
    - Non-stationary markets
    - Psychological difficulty of large positions

    Fractional Kelly uses a fraction (typically 1/4 to 1/2) of the full Kelly.

    Args:
        win_rate: Percentage of winning trades (0 to 1).
        win_loss_ratio: Ratio of average win to average loss.
        fraction: Fraction of Kelly to use (default: 0.25 for quarter Kelly).

    Returns:
        Fractional Kelly percentage.

    Example:
        >>> # Quarter Kelly with 55% win rate, 2:1 win/loss
        >>> kelly = fractional_kelly(win_rate=0.55, win_loss_ratio=2.0, fraction=0.25)
        >>> print(f"Position size: {kelly:.2%}")
        Position size: 8.12%
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1")

    full_kelly = kelly_win_rate(win_rate, win_loss_ratio)

    # Don't allow negative positions
    if full_kelly < 0:
        return 0.0

    return float(full_kelly * fraction)


def kelly_from_sharpe(
    sharpe_ratio: float,
    fraction: float = 0.25,
) -> float:
    """Estimate Kelly fraction from Sharpe ratio.

    For normally distributed returns:
    Kelly ≈ Sharpe / (Excess Kurtosis + Volatility²)

    Simplified approximation: Kelly ≈ Sharpe² / 2

    This is useful when you have a Sharpe ratio but not detailed
    win/loss statistics.

    Args:
        sharpe_ratio: Sharpe ratio of the strategy.
        fraction: Fractional Kelly multiplier (default: 0.25).

    Returns:
        Estimated Kelly percentage.

    Example:
        >>> # Sharpe ratio of 1.5, use quarter Kelly
        >>> kelly = kelly_from_sharpe(sharpe_ratio=1.5, fraction=0.25)
        >>> print(f"Position size: {kelly:.2%}")
    """
    if sharpe_ratio < 0:
        return 0.0

    # Simplified Kelly approximation from Sharpe
    full_kelly = (sharpe_ratio ** 2) / 2

    return float(full_kelly * fraction)


def optimal_f(
    trades_pnl: Union[pl.Series, List[float], np.ndarray],
    initial_capital: float = 10000,
) -> float:
    """Calculate Ralph Vince's Optimal F.

    Optimal F is similar to Kelly but handles actual trade PnL directly
    without requiring win rate and win/loss ratio.

    It finds the position sizing fraction that maximizes terminal wealth factor.

    Args:
        trades_pnl: Series or list of trade profit/loss values.
        initial_capital: Initial capital amount (default: 10000).

    Returns:
        Optimal F as a decimal (0 to 1).

    Example:
        >>> trades = [100, -50, 150, -30, 200, -80]
        >>> opt_f = optimal_f(trades)
        >>> print(f"Optimal F: {opt_f:.4f}")
    """
    if isinstance(trades_pnl, pl.Series):
        pnl = trades_pnl.to_numpy()
    elif isinstance(trades_pnl, list):
        pnl = np.array(trades_pnl)
    else:
        pnl = trades_pnl

    if len(pnl) == 0:
        return 0.0

    # Find the largest loss (use as divisor)
    largest_loss = abs(np.min(pnl))

    if largest_loss == 0:
        return 0.0

    # Test different f values and find the one that maximizes TWR
    best_f = 0.0
    best_twr = 0.0

    for f in np.arange(0.01, 1.01, 0.01):
        twr = 1.0
        for trade in pnl:
            # HPR = 1 + (f * trade / largest_loss)
            hpr = 1 + (f * trade / largest_loss)
            twr *= hpr

            # Stop if we'd blow up
            if twr <= 0:
                twr = 0.0
                break

        if twr > best_twr:
            best_twr = twr
            best_f = f

    return float(best_f)


def kelly_from_trades(
    trades: pl.DataFrame,
    pnl_col: str = "pnl",
    fraction: float = 0.25,
) -> float:
    """Calculate Kelly percentage from trade history.

    Analyzes trade PnL to extract win rate and win/loss ratio,
    then calculates fractional Kelly.

    Args:
        trades: DataFrame with trade PnL data.
        pnl_col: Column name for PnL data.
        fraction: Fractional Kelly multiplier (default: 0.25).

    Returns:
        Fractional Kelly percentage.

    Example:
        >>> import polars as pl
        >>> trades = pl.DataFrame({"pnl": [100, -50, 150, -30, 200, -80]})
        >>> kelly = kelly_from_trades(trades, fraction=0.25)
        >>> print(f"Position size: {kelly:.2%}")
    """
    pnl = trades[pnl_col]

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    if len(pnl) == 0:
        return 0.0

    win_rate = len(winning_trades) / len(pnl)

    avg_win = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_loss = abs(float(losing_trades.mean())) if len(losing_trades) > 0 else 1.0

    if avg_loss == 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss

    return fractional_kelly(win_rate, win_loss_ratio, fraction)


def kelly_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate multi-asset Kelly optimal portfolio weights.

    For multiple assets, Kelly criterion becomes:
    K = Σ^(-1) * μ

    where Σ is the covariance matrix and μ is the expected returns vector.

    Args:
        expected_returns: Array of expected returns for each asset.
        covariance_matrix: Covariance matrix of asset returns.

    Returns:
        Array of Kelly optimal weights (may sum to > 1 if leverage).

    Example:
        >>> expected_returns = np.array([0.10, 0.08, 0.12])
        >>> cov_matrix = np.array([
        ...     [0.04, 0.01, 0.02],
        ...     [0.01, 0.03, 0.015],
        ...     [0.02, 0.015, 0.05]
        ... ])
        >>> weights = kelly_portfolio(expected_returns, cov_matrix)
        >>> print(f"Kelly weights: {weights}")
    """
    if expected_returns.shape[0] != covariance_matrix.shape[0]:
        raise ValueError("Dimension mismatch between returns and covariance matrix")

    try:
        # K = Σ^(-1) * μ
        inv_cov = np.linalg.inv(covariance_matrix)
        kelly_weights = inv_cov @ expected_returns

        return kelly_weights

    except np.linalg.LinAlgError:
        # Singular matrix, return equal weights
        n = len(expected_returns)
        return np.ones(n) / n


def calculate_kelly_metrics(
    trades: pl.DataFrame,
    pnl_col: str = "pnl",
    fractions: List[float] = [0.25, 0.50, 1.0],
) -> Dict[str, float]:
    """Calculate comprehensive Kelly-based position sizing metrics.

    Args:
        trades: DataFrame with trade PnL data.
        pnl_col: Column name for PnL data.
        fractions: List of Kelly fractions to calculate (default: [0.25, 0.5, 1.0]).

    Returns:
        Dictionary containing:
            - win_rate: Win rate from trades
            - win_loss_ratio: Average win / average loss
            - full_kelly: Full Kelly percentage
            - kelly_<fraction>: Kelly at each fraction (e.g., kelly_0.25)
            - optimal_f: Ralph Vince's Optimal F

    Example:
        >>> trades = pl.DataFrame({"pnl": [100, -50, 150, -30, 200, -80]})
        >>> metrics = calculate_kelly_metrics(trades)
        >>> for key, value in metrics.items():
        ...     print(f"{key}: {value:.4f}")
    """
    pnl = trades[pnl_col]

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    if len(pnl) == 0:
        base_metrics = {
            "win_rate": 0.0,
            "win_loss_ratio": 0.0,
            "full_kelly": 0.0,
            "optimal_f": 0.0,
        }
        for frac in fractions:
            base_metrics[f"kelly_{frac}"] = 0.0
        return base_metrics

    win_rate = len(winning_trades) / len(pnl)
    avg_win = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_loss = abs(float(losing_trades.mean())) if len(losing_trades) > 0 else 1.0

    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    full_kelly = kelly_win_rate(win_rate, win_loss_ratio)

    metrics = {
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "full_kelly": max(0.0, full_kelly),  # Don't recommend negative positions
    }

    # Add fractional Kelly values
    for frac in fractions:
        metrics[f"kelly_{frac}"] = fractional_kelly(win_rate, win_loss_ratio, frac)

    # Add Optimal F
    metrics["optimal_f"] = optimal_f(pnl)

    return metrics
