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
    Optimal position size: 8.12%
"""

from typing import Dict, List, Sequence, Tuple, Union

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
        Kelly %: 40.00%
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
        Optimal F as a decimal (0 to 1). Returns 0.0 when no f beats sitting
        out, which is the case for any strategy with a negative edge.

    Example:
        >>> trades = [100, -50, 150, -30, 200, -80]
        >>> opt_f = optimal_f(trades)
        >>> print(f"Optimal F: {opt_f:.4f}")
    """
    pnl = np.asarray(
        trades_pnl.to_numpy() if isinstance(trades_pnl, pl.Series) else trades_pnl,
        dtype=float,
    )

    if pnl.size == 0:
        return 0.0

    # The divisor is the worst *loss*; np.min() alone would pick the smallest
    # profit on an all-winning record and scale every HPR by garbage.
    largest_loss = -min(float(pnl.min()), 0.0)

    if largest_loss == 0:
        # Nothing ever lost, so TWR grows without bound in f. Cap at full risk.
        return 1.0 if float(pnl.max()) > 0 else 0.0

    # HPR = 1 + (f * trade / largest_loss), vectorised over the f grid.
    f_grid = np.arange(0.01, 1.01, 0.01)
    hpr = 1.0 + np.outer(f_grid, pnl) / largest_loss

    # Any non-positive HPR wipes the account out at that f.
    wiped = (hpr <= 0).any(axis=1)
    with np.errstate(over="ignore"):
        twr = np.prod(np.where(wiped[:, None], 1.0, hpr), axis=1)
    twr[wiped] = 0.0

    best = int(np.argmax(twr))

    # f = 0 has TWR 1.0, so only bet when some f actually beats not betting.
    return float(f_grid[best]) if twr[best] > 1.0 else 0.0


def _trade_win_stats(pnl: pl.Series) -> Tuple[float, float]:
    """Extract win rate and average-win / average-loss ratio from trade PnL.

    Degenerate records are reported rather than papered over, because the
    Kelly formulas reject a non-positive ratio:
        - no trades, or no winners -> (win_rate, 0.0)
        - winners but no losers     -> (win_rate, inf)

    Args:
        pnl: Series of per-trade profit/loss values.

    Returns:
        Tuple of (win_rate, win_loss_ratio).
    """
    pnl = pnl.drop_nulls()
    if pnl.is_empty():
        return 0.0, 0.0

    wins = pnl.filter(pnl > 0)
    losses = pnl.filter(pnl < 0)
    win_rate = len(wins) / len(pnl)

    if wins.is_empty():
        return win_rate, 0.0
    if losses.is_empty():
        return win_rate, float("inf")

    avg_win = float(wins.mean())
    avg_loss = abs(float(losses.mean()))

    return win_rate, avg_win / avg_loss


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
    win_rate, win_loss_ratio = _trade_win_stats(trades[pnl_col])

    if win_loss_ratio <= 0:
        # No winners (or no trades): the edge is negative, so size at zero.
        return 0.0

    if not np.isfinite(win_loss_ratio):
        # Nothing ever lost, so full Kelly is 1.0 and fractional Kelly is the
        # fraction itself.
        return float(fraction)

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
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
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
    pnl = trades[pnl_col].drop_nulls()
    win_rate, win_loss_ratio = _trade_win_stats(pnl)

    if win_loss_ratio <= 0:
        # No trades or no winners: every sizing recommendation is zero.
        metrics = {
            "win_rate": win_rate,
            "win_loss_ratio": 0.0,
            "full_kelly": 0.0,
            "optimal_f": optimal_f(pnl),
        }
        metrics.update({f"kelly_{frac}": 0.0 for frac in fractions})
        return metrics

    # Nothing ever lost: full Kelly saturates at 1.0 instead of dividing by zero.
    full_kelly = 1.0 if not np.isfinite(win_loss_ratio) else kelly_win_rate(win_rate, win_loss_ratio)

    metrics = {
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "full_kelly": max(0.0, full_kelly),  # Don't recommend negative positions
    }

    # Add fractional Kelly values
    for frac in fractions:
        metrics[f"kelly_{frac}"] = float(max(0.0, full_kelly) * frac)

    # Add Optimal F
    metrics["optimal_f"] = optimal_f(pnl)

    return metrics
