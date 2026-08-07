"""Trade performance metrics for quantitative trading analysis.

This module provides key metrics for evaluating trading strategy performance
including win rate, profit factor, expectancy, and related statistics.

Functions:
    calculate_win_rate: Win rate percentage from trade results
    calculate_profit_factor: Ratio of gross profits to gross losses
    calculate_expectancy: Average profit/loss per trade
    calculate_r_multiples: Risk-adjusted returns per trade
    calculate_trade_metrics: Comprehensive trade performance metrics

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.trade_performance import calculate_trade_metrics
    >>> trades = pl.DataFrame({
    ...     "pnl": [100, -50, 150, -30, 200, -80],
    ...     "entry_price": [100, 105, 102, 108, 110, 112],
    ...     "exit_price": [110, 100, 117, 105, 130, 104],
    ...     "risk": [50, 50, 50, 50, 50, 50]
    ... })
    >>> metrics = calculate_trade_metrics(trades)
    >>> print(f"Win Rate: {metrics['win_rate']:.2%}")
    >>> print(f"Profit Factor: {metrics['profit_factor']:.2f}")
"""

from typing import Dict, Union

import polars as pl


def _pnl_series(pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl") -> pl.Series:
    """Normalise a PnL argument to a null-free Series.

    Every metric below accepts either a Series or a DataFrame plus a column
    name, and none of them should count a null trade in a denominator or
    compare one against zero.

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Series of non-null PnL values.
    """
    series = pnl[pnl_col] if isinstance(pnl, pl.DataFrame) else pnl
    return series.drop_nulls()


def calculate_win_rate(
    pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl"
) -> float:
    """Calculate win rate (percentage of profitable trades).

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Win rate as decimal (e.g., 0.65 = 65% win rate).

    Example:
        >>> pnl = pl.Series("pnl", [100, -50, 150, -30, 200])
        >>> win_rate = calculate_win_rate(pnl)
        >>> print(f"Win Rate: {win_rate:.2%}")
        Win Rate: 60.00%
    """
    pnl_series = _pnl_series(pnl, pnl_col)

    if len(pnl_series) == 0:
        return 0.0

    winning_trades = pnl_series.filter(pnl_series > 0)
    return float(len(winning_trades) / len(pnl_series))


def calculate_profit_factor(
    pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl"
) -> float:
    """Calculate profit factor (gross profits / gross losses).

    A profit factor > 1.0 indicates a profitable strategy.
    Typical good strategies have profit factors between 1.5 and 3.0.

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Profit factor, or inf when there are profits and no losses at all.
        Returns 0.0 when there is nothing to measure.

    Example:
        >>> pnl = pl.Series("pnl", [100, -50, 150, -30, 200])
        >>> pf = calculate_profit_factor(pnl)
        >>> print(f"Profit Factor: {pf:.2f}")
        Profit Factor: 5.62
    """
    pnl_series = _pnl_series(pnl, pnl_col)

    if len(pnl_series) == 0:
        return 0.0

    gross_profit = float(pnl_series.filter(pnl_series > 0).sum())
    gross_loss = abs(float(pnl_series.filter(pnl_series < 0).sum()))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_expectancy(
    pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl"
) -> float:
    """Calculate trade expectancy (average profit/loss per trade).

    Expectancy represents the average amount you can expect to win or
    lose per trade over the long run.

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Average expectancy per trade.

    Example:
        >>> pnl = pl.Series("pnl", [100, -50, 150, -30, 200])
        >>> expectancy = calculate_expectancy(pnl)
        >>> print(f"Expectancy: ${expectancy:.2f}")
        Expectancy: $74.00
    """
    pnl_series = _pnl_series(pnl, pnl_col)

    if len(pnl_series) == 0:
        return 0.0

    return float(pnl_series.mean())


def calculate_r_multiples(
    pnl: pl.Series, risk: pl.Series
) -> pl.Series:
    """Calculate R-multiples (risk-adjusted returns per trade).

    R-multiple is the profit/loss divided by the initial risk per trade.
    An R-multiple of 2.0 means you made 2x your initial risk.

    Args:
        pnl: Series of profit/loss values.
        risk: Series of initial risk amounts per trade.

    Returns:
        Series of R-multiples.

    Example:
        >>> pnl = pl.Series("pnl", [100, -50, 150])
        >>> risk = pl.Series("risk", [50, 50, 50])
        >>> r_mults = calculate_r_multiples(pnl, risk)
        >>> print(r_mults)
        shape: (3,)
        Series: '' [f64]
        [
            2.0
            -1.0
            3.0
        ]
    """
    if len(pnl) != len(risk):
        raise ValueError("pnl and risk must have the same length")

    # Avoid division by zero
    risk_safe = risk.replace(0, None)
    return pnl / risk_safe


def calculate_average_win_loss(
    pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl"
) -> Dict[str, float]:
    """Calculate average winning and losing trade amounts.

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Dictionary with:
            - avg_win: Average winning trade amount
            - avg_loss: Average losing trade amount (positive value)
            - win_loss_ratio: Ratio of avg_win to avg_loss

    Example:
        >>> pnl = pl.Series("pnl", [100, -50, 150, -30, 200])
        >>> metrics = calculate_average_win_loss(pnl)
        >>> print(f"Avg Win: ${metrics['avg_win']:.2f}")
        >>> print(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
    """
    pnl_series = _pnl_series(pnl, pnl_col)

    winning_trades = pnl_series.filter(pnl_series > 0)
    losing_trades = pnl_series.filter(pnl_series < 0)

    avg_win = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_loss = abs(float(losing_trades.mean())) if len(losing_trades) > 0 else 0.0

    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else float("inf")

    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
    }


def calculate_consecutive_streaks(
    pnl: Union[pl.Series, pl.DataFrame], pnl_col: str = "pnl"
) -> Dict[str, int]:
    """Calculate maximum consecutive winning and losing streaks.

    Args:
        pnl: Series or DataFrame containing profit/loss data.
        pnl_col: Column name if pnl is a DataFrame.

    Returns:
        Dictionary with:
            - max_win_streak: Maximum consecutive wins
            - max_loss_streak: Maximum consecutive losses

    Example:
        >>> pnl = pl.Series("pnl", [100, 50, -30, -20, 150, 200, -10])
        >>> streaks = calculate_consecutive_streaks(pnl)
        >>> print(f"Max Win Streak: {streaks['max_win_streak']}")
        Max Win Streak: 2
    """
    pnl_series = _pnl_series(pnl, pnl_col)

    if len(pnl_series) == 0:
        return {"max_win_streak": 0, "max_loss_streak": 0}

    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for value in pnl_series:
        if value > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif value < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0

    return {
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def calculate_trade_metrics(
    trades: pl.DataFrame,
    pnl_col: str = "pnl",
    risk_col: str = "risk",
) -> Dict[str, float]:
    """Calculate comprehensive trade performance metrics.

    Args:
        trades: DataFrame containing trade data with PnL and risk columns.
        pnl_col: Column name for profit/loss data.
        risk_col: Column name for risk data (optional, for R-multiples).

    Returns:
        Dictionary containing:
            - total_trades: Total number of trades
            - winning_trades: Number of profitable trades
            - losing_trades: Number of losing trades
            - win_rate: Win rate as decimal
            - profit_factor: Gross profit / gross loss ratio
            - expectancy: Average profit per trade
            - total_pnl: Total profit/loss
            - avg_win: Average winning trade
            - avg_loss: Average losing trade (positive value)
            - win_loss_ratio: Ratio of avg win to avg loss
            - max_win: Largest winning trade
            - max_loss: Largest losing trade (positive value)
            - max_win_streak: Maximum consecutive wins
            - max_loss_streak: Maximum consecutive losses
            - avg_r_multiple: Average R-multiple (if risk_col available)

    Example:
        >>> trades = pl.DataFrame({
        ...     "pnl": [100, -50, 150, -30, 200, -80],
        ...     "risk": [50, 50, 50, 50, 50, 50]
        ... })
        >>> metrics = calculate_trade_metrics(trades)
        >>> for key, value in metrics.items():
        ...     print(f"{key}: {value}")
    """
    pnl = _pnl_series(trades, pnl_col)

    if len(pnl) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    avg_metrics = calculate_average_win_loss(pnl)
    streaks = calculate_consecutive_streaks(pnl)

    metrics = {
        "total_trades": len(pnl),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": calculate_win_rate(pnl),
        "profit_factor": calculate_profit_factor(pnl),
        "expectancy": calculate_expectancy(pnl),
        "total_pnl": float(pnl.sum()),
        # Taken from the winners and losers themselves: pnl.max() on an
        # all-losing record is the *smallest loss*, not a winning trade.
        "max_win": float(winning_trades.max()) if len(winning_trades) > 0 else 0.0,
        "max_loss": abs(float(losing_trades.min())) if len(losing_trades) > 0 else 0.0,
        "avg_win": avg_metrics["avg_win"],
        "avg_loss": avg_metrics["avg_loss"],
        "win_loss_ratio": avg_metrics["win_loss_ratio"],
        "max_win_streak": streaks["max_win_streak"],
        "max_loss_streak": streaks["max_loss_streak"],
    }

    # Add R-multiple if risk column exists. Pair the columns before dropping
    # nulls so the two series stay aligned trade for trade.
    if risk_col in trades.columns:
        paired = trades.select(pnl_col, risk_col).drop_nulls()
        r_multiples = calculate_r_multiples(paired[pnl_col], paired[risk_col])
        avg_r = r_multiples.mean()
        metrics["avg_r_multiple"] = float(avg_r) if avg_r is not None else 0.0

    return metrics
