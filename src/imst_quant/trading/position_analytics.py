"""Position-level analytics and performance tracking.

Provides detailed analysis of individual positions including:
- Entry/exit timing quality
- Hold period statistics
- Profit factor by position size
- Win/loss streak detection
- Position size vs. performance correlation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class Position:
    """Individual position data."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    commission: float = 0.0
    tags: Optional[List[str]] = None


@dataclass
class PositionMetrics:
    """Aggregated position-level metrics."""
    total_positions: int
    winning_positions: int
    losing_positions: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_hold_time_hours: float
    best_position: float
    worst_position: float
    avg_position_pnl: float
    median_position_pnl: float
    position_pnl_std: float
    sharpe_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int


def analyze_positions(positions: List[Position]) -> PositionMetrics:
    """Compute comprehensive position-level analytics.

    Args:
        positions: List of Position objects

    Returns:
        PositionMetrics with aggregated statistics

    Example:
        >>> positions = [Position(...), Position(...)]
        >>> metrics = analyze_positions(positions)
        >>> print(f"Win rate: {metrics.win_rate:.1%}")
    """
    if not positions:
        raise ValueError("No positions provided")

    total = len(positions)
    wins = [p for p in positions if p.pnl > 0]
    losses = [p for p in positions if p.pnl < 0]

    winning_count = len(wins)
    losing_count = len(losses)
    win_rate = winning_count / total if total > 0 else 0.0

    avg_win = np.mean([p.pnl for p in wins]) if wins else 0.0
    avg_loss = np.mean([p.pnl for p in losses]) if losses else 0.0

    total_wins = sum(p.pnl for p in wins)
    total_losses = abs(sum(p.pnl for p in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    hold_times = [(p.exit_date - p.entry_date).total_seconds() / 3600 for p in positions]
    avg_hold_time = np.mean(hold_times)

    pnls = [p.pnl for p in positions]
    best = max(pnls)
    worst = min(pnls)
    avg_pnl = np.mean(pnls)
    median_pnl = np.median(pnls)
    std_pnl = np.std(pnls)

    # Sharpe ratio (assuming daily positions)
    sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0

    # Streak detection
    max_win_streak = _max_streak(positions, True)
    max_loss_streak = _max_streak(positions, False)

    return PositionMetrics(
        total_positions=total,
        winning_positions=winning_count,
        losing_positions=losing_count,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_hold_time_hours=avg_hold_time,
        best_position=best,
        worst_position=worst,
        avg_position_pnl=avg_pnl,
        median_position_pnl=median_pnl,
        position_pnl_std=std_pnl,
        sharpe_ratio=sharpe,
        max_consecutive_wins=max_win_streak,
        max_consecutive_losses=max_loss_streak,
    )


def _max_streak(positions: List[Position], winning: bool) -> int:
    """Find maximum consecutive win or loss streak."""
    max_streak = 0
    current_streak = 0

    for pos in positions:
        is_win = pos.pnl > 0
        if is_win == winning:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def position_size_correlation(positions: List[Position]) -> Tuple[float, float]:
    """Analyze correlation between position size and PnL.

    Returns:
        (correlation_coefficient, p_value)

    A positive correlation means larger positions tend to be more profitable.
    """
    if len(positions) < 3:
        return 0.0, 1.0

    sizes = [abs(p.quantity * p.entry_price) for p in positions]
    pnls = [p.pnl for p in positions]

    corr, p_value = stats.pearsonr(sizes, pnls)
    return corr, p_value


def profit_factor_by_size_quintile(positions: List[Position]) -> Dict[str, float]:
    """Compute profit factor for each position size quintile.

    Returns:
        Dictionary with quintile labels (Q1-Q5) and profit factors
    """
    if len(positions) < 5:
        return {"all": _compute_profit_factor(positions)}

    # Calculate position sizes
    df = pd.DataFrame([
        {
            "size": abs(p.quantity * p.entry_price),
            "pnl": p.pnl,
        }
        for p in positions
    ])

    df["quintile"] = pd.qcut(df["size"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    results = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_positions = [p for p, qval in zip(positions, df["quintile"]) if qval == q]
        results[q] = _compute_profit_factor(q_positions)

    return results


def _compute_profit_factor(positions: List[Position]) -> float:
    """Helper to compute profit factor for a list of positions."""
    wins = [p.pnl for p in positions if p.pnl > 0]
    losses = [abs(p.pnl) for p in positions if p.pnl < 0]

    total_wins = sum(wins)
    total_losses = sum(losses)

    return total_wins / total_losses if total_losses > 0 else float('inf')


def hold_time_performance(positions: List[Position]) -> pd.DataFrame:
    """Analyze PnL by hold time buckets.

    Returns:
        DataFrame with columns: bucket, count, avg_pnl, win_rate, profit_factor
    """
    if not positions:
        return pd.DataFrame()

    # Calculate hold times in hours
    data = []
    for p in positions:
        hold_hours = (p.exit_date - p.entry_date).total_seconds() / 3600
        data.append({"hold_hours": hold_hours, "pnl": p.pnl})

    df = pd.DataFrame(data)

    # Define buckets
    bins = [0, 4, 12, 24, 48, 168, float('inf')]
    labels = ["<4h", "4-12h", "12-24h", "1-2d", "2-7d", ">7d"]
    df["bucket"] = pd.cut(df["hold_hours"], bins=bins, labels=labels)

    # Aggregate
    results = []
    for bucket in labels:
        bucket_data = df[df["bucket"] == bucket]
        if len(bucket_data) == 0:
            continue

        count = len(bucket_data)
        avg_pnl = bucket_data["pnl"].mean()
        win_rate = (bucket_data["pnl"] > 0).mean()

        wins = bucket_data[bucket_data["pnl"] > 0]["pnl"].sum()
        losses = abs(bucket_data[bucket_data["pnl"] < 0]["pnl"].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')

        results.append({
            "bucket": bucket,
            "count": count,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        })

    return pd.DataFrame(results)


def entry_timing_quality(positions: List[Position], market_data: pd.DataFrame) -> Dict[str, float]:
    """Assess quality of entry timing relative to price movements.

    Args:
        positions: List of positions
        market_data: DataFrame with columns [date, symbol, open, high, low, close]

    Returns:
        Dictionary with timing metrics:
        - avg_entry_vs_daily_low: How close entry was to daily low (0-1, higher is better)
        - avg_exit_vs_daily_high: How close exit was to daily high (0-1, higher is better)
        - missed_profit_pct: Average unrealized profit from not selling at daily high
    """
    entry_scores = []
    exit_scores = []
    missed_profits = []

    for pos in positions:
        # Find market data for entry day
        entry_day = market_data[
            (market_data["symbol"] == pos.symbol) &
            (market_data["date"].dt.date == pos.entry_date.date())
        ]

        if entry_day.empty:
            continue

        row = entry_day.iloc[0]
        daily_range = row["high"] - row["low"]

        if daily_range > 0:
            # Entry quality: 0 if bought at high, 1 if bought at low
            if pos.side == "long":
                entry_score = (row["high"] - pos.entry_price) / daily_range
            else:
                entry_score = (pos.entry_price - row["low"]) / daily_range
            entry_scores.append(entry_score)

        # Find market data for exit day
        exit_day = market_data[
            (market_data["symbol"] == pos.symbol) &
            (market_data["date"].dt.date == pos.exit_date.date())
        ]

        if exit_day.empty:
            continue

        exit_row = exit_day.iloc[0]
        exit_range = exit_row["high"] - exit_row["low"]

        if exit_range > 0:
            # Exit quality: 0 if sold at low, 1 if sold at high
            if pos.side == "long":
                exit_score = (pos.exit_price - exit_row["low"]) / exit_range
                best_exit = exit_row["high"]
            else:
                exit_score = (exit_row["high"] - pos.exit_price) / exit_range
                best_exit = exit_row["low"]

            exit_scores.append(exit_score)

            # Missed profit
            actual_pnl_per_share = abs(pos.exit_price - pos.entry_price)
            potential_pnl_per_share = abs(best_exit - pos.entry_price)
            if actual_pnl_per_share > 0:
                missed_pct = (potential_pnl_per_share - actual_pnl_per_share) / actual_pnl_per_share
                missed_profits.append(missed_pct)

    return {
        "avg_entry_vs_daily_low": np.mean(entry_scores) if entry_scores else 0.0,
        "avg_exit_vs_daily_high": np.mean(exit_scores) if exit_scores else 0.0,
        "missed_profit_pct": np.mean(missed_profits) if missed_profits else 0.0,
    }
