"""Win/loss streak analysis for trading performance.

Provides tools for analyzing consecutive wins/losses, including:
- Streak identification and tracking
- Maximum favorable/adverse excursions
- Run statistics and probability
- Gambler's ruin analysis
- Recovery time estimation

Example:
    >>> from imst_quant.utils.streak_analysis import analyze_streaks
    >>> results = analyze_streaks(trade_returns)
    >>> print(f"Max win streak: {results.max_win_streak}")
    Max win streak: 7

References:
    - Wald (1945): Sequential Analysis
    - Tharp (2008): Trade Your Way to Financial Freedom
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class StreakPeriod:
    """A single streak period.

    Attributes:
        start_idx: Starting index.
        end_idx: Ending index (inclusive).
        length: Number of periods.
        streak_type: "win" or "loss".
        total_return: Cumulative return during streak.
        avg_return: Average return per period.
        peak_return: Maximum cumulative return.
        trough_return: Minimum cumulative return.
    """
    start_idx: int
    end_idx: int
    length: int
    streak_type: str
    total_return: float
    avg_return: float
    peak_return: float
    trough_return: float


@dataclass
class StreakStatistics:
    """Comprehensive streak analysis results.

    Attributes:
        total_periods: Total number of periods analyzed.
        win_count: Number of winning periods.
        loss_count: Number of losing periods.
        win_rate: Percentage of winning periods.
        max_win_streak: Longest winning streak.
        max_loss_streak: Longest losing streak.
        avg_win_streak: Average winning streak length.
        avg_loss_streak: Average losing streak length.
        current_streak: Current streak length (positive=wins).
        win_streaks: List of all win streak periods.
        loss_streaks: List of all loss streak periods.
        expected_max_streak: Expected maximum streak given win rate.
        streak_runs_test_p: P-value for runs test (randomness).
        max_favorable_excursion: Best peak during any streak.
        max_adverse_excursion: Worst trough during any streak.
        recovery_times: Periods needed to recover from losses.
    """
    total_periods: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    current_streak: int = 0
    win_streaks: list[StreakPeriod] = field(default_factory=list)
    loss_streaks: list[StreakPeriod] = field(default_factory=list)
    expected_max_streak: float = 0.0
    streak_runs_test_p: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    recovery_times: list[int] = field(default_factory=list)


def identify_streaks(
    returns: pl.Series,
    threshold: float = 0.0,
) -> tuple[list[StreakPeriod], list[StreakPeriod]]:
    """Identify winning and losing streaks in return series.

    Args:
        returns: Series of period returns.
        threshold: Minimum return to count as win (default 0).

    Returns:
        Tuple of (win_streaks, loss_streaks).
    """
    values = returns.drop_nulls().to_numpy()
    n = len(values)

    if n == 0:
        return [], []

    win_streaks = []
    loss_streaks = []

    current_type: Optional[str] = None
    start_idx = 0
    cumulative = 0.0
    peak = 0.0
    trough = 0.0

    for i, ret in enumerate(values):
        is_win = ret > threshold
        new_type = "win" if is_win else "loss"

        if current_type is None:
            current_type = new_type
            start_idx = i
            cumulative = ret
            peak = max(0, ret)
            trough = min(0, ret)
        elif new_type != current_type:
            # End current streak
            period = StreakPeriod(
                start_idx=start_idx,
                end_idx=i - 1,
                length=i - start_idx,
                streak_type=current_type,
                total_return=cumulative,
                avg_return=cumulative / (i - start_idx),
                peak_return=peak,
                trough_return=trough,
            )

            if current_type == "win":
                win_streaks.append(period)
            else:
                loss_streaks.append(period)

            # Start new streak
            current_type = new_type
            start_idx = i
            cumulative = ret
            peak = max(0, ret)
            trough = min(0, ret)
        else:
            # Continue current streak
            cumulative += ret
            peak = max(peak, cumulative)
            trough = min(trough, cumulative)

    # Close final streak
    if current_type is not None:
        period = StreakPeriod(
            start_idx=start_idx,
            end_idx=n - 1,
            length=n - start_idx,
            streak_type=current_type,
            total_return=cumulative,
            avg_return=cumulative / (n - start_idx),
            peak_return=peak,
            trough_return=trough,
        )

        if current_type == "win":
            win_streaks.append(period)
        else:
            loss_streaks.append(period)

    return win_streaks, loss_streaks


def analyze_streaks(
    returns: pl.Series,
    threshold: float = 0.0,
) -> StreakStatistics:
    """Perform comprehensive streak analysis.

    Args:
        returns: Series of period returns.
        threshold: Minimum return to count as win.

    Returns:
        StreakStatistics with detailed analysis.
    """
    values = returns.drop_nulls().to_numpy()
    n = len(values)

    if n == 0:
        return StreakStatistics()

    # Basic counts
    wins = values > threshold
    win_count = int(np.sum(wins))
    loss_count = n - win_count
    win_rate = win_count / n if n > 0 else 0.0

    # Identify streaks
    win_streaks, loss_streaks = identify_streaks(returns, threshold)

    # Streak lengths
    win_lengths = [s.length for s in win_streaks]
    loss_lengths = [s.length for s in loss_streaks]

    max_win = max(win_lengths) if win_lengths else 0
    max_loss = max(loss_lengths) if loss_lengths else 0
    avg_win = np.mean(win_lengths) if win_lengths else 0.0
    avg_loss = np.mean(loss_lengths) if loss_lengths else 0.0

    # Current streak
    if n > 0:
        current_streak = 1
        current_is_win = values[-1] > threshold
        for i in range(n - 2, -1, -1):
            if (values[i] > threshold) == current_is_win:
                current_streak += 1
            else:
                break
        if not current_is_win:
            current_streak = -current_streak
    else:
        current_streak = 0

    # Expected maximum streak (geometric distribution)
    if win_rate > 0 and win_rate < 1:
        expected_max_win = np.log(n * win_rate * (1 - win_rate)) / np.log(win_rate)
        expected_max_loss = np.log(n * win_rate * (1 - win_rate)) / np.log(1 - win_rate)
        expected_max = max(expected_max_win, expected_max_loss)
    else:
        expected_max = n

    # Runs test for randomness
    runs_p = _runs_test(wins)

    # Maximum excursions
    mfe = max((s.peak_return for s in win_streaks), default=0.0)
    mae = min((s.trough_return for s in loss_streaks), default=0.0)

    # Recovery times (periods to recover from drawdown)
    recovery_times = _calculate_recovery_times(values)

    return StreakStatistics(
        total_periods=n,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        max_win_streak=max_win,
        max_loss_streak=max_loss,
        avg_win_streak=float(avg_win),
        avg_loss_streak=float(avg_loss),
        current_streak=current_streak,
        win_streaks=win_streaks,
        loss_streaks=loss_streaks,
        expected_max_streak=float(expected_max),
        streak_runs_test_p=runs_p,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        recovery_times=recovery_times,
    )


def _runs_test(wins: np.ndarray) -> float:
    """Wald-Wolfowitz runs test for randomness.

    Tests whether the sequence of wins/losses deviates from
    what would be expected under randomness.

    Args:
        wins: Boolean array of wins.

    Returns:
        P-value (low = non-random).
    """
    n = len(wins)
    if n < 10:
        return 1.0  # Not enough data

    n1 = int(np.sum(wins))
    n2 = n - n1

    if n1 == 0 or n2 == 0:
        return 1.0

    # Count runs
    runs = 1
    for i in range(1, n):
        if wins[i] != wins[i - 1]:
            runs += 1

    # Expected runs and variance under null
    expected_runs = (2 * n1 * n2 / n) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))

    if var_runs <= 0:
        return 1.0

    # Z-score
    z = (runs - expected_runs) / np.sqrt(var_runs)

    # Two-tailed p-value
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(p_value)


def _calculate_recovery_times(returns: np.ndarray) -> list[int]:
    """Calculate time to recover from each drawdown.

    Args:
        returns: Array of period returns.

    Returns:
        List of recovery periods for each drawdown.
    """
    if len(returns) == 0:
        return []

    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)

    recovery_times = []
    in_drawdown = False
    drawdown_start = 0

    for i in range(len(returns)):
        if cumulative[i] < running_max[i]:
            if not in_drawdown:
                in_drawdown = True
                drawdown_start = i
        else:
            if in_drawdown:
                recovery_times.append(i - drawdown_start)
                in_drawdown = False

    return recovery_times


def calculate_gambler_ruin_prob(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    bankroll_units: float,
) -> dict:
    """Estimate probability of ruin under geometric random walk.

    Uses simplified model assuming symmetric position sizing.

    Args:
        win_rate: Probability of winning trade.
        avg_win: Average winning trade size.
        avg_loss: Average losing trade size.
        bankroll_units: Starting bankroll in average loss units.

    Returns:
        Dict with ruin statistics.
    """
    if win_rate <= 0 or win_rate >= 1:
        return {"error": "Win rate must be between 0 and 1"}

    if avg_loss <= 0:
        return {"error": "Average loss must be positive"}

    # Risk-reward ratio
    rr_ratio = avg_win / avg_loss

    # Edge (expected value per unit risked)
    edge = win_rate * rr_ratio - (1 - win_rate)

    # Kelly criterion
    kelly = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio if rr_ratio > 0 else 0

    # Probability of ruin (simplified geometric model)
    if edge <= 0:
        ruin_prob = 1.0  # Negative edge = certain ruin
    else:
        # Approximate using exponential decay
        q_over_p = (1 - win_rate) / win_rate
        if rr_ratio == 1:
            ruin_prob = (q_over_p ** bankroll_units)
        else:
            # Adjust for asymmetric payoffs
            effective_units = bankroll_units * (1 + rr_ratio) / 2
            ruin_prob = max(0, min(1, q_over_p ** effective_units))

    return {
        "win_rate": win_rate,
        "risk_reward": rr_ratio,
        "edge": edge,
        "kelly_fraction": max(0, kelly),
        "ruin_probability": ruin_prob,
        "survival_probability": 1 - ruin_prob,
        "bankroll_units": bankroll_units,
        "expected_growth": edge / bankroll_units if bankroll_units > 0 else 0,
        "is_positive_expectancy": edge > 0,
    }


def generate_streak_report(stats: StreakStatistics) -> str:
    """Generate a formatted streak analysis report.

    Args:
        stats: StreakStatistics from analyze_streaks.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 50,
        "STREAK ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Total Periods:          {stats.total_periods:,}",
        f"Win Rate:               {stats.win_rate:.1%}",
        f"Wins / Losses:          {stats.win_count} / {stats.loss_count}",
        "",
        "--- Streak Lengths ---",
        f"Max Win Streak:         {stats.max_win_streak}",
        f"Max Loss Streak:        {stats.max_loss_streak}",
        f"Avg Win Streak:         {stats.avg_win_streak:.1f}",
        f"Avg Loss Streak:        {stats.avg_loss_streak:.1f}",
        f"Expected Max Streak:    {stats.expected_max_streak:.1f}",
        "",
        f"Current Streak:         {'+' if stats.current_streak > 0 else ''}{stats.current_streak}",
        "",
        "--- Excursions ---",
        f"Max Favorable (MFE):    {stats.max_favorable_excursion:+.2%}",
        f"Max Adverse (MAE):      {stats.max_adverse_excursion:+.2%}",
        "",
        "--- Randomness Test ---",
        f"Runs Test P-Value:      {stats.streak_runs_test_p:.4f}",
        f"Pattern Detected:       {'Yes (p < 0.05)' if stats.streak_runs_test_p < 0.05 else 'No'}",
        "",
    ]

    if stats.recovery_times:
        avg_recovery = np.mean(stats.recovery_times)
        max_recovery = max(stats.recovery_times)
        lines.extend([
            "--- Recovery Analysis ---",
            f"Avg Recovery Time:      {avg_recovery:.1f} periods",
            f"Max Recovery Time:      {max_recovery} periods",
            f"Drawdown Count:         {len(stats.recovery_times)}",
            "",
        ])

    return "\n".join(lines)
