"""Equity curve analysis utilities.

Provides tools for analyzing the equity curve of a trading strategy or
portfolio, including smoothing, regime labeling, performance attribution
by time period, and comparison with benchmarks.

Functions:
    build_equity_curve: Construct equity curve from returns
    equity_curve_statistics: Compute key statistics of an equity curve
    equity_curve_regimes: Label equity curve into growth/drawdown regimes
    compare_equity_curves: Compare multiple strategies' equity curves
    rolling_cagr: Calculate rolling compound annual growth rate
    time_period_returns: Break down returns by calendar periods

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.equity_curve import build_equity_curve
    >>> returns = pl.Series("returns", [0.01, -0.005, 0.02, -0.01, 0.015])
    >>> curve = build_equity_curve(returns, initial_capital=10000.0)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class EquityCurveStats:
    """Summary statistics for an equity curve.

    Attributes:
        total_return: Cumulative return over the entire period.
        cagr: Compound annual growth rate.
        max_drawdown: Maximum peak-to-trough decline.
        max_drawdown_duration: Longest drawdown period in observations.
        sharpe_ratio: Annualized Sharpe ratio (assuming 252 trading days).
        calmar_ratio: CAGR / max drawdown.
        final_value: Final equity value.
        peak_value: Highest equity value reached.
        num_periods: Total number of return observations.
        positive_periods: Number of positive return periods.
        negative_periods: Number of negative return periods.
    """

    total_return: float
    cagr: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    calmar_ratio: float
    final_value: float
    peak_value: float
    num_periods: int
    positive_periods: int
    negative_periods: int


@dataclass
class PeriodReturn:
    """Return for a specific calendar period.

    Attributes:
        period: Period label (e.g., "2024-Q1", "2024-01").
        total_return: Cumulative return for the period.
        num_observations: Number of trading days in the period.
    """

    period: str
    total_return: float
    num_observations: int


def build_equity_curve(
    returns: pl.Series,
    initial_capital: float = 10000.0,
) -> pl.DataFrame:
    """Construct an equity curve from a series of returns.

    Args:
        returns: Series of periodic returns (e.g., daily).
        initial_capital: Starting capital value.

    Returns:
        DataFrame with columns:
        - period: Period index (0-based)
        - returns: Original returns
        - cumulative_return: Cumulative compounded return
        - equity: Equity value
        - peak: Running peak equity
        - drawdown: Current drawdown from peak (negative value)
        - drawdown_pct: Drawdown as percentage of peak
    """
    n = len(returns)
    returns_list = returns.to_list()

    # Compute cumulative product of (1 + r)
    cum_product = np.cumprod([1.0 + r for r in returns_list])
    equity = initial_capital * cum_product

    # Running peak
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    drawdown_pct = np.where(peak > 0, drawdown / peak, 0.0)

    return pl.DataFrame(
        {
            "period": list(range(n)),
            "returns": returns_list,
            "cumulative_return": (cum_product - 1.0).tolist(),
            "equity": equity.tolist(),
            "peak": peak.tolist(),
            "drawdown": drawdown.tolist(),
            "drawdown_pct": drawdown_pct.tolist(),
        }
    )


def equity_curve_statistics(
    returns: pl.Series,
    initial_capital: float = 10000.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> EquityCurveStats:
    """Compute key statistics of an equity curve.

    Args:
        returns: Series of periodic returns.
        initial_capital: Starting capital.
        periods_per_year: Trading periods per year (252 for daily).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        EquityCurveStats with comprehensive metrics.
    """
    returns_arr = np.array(returns.to_list())
    n = len(returns_arr)

    if n == 0:
        return EquityCurveStats(
            total_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            final_value=initial_capital,
            peak_value=initial_capital,
            num_periods=0,
            positive_periods=0,
            negative_periods=0,
        )

    # Cumulative return
    cum_product = np.cumprod(1.0 + returns_arr)
    total_return = float(cum_product[-1] - 1.0)
    final_value = initial_capital * cum_product[-1]

    # CAGR
    years = n / periods_per_year
    if years > 0 and cum_product[-1] > 0:
        cagr = float(cum_product[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    # Max drawdown
    equity = initial_capital * cum_product
    peak = np.maximum.accumulate(equity)
    dd_pct = np.where(peak > 0, (equity - peak) / peak, 0.0)
    max_drawdown = float(np.min(dd_pct))
    peak_value = float(np.max(peak))

    # Max drawdown duration
    in_drawdown = equity < peak
    max_dd_duration = 0
    current_dd_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0

    # Sharpe ratio
    excess_returns = returns_arr - risk_free_rate / periods_per_year
    std = float(np.std(excess_returns, ddof=1)) if n > 1 else 1.0
    sharpe = float(np.mean(excess_returns) / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    # Calmar ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return EquityCurveStats(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        final_value=final_value,
        peak_value=peak_value,
        num_periods=n,
        positive_periods=int(np.sum(returns_arr > 0)),
        negative_periods=int(np.sum(returns_arr < 0)),
    )


def equity_curve_regimes(
    returns: pl.Series,
    initial_capital: float = 10000.0,
    drawdown_threshold: float = -0.05,
) -> pl.DataFrame:
    """Label equity curve periods as growth or drawdown regimes.

    Args:
        returns: Series of periodic returns.
        initial_capital: Starting capital.
        drawdown_threshold: Drawdown percentage below which regime = "drawdown".

    Returns:
        DataFrame with equity curve data plus:
        - regime: "growth" or "drawdown"
        - regime_duration: Consecutive periods in current regime
    """
    curve = build_equity_curve(returns, initial_capital)

    # Label regimes
    curve = curve.with_columns(
        pl.when(pl.col("drawdown_pct") <= drawdown_threshold)
        .then(pl.lit("drawdown"))
        .otherwise(pl.lit("growth"))
        .alias("regime")
    )

    # Calculate regime duration (consecutive periods in same regime)
    regimes = curve["regime"].to_list()
    durations = []
    current_duration = 0
    prev_regime = None
    for regime in regimes:
        if regime == prev_regime:
            current_duration += 1
        else:
            current_duration = 1
            prev_regime = regime
        durations.append(current_duration)

    curve = curve.with_columns(pl.Series("regime_duration", durations))

    return curve


def compare_equity_curves(
    returns_dict: Dict[str, pl.Series],
    initial_capital: float = 10000.0,
    periods_per_year: int = 252,
) -> Dict[str, EquityCurveStats]:
    """Compare multiple strategies' equity curves.

    Args:
        returns_dict: Mapping of strategy name to returns series.
        initial_capital: Starting capital for each strategy.
        periods_per_year: Trading periods per year.

    Returns:
        Dict mapping strategy name to EquityCurveStats.
    """
    return {
        name: equity_curve_statistics(returns, initial_capital, periods_per_year)
        for name, returns in returns_dict.items()
    }


def rolling_cagr(
    returns: pl.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pl.Series:
    """Calculate rolling compound annual growth rate.

    Args:
        returns: Series of periodic returns.
        window: Rolling window size in periods.
        periods_per_year: Periods per year for annualization.

    Returns:
        Series of rolling CAGR values (NaN for initial window).
    """
    returns_arr = np.array(returns.to_list())
    n = len(returns_arr)
    cagr_values = [float("nan")] * min(window - 1, n)

    for i in range(window - 1, n):
        window_returns = returns_arr[i - window + 1 : i + 1]
        cum_return = float(np.prod(1.0 + window_returns))
        years = window / periods_per_year
        if cum_return > 0 and years > 0:
            cagr_val = cum_return ** (1.0 / years) - 1.0
        else:
            cagr_val = 0.0
        cagr_values.append(float(cagr_val))

    return pl.Series("rolling_cagr", cagr_values)


def time_period_returns(
    returns: pl.Series,
    dates: pl.Series,
    period: str = "month",
) -> List[PeriodReturn]:
    """Break down returns by calendar periods.

    Args:
        returns: Series of periodic returns.
        dates: Series of date values (Date or Datetime).
        period: Aggregation period - "month", "quarter", or "year".

    Returns:
        List of PeriodReturn objects for each calendar period.
    """
    df = pl.DataFrame({"date": dates, "returns": returns})

    if period == "month":
        df = df.with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("period_label")
        )
    elif period == "quarter":
        df = df.with_columns(
            (
                pl.col("date").dt.year().cast(pl.Utf8)
                + pl.lit("-Q")
                + pl.col("date").dt.quarter().cast(pl.Utf8)
            ).alias("period_label")
        )
    elif period == "year":
        df = df.with_columns(
            pl.col("date").dt.year().cast(pl.Utf8).alias("period_label")
        )
    else:
        raise ValueError(f"Unknown period: {period}. Use 'month', 'quarter', or 'year'.")

    # Compute compounded return per period
    grouped = df.group_by("period_label", maintain_order=True).agg(
        ((1.0 + pl.col("returns")).product() - 1.0).alias("total_return"),
        pl.col("returns").count().alias("num_observations"),
    )

    results = []
    for row in grouped.iter_rows(named=True):
        results.append(
            PeriodReturn(
                period=row["period_label"],
                total_return=row["total_return"],
                num_observations=row["num_observations"],
            )
        )

    return results
