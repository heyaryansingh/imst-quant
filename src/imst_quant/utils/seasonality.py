"""Seasonality analysis for identifying temporal patterns in market data.

This module provides tools for detecting and analyzing seasonal patterns
in returns, volume, and volatility across different time horizons.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.seasonality import (
    ...     analyze_day_of_week_effect,
    ...     analyze_monthly_seasonality,
    ...     analyze_time_of_day_patterns,
    ... )
    >>> df = pl.read_parquet("features.parquet")
    >>> dow_effect = analyze_day_of_week_effect(df)
    >>> print(dow_effect)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass
class SeasonalityResults:
    """Container for seasonality analysis results.

    Attributes:
        period_type: Type of seasonality ('day_of_week', 'month', 'hour', etc.).
        by_period: DataFrame with metrics by period.
        best_period: Period with highest average return.
        worst_period: Period with lowest average return.
        spread: Difference between best and worst period returns.
        is_significant: Whether the pattern appears statistically significant.
        periods_analyzed: Number of data points in analysis.
    """

    period_type: str
    by_period: pl.DataFrame
    best_period: str | int
    worst_period: str | int
    spread: float
    is_significant: bool
    periods_analyzed: int


def analyze_day_of_week_effect(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
) -> SeasonalityResults:
    """Analyze returns by day of week.

    Identifies patterns like the "Monday effect" where returns differ
    systematically by day of week.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.

    Returns:
        SeasonalityResults with day-of-week breakdown.

    Example:
        >>> results = analyze_day_of_week_effect(df)
        >>> print(f"Best day: {results.best_period}")
    """
    if date_col not in df.columns or return_col not in df.columns:
        return SeasonalityResults(
            period_type="day_of_week",
            by_period=pl.DataFrame(),
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    # Extract day of week (0=Monday, 6=Sunday)
    analysis = df.with_columns(
        pl.col(date_col).dt.weekday().alias("dow")
    )

    # Group by day of week
    by_dow = (
        analysis.group_by("dow")
        .agg(
            pl.col(return_col).mean().alias("avg_return"),
            pl.col(return_col).std().alias("std_return"),
            pl.col(return_col).median().alias("median_return"),
            pl.len().alias("count"),
            (pl.col(return_col) > 0).mean().alias("win_rate"),
        )
        .sort("dow")
    )

    # Map day numbers to names
    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    by_dow = by_dow.with_columns(
        pl.col("dow").replace_strict(day_names, default="Unknown").alias("day_name")
    )

    # Find best/worst days
    avg_returns = by_dow["avg_return"].to_list()
    day_nums = by_dow["dow"].to_list()

    if not avg_returns:
        return SeasonalityResults(
            period_type="day_of_week",
            by_period=by_dow,
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    best_idx = avg_returns.index(max(avg_returns))
    worst_idx = avg_returns.index(min(avg_returns))

    best_day = day_names.get(day_nums[best_idx], "Unknown")
    worst_day = day_names.get(day_nums[worst_idx], "Unknown")
    spread = max(avg_returns) - min(avg_returns)

    # Simple significance test: spread > 2x average std
    avg_std = by_dow["std_return"].mean()
    is_significant = spread > (2 * avg_std) if avg_std else False

    return SeasonalityResults(
        period_type="day_of_week",
        by_period=by_dow,
        best_period=best_day,
        worst_period=worst_day,
        spread=spread,
        is_significant=is_significant,
        periods_analyzed=df.height,
    )


def analyze_monthly_seasonality(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
) -> SeasonalityResults:
    """Analyze returns by month of year.

    Identifies patterns like the "January effect" or "Sell in May" anomaly.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.

    Returns:
        SeasonalityResults with monthly breakdown.

    Example:
        >>> results = analyze_monthly_seasonality(df)
        >>> print(f"Best month: {results.best_period}")
    """
    if date_col not in df.columns or return_col not in df.columns:
        return SeasonalityResults(
            period_type="month",
            by_period=pl.DataFrame(),
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    # Extract month
    analysis = df.with_columns(
        pl.col(date_col).dt.month().alias("month")
    )

    # Group by month
    by_month = (
        analysis.group_by("month")
        .agg(
            pl.col(return_col).mean().alias("avg_return"),
            pl.col(return_col).std().alias("std_return"),
            pl.col(return_col).sum().alias("total_return"),
            pl.len().alias("count"),
            (pl.col(return_col) > 0).mean().alias("win_rate"),
        )
        .sort("month")
    )

    # Map month numbers to names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    by_month = by_month.with_columns(
        pl.col("month").replace_strict(month_names, default="Unknown").alias("month_name")
    )

    # Find best/worst months
    avg_returns = by_month["avg_return"].to_list()
    month_nums = by_month["month"].to_list()

    if not avg_returns:
        return SeasonalityResults(
            period_type="month",
            by_period=by_month,
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    best_idx = avg_returns.index(max(avg_returns))
    worst_idx = avg_returns.index(min(avg_returns))

    best_month = month_names.get(month_nums[best_idx], "Unknown")
    worst_month = month_names.get(month_nums[worst_idx], "Unknown")
    spread = max(avg_returns) - min(avg_returns)

    avg_std = by_month["std_return"].mean()
    is_significant = spread > (2 * avg_std) if avg_std else False

    return SeasonalityResults(
        period_type="month",
        by_period=by_month,
        best_period=best_month,
        worst_period=worst_month,
        spread=spread,
        is_significant=is_significant,
        periods_analyzed=df.height,
    )


def analyze_quarter_end_effect(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
    window_days: int = 5,
) -> pl.DataFrame:
    """Analyze returns around quarter end dates.

    Detects patterns related to portfolio rebalancing and window dressing.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.
        window_days: Days around quarter end to analyze.

    Returns:
        DataFrame with quarter-end analysis.

    Example:
        >>> quarter_effect = analyze_quarter_end_effect(df)
        >>> print(quarter_effect)
    """
    if date_col not in df.columns or return_col not in df.columns:
        return pl.DataFrame()

    # Add quarter info
    analysis = df.with_columns([
        pl.col(date_col).dt.month().alias("month"),
        pl.col(date_col).dt.day().alias("day"),
    ])

    # Quarter end months: 3, 6, 9, 12
    quarter_ends = [3, 6, 9, 12]

    # Tag quarter end proximity
    analysis = analysis.with_columns(
        pl.when(
            pl.col("month").is_in(quarter_ends) & (pl.col("day") >= (28 - window_days))
        ).then(pl.lit("last_week_of_quarter"))
        .when(
            pl.col("month").is_in([m + 1 if m < 12 else 1 for m in quarter_ends])
            & (pl.col("day") <= window_days)
        ).then(pl.lit("first_week_of_quarter"))
        .otherwise(pl.lit("mid_quarter"))
        .alias("quarter_position")
    )

    # Group by position
    result = (
        analysis.group_by("quarter_position")
        .agg(
            pl.col(return_col).mean().alias("avg_return"),
            pl.col(return_col).std().alias("std_return"),
            pl.len().alias("count"),
            (pl.col(return_col) > 0).mean().alias("win_rate"),
        )
    )

    return result


def analyze_holiday_effect(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
    pre_holiday_days: int = 2,
    post_holiday_days: int = 2,
) -> dict[str, float]:
    """Analyze returns before and after major market holidays.

    Detects the well-documented pre-holiday drift phenomenon.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.
        pre_holiday_days: Days before holiday to analyze.
        post_holiday_days: Days after holiday to analyze.

    Returns:
        Dictionary with pre/post holiday statistics.

    Example:
        >>> effect = analyze_holiday_effect(df)
        >>> print(f"Pre-holiday avg: {effect['pre_holiday_avg']:.4%}")
    """
    if date_col not in df.columns or return_col not in df.columns:
        return {
            "pre_holiday_avg": 0.0,
            "post_holiday_avg": 0.0,
            "normal_avg": 0.0,
            "pre_holiday_excess": 0.0,
        }

    # US market holidays (approximate dates)
    # This is simplified; production would use actual holiday calendar
    analysis = df.with_columns([
        pl.col(date_col).dt.month().alias("month"),
        pl.col(date_col).dt.day().alias("day"),
    ])

    # Major holidays: New Year, MLK, Presidents Day, Memorial Day, July 4,
    # Labor Day, Thanksgiving, Christmas
    # Using week-based approximations for floating holidays

    # Tag potential holiday proximity (simplified)
    analysis = analysis.with_columns(
        pl.when(
            # Christmas/New Year period
            ((pl.col("month") == 12) & (pl.col("day") >= 23))
            | ((pl.col("month") == 1) & (pl.col("day") <= 3))
            # July 4th
            | ((pl.col("month") == 7) & (pl.col("day").is_between(2, 6)))
            # Thanksgiving (approximate)
            | ((pl.col("month") == 11) & (pl.col("day").is_between(22, 28)))
        ).then(pl.lit("holiday_adjacent"))
        .otherwise(pl.lit("normal"))
        .alias("holiday_flag")
    )

    # Calculate averages
    holiday_adj = analysis.filter(pl.col("holiday_flag") == "holiday_adjacent")
    normal = analysis.filter(pl.col("holiday_flag") == "normal")

    holiday_avg = float(holiday_adj[return_col].mean()) if holiday_adj.height > 0 else 0.0
    normal_avg = float(normal[return_col].mean()) if normal.height > 0 else 0.0

    return {
        "pre_holiday_avg": holiday_avg,
        "post_holiday_avg": holiday_avg,  # Simplified
        "normal_avg": normal_avg,
        "pre_holiday_excess": holiday_avg - normal_avg,
        "holiday_days": holiday_adj.height,
        "normal_days": normal.height,
    }


def analyze_turn_of_month_effect(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
    tom_window: int = 4,
) -> dict[str, float]:
    """Analyze turn-of-month effect on returns.

    The turn-of-month effect suggests higher returns in the last and first
    few trading days of each month.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.
        tom_window: Days at start/end of month to consider.

    Returns:
        Dictionary with turn-of-month statistics.

    Example:
        >>> tom = analyze_turn_of_month_effect(df)
        >>> print(f"TOM avg: {tom['tom_avg']:.4%}")
    """
    if date_col not in df.columns or return_col not in df.columns:
        return {
            "tom_avg": 0.0,
            "mid_month_avg": 0.0,
            "tom_excess": 0.0,
            "tom_win_rate": 0.0,
            "mid_month_win_rate": 0.0,
        }

    # Extract day of month
    analysis = df.with_columns([
        pl.col(date_col).dt.day().alias("day"),
        pl.col(date_col).dt.month_end().dt.day().alias("days_in_month"),
    ])

    # Tag turn of month days
    analysis = analysis.with_columns(
        pl.when(
            (pl.col("day") <= tom_window)
            | (pl.col("day") >= pl.col("days_in_month") - tom_window + 1)
        ).then(pl.lit("turn_of_month"))
        .otherwise(pl.lit("mid_month"))
        .alias("tom_flag")
    )

    tom = analysis.filter(pl.col("tom_flag") == "turn_of_month")
    mid = analysis.filter(pl.col("tom_flag") == "mid_month")

    tom_avg = float(tom[return_col].mean()) if tom.height > 0 else 0.0
    mid_avg = float(mid[return_col].mean()) if mid.height > 0 else 0.0

    tom_win = float((tom[return_col] > 0).mean()) if tom.height > 0 else 0.0
    mid_win = float((mid[return_col] > 0).mean()) if mid.height > 0 else 0.0

    return {
        "tom_avg": tom_avg,
        "mid_month_avg": mid_avg,
        "tom_excess": tom_avg - mid_avg,
        "tom_win_rate": tom_win,
        "mid_month_win_rate": mid_win,
        "tom_days": tom.height,
        "mid_month_days": mid.height,
    }


def analyze_intraday_seasonality(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    return_col: str = "return",
) -> SeasonalityResults:
    """Analyze intraday return patterns by hour.

    Identifies patterns like higher volatility at market open/close.

    Args:
        df: DataFrame with timestamp and return data.
        time_col: Column name for timestamp.
        return_col: Column name for returns.

    Returns:
        SeasonalityResults with hourly breakdown.

    Example:
        >>> results = analyze_intraday_seasonality(df)
        >>> print(f"Best hour: {results.best_period}")
    """
    if time_col not in df.columns or return_col not in df.columns:
        return SeasonalityResults(
            period_type="hour",
            by_period=pl.DataFrame(),
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    # Extract hour
    analysis = df.with_columns(
        pl.col(time_col).dt.hour().alias("hour")
    )

    # Group by hour
    by_hour = (
        analysis.group_by("hour")
        .agg(
            pl.col(return_col).mean().alias("avg_return"),
            pl.col(return_col).std().alias("std_return"),
            pl.len().alias("count"),
            (pl.col(return_col) > 0).mean().alias("win_rate"),
        )
        .sort("hour")
    )

    avg_returns = by_hour["avg_return"].to_list()
    hours = by_hour["hour"].to_list()

    if not avg_returns:
        return SeasonalityResults(
            period_type="hour",
            by_period=by_hour,
            best_period="N/A",
            worst_period="N/A",
            spread=0.0,
            is_significant=False,
            periods_analyzed=0,
        )

    best_idx = avg_returns.index(max(avg_returns))
    worst_idx = avg_returns.index(min(avg_returns))
    spread = max(avg_returns) - min(avg_returns)

    avg_std = by_hour["std_return"].mean()
    is_significant = spread > (2 * avg_std) if avg_std else False

    return SeasonalityResults(
        period_type="hour",
        by_period=by_hour,
        best_period=hours[best_idx],
        worst_period=hours[worst_idx],
        spread=spread,
        is_significant=is_significant,
        periods_analyzed=df.height,
    )


def calculate_seasonal_factors(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
    method: str = "multiplicative",
) -> pl.DataFrame:
    """Calculate seasonal adjustment factors for returns.

    Creates factors that can be used to de-seasonalize returns or
    incorporate seasonality into forecasts.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.
        method: 'multiplicative' or 'additive' seasonal adjustment.

    Returns:
        DataFrame with seasonal factors by month and day of week.

    Example:
        >>> factors = calculate_seasonal_factors(df)
        >>> # Apply to forecast: forecast * factor
    """
    if date_col not in df.columns or return_col not in df.columns:
        return pl.DataFrame()

    analysis = df.with_columns([
        pl.col(date_col).dt.month().alias("month"),
        pl.col(date_col).dt.weekday().alias("dow"),
    ])

    overall_mean = analysis[return_col].mean()

    # Monthly factors
    monthly = (
        analysis.group_by("month")
        .agg(pl.col(return_col).mean().alias("month_mean"))
    )

    if method == "multiplicative" and overall_mean and overall_mean != 0:
        monthly = monthly.with_columns(
            (pl.col("month_mean") / overall_mean).alias("month_factor")
        )
    else:
        monthly = monthly.with_columns(
            (pl.col("month_mean") - overall_mean).alias("month_factor")
        )

    # Day of week factors
    dow = (
        analysis.group_by("dow")
        .agg(pl.col(return_col).mean().alias("dow_mean"))
    )

    if method == "multiplicative" and overall_mean and overall_mean != 0:
        dow = dow.with_columns(
            (pl.col("dow_mean") / overall_mean).alias("dow_factor")
        )
    else:
        dow = dow.with_columns(
            (pl.col("dow_mean") - overall_mean).alias("dow_factor")
        )

    # Combine into factors table
    result = (
        monthly.select("month", "month_factor")
        .join(
            dow.select("dow", "dow_factor"),
            how="cross",
        )
        .sort(["month", "dow"])
    )

    if method == "multiplicative":
        result = result.with_columns(
            (pl.col("month_factor") * pl.col("dow_factor")).alias("combined_factor")
        )
    else:
        result = result.with_columns(
            (pl.col("month_factor") + pl.col("dow_factor")).alias("combined_factor")
        )

    return result


def generate_seasonality_report(
    df: pl.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
) -> str:
    """Generate a comprehensive seasonality analysis report.

    Args:
        df: DataFrame with date and return data.
        date_col: Column name for date.
        return_col: Column name for returns.

    Returns:
        Formatted string with seasonality analysis.

    Example:
        >>> report = generate_seasonality_report(df)
        >>> print(report)
    """
    lines = [
        "=" * 60,
        "SEASONALITY ANALYSIS REPORT",
        "=" * 60,
        "",
    ]

    # Day of week analysis
    dow = analyze_day_of_week_effect(df, date_col, return_col)
    lines.extend([
        "DAY OF WEEK EFFECT",
        "-" * 40,
        f"Best Day:     {dow.best_period}",
        f"Worst Day:    {dow.worst_period}",
        f"Spread:       {dow.spread:.4%}",
        f"Significant:  {'Yes' if dow.is_significant else 'No'}",
        "",
    ])

    # Monthly analysis
    monthly = analyze_monthly_seasonality(df, date_col, return_col)
    lines.extend([
        "MONTHLY SEASONALITY",
        "-" * 40,
        f"Best Month:   {monthly.best_period}",
        f"Worst Month:  {monthly.worst_period}",
        f"Spread:       {monthly.spread:.4%}",
        f"Significant:  {'Yes' if monthly.is_significant else 'No'}",
        "",
    ])

    # Turn of month
    tom = analyze_turn_of_month_effect(df, date_col, return_col)
    lines.extend([
        "TURN OF MONTH EFFECT",
        "-" * 40,
        f"TOM Avg Return:     {tom['tom_avg']:.4%}",
        f"Mid-Month Avg:      {tom['mid_month_avg']:.4%}",
        f"TOM Excess:         {tom['tom_excess']:.4%}",
        f"TOM Win Rate:       {tom['tom_win_rate']:.1%}",
        "",
    ])

    # Holiday effect
    holiday = analyze_holiday_effect(df, date_col, return_col)
    lines.extend([
        "HOLIDAY EFFECT",
        "-" * 40,
        f"Holiday-Adj Avg:    {holiday['pre_holiday_avg']:.4%}",
        f"Normal Avg:         {holiday['normal_avg']:.4%}",
        f"Holiday Excess:     {holiday['pre_holiday_excess']:.4%}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
