"""Temporal feature engineering for time series analysis.

Extract calendar-based features, cyclical encodings, and time-based patterns
for financial time series modeling.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.temporal_features import add_calendar_features
    >>> df = pl.DataFrame({"date": pl.date_range(start="2024-01-01", end="2024-01-10", eager=True)})
    >>> df = add_calendar_features(df, "date")
"""

import polars as pl
import numpy as np
from typing import List, Optional


def add_calendar_features(
    df: pl.DataFrame,
    date_col: str = "date",
    features: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Add calendar-based features from date column.

    Args:
        df: Input DataFrame with date column.
        date_col: Name of date column (default: "date").
        features: List of features to add. Options:
            - "year", "month", "day", "dayofweek", "dayofyear"
            - "weekofyear", "quarter", "is_month_start", "is_month_end"
            - "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
            Defaults to ["month", "dayofweek", "quarter"].

    Returns:
        DataFrame with added calendar features.

    Example:
        >>> df = pl.DataFrame({"date": pl.date_range(start="2024-01-01", end="2024-03-01", eager=True)})
        >>> df = add_calendar_features(df, features=["month", "dayofweek"])
    """
    if features is None:
        features = ["month", "dayofweek", "quarter"]

    result = df.clone()

    if "year" in features:
        result = result.with_columns(pl.col(date_col).dt.year().alias("year"))

    if "month" in features:
        result = result.with_columns(pl.col(date_col).dt.month().alias("month"))

    if "day" in features:
        result = result.with_columns(pl.col(date_col).dt.day().alias("day"))

    if "dayofweek" in features:
        result = result.with_columns(pl.col(date_col).dt.weekday().alias("dayofweek"))

    if "dayofyear" in features:
        result = result.with_columns(pl.col(date_col).dt.ordinal_day().alias("dayofyear"))

    if "weekofyear" in features:
        result = result.with_columns(pl.col(date_col).dt.week().alias("weekofyear"))

    if "quarter" in features:
        result = result.with_columns(pl.col(date_col).dt.quarter().alias("quarter"))

    if "is_month_start" in features:
        result = result.with_columns(
            (pl.col(date_col).dt.day() == 1).alias("is_month_start")
        )

    if "is_month_end" in features:
        result = result.with_columns(
            (pl.col(date_col).dt.day() == pl.col(date_col).dt.days_in_month()).alias("is_month_end")
        )

    if "is_quarter_start" in features:
        result = result.with_columns(
            ((pl.col(date_col).dt.month() % 3 == 1) & (pl.col(date_col).dt.day() == 1)).alias("is_quarter_start")
        )

    if "is_quarter_end" in features:
        is_q_end_month = pl.col(date_col).dt.month().is_in([3, 6, 9, 12])
        is_last_day = pl.col(date_col).dt.day() == pl.col(date_col).dt.days_in_month()
        result = result.with_columns((is_q_end_month & is_last_day).alias("is_quarter_end"))

    if "is_year_start" in features:
        result = result.with_columns(
            ((pl.col(date_col).dt.month() == 1) & (pl.col(date_col).dt.day() == 1)).alias("is_year_start")
        )

    if "is_year_end" in features:
        result = result.with_columns(
            ((pl.col(date_col).dt.month() == 12) & (pl.col(date_col).dt.day() == 31)).alias("is_year_end")
        )

    return result


def add_cyclical_encoding(
    df: pl.DataFrame,
    col: str,
    period: int,
    prefix: Optional[str] = None,
) -> pl.DataFrame:
    """Encode cyclical features using sin/cos transformation.

    Converts cyclical features (e.g., month, hour) into continuous sin/cos
    pairs that preserve cyclical nature (e.g., December is close to January).

    Args:
        df: Input DataFrame.
        col: Column to encode (must be numeric).
        period: Period of the cycle (e.g., 12 for months, 24 for hours).
        prefix: Prefix for output columns (default: col name).

    Returns:
        DataFrame with {prefix}_sin and {prefix}_cos columns.

    Example:
        >>> df = pl.DataFrame({"month": [1, 6, 12]})
        >>> df = add_cyclical_encoding(df, "month", period=12)
        >>> print(df[["month_sin", "month_cos"]])
    """
    if prefix is None:
        prefix = col

    result = df.with_columns([
        (2 * np.pi * pl.col(col) / period).sin().alias(f"{prefix}_sin"),
        (2 * np.pi * pl.col(col) / period).cos().alias(f"{prefix}_cos"),
    ])

    return result


def add_trading_day_features(
    df: pl.DataFrame,
    date_col: str = "date",
) -> pl.DataFrame:
    """Add trading-specific calendar features.

    Args:
        df: Input DataFrame with date column.
        date_col: Name of date column (default: "date").

    Returns:
        DataFrame with:
        - is_monday, is_friday: Boolean flags
        - days_to_month_end: Days until end of month
        - days_from_month_start: Days since start of month

    Example:
        >>> df = pl.DataFrame({"date": pl.date_range(start="2024-01-01", end="2024-01-10", eager=True)})
        >>> df = add_trading_day_features(df)
    """
    result = df.with_columns([
        (pl.col(date_col).dt.weekday() == 1).alias("is_monday"),
        (pl.col(date_col).dt.weekday() == 5).alias("is_friday"),
        pl.col(date_col).dt.day().alias("day_of_month"),
        pl.col(date_col).dt.days_in_month().alias("days_in_month"),
    ])

    result = result.with_columns([
        (pl.col("day_of_month") - 1).alias("days_from_month_start"),
        (pl.col("days_in_month") - pl.col("day_of_month")).alias("days_to_month_end"),
    ])

    return result.drop(["day_of_month", "days_in_month"])


def add_lag_features(
    df: pl.DataFrame,
    cols: List[str],
    lags: List[int],
    fill_null: Optional[float] = None,
) -> pl.DataFrame:
    """Add lagged versions of columns.

    Args:
        df: Input DataFrame.
        cols: Columns to create lags for.
        lags: List of lag periods (e.g., [1, 5, 10]).
        fill_null: Value to fill nulls with (default: None).

    Returns:
        DataFrame with lagged columns named {col}_lag{lag}.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 105, 108, 110]})
        >>> df = add_lag_features(df, cols=["price"], lags=[1, 2])
        >>> print(df[["price", "price_lag1", "price_lag2"]])
    """
    result = df.clone()

    for col in cols:
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            lagged = pl.col(col).shift(lag)

            if fill_null is not None:
                lagged = lagged.fill_null(fill_null)

            result = result.with_columns(lagged.alias(lag_col))

    return result


def add_diff_features(
    df: pl.DataFrame,
    cols: List[str],
    periods: List[int] = [1],
) -> pl.DataFrame:
    """Add differenced features (value - value_at_t-n).

    Args:
        df: Input DataFrame.
        cols: Columns to difference.
        periods: Periods to difference over (default: [1]).

    Returns:
        DataFrame with diff columns named {col}_diff{period}.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 105, 108, 110]})
        >>> df = add_diff_features(df, cols=["price"], periods=[1])
        >>> print(df[["price", "price_diff1"]])
    """
    result = df.clone()

    for col in cols:
        for period in periods:
            diff_col = f"{col}_diff{period}"
            result = result.with_columns(
                pl.col(col).diff(period).alias(diff_col)
            )

    return result


def add_pct_change_features(
    df: pl.DataFrame,
    cols: List[str],
    periods: List[int] = [1],
) -> pl.DataFrame:
    """Add percentage change features.

    Args:
        df: Input DataFrame.
        cols: Columns to compute pct change for.
        periods: Periods to compute change over (default: [1]).

    Returns:
        DataFrame with columns named {col}_pct{period}.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 105, 108, 110]})
        >>> df = add_pct_change_features(df, cols=["price"], periods=[1])
        >>> print(df[["price", "price_pct1"]])
    """
    result = df.clone()

    for col in cols:
        for period in periods:
            pct_col = f"{col}_pct{period}"
            result = result.with_columns(
                (pl.col(col) / pl.col(col).shift(period) - 1.0).alias(pct_col)
            )

    return result


def add_time_since_event(
    df: pl.DataFrame,
    date_col: str,
    event_dates: List[str],
    unit: str = "days",
) -> pl.DataFrame:
    """Add features measuring time since key events.

    Args:
        df: Input DataFrame with date column.
        date_col: Name of date column.
        event_dates: List of event dates in YYYY-MM-DD format.
        unit: Time unit ("days", "weeks", "months"). Default: "days".

    Returns:
        DataFrame with days_since_event_{i} columns.

    Example:
        >>> df = pl.DataFrame({"date": pl.date_range(start="2024-01-10", end="2024-01-20", eager=True)})
        >>> df = add_time_since_event(df, "date", event_dates=["2024-01-01"])
        >>> print(df[["date", "days_since_event_0"]])
    """
    result = df.clone()

    for i, event_date in enumerate(event_dates):
        event_dt = pl.lit(event_date).str.to_date()
        time_diff = (pl.col(date_col) - event_dt).dt.total_days()

        if unit == "weeks":
            time_diff = time_diff / 7.0
        elif unit == "months":
            time_diff = time_diff / 30.0

        result = result.with_columns(
            time_diff.alias(f"{unit}_since_event_{i}")
        )

    return result


def add_business_day_count(
    df: pl.DataFrame,
    date_col: str = "date",
    window: int = 20,
) -> pl.DataFrame:
    """Count business days (Mon-Fri) in rolling window.

    Args:
        df: Input DataFrame.
        date_col: Name of date column (default: "date").
        window: Rolling window size (default: 20).

    Returns:
        DataFrame with business_days_{window} column.

    Example:
        >>> df = pl.DataFrame({"date": pl.date_range(start="2024-01-01", end="2024-01-31", eager=True)})
        >>> df = add_business_day_count(df, window=7)
    """
    is_business_day = (pl.col(date_col).dt.weekday() <= 5).cast(pl.Int32)

    result = df.with_columns(
        is_business_day.rolling_sum(window_size=window).alias(f"business_days_{window}")
    )

    return result
