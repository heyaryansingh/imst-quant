"""Data quality validation and integrity checking utilities.

This module provides comprehensive data quality checks for trading data,
including completeness, validity, and consistency validations. Essential
for ensuring reliable backtest and model training results.

Functions:
    validate_ohlcv: Validate OHLCV price data integrity
    validate_returns: Check return data for outliers and issues
    check_missing_dates: Identify gaps in time series data
    detect_outliers: Statistical outlier detection
    generate_quality_report: Create comprehensive quality assessment

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.data_quality import generate_quality_report
    >>> df = pl.read_parquet("data/bronze/market.parquet")
    >>> report = generate_quality_report(df)
    >>> print(f"Quality Score: {report['quality_score']:.1%}")
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl


@dataclass
class QualityIssue:
    """Represents a data quality issue found during validation.

    Attributes:
        severity: Issue severity ('error', 'warning', 'info')
        category: Issue category (e.g., 'missing_data', 'outlier')
        message: Human-readable description of the issue
        affected_rows: Number of rows affected by the issue
        details: Additional context or example values
    """

    severity: str
    category: str
    message: str
    affected_rows: int = 0
    details: Optional[Dict[str, Any]] = None


def validate_ohlcv(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> List[QualityIssue]:
    """Validate OHLCV (Open-High-Low-Close-Volume) data integrity.

    Checks for common data quality issues in price data including:
    - High >= Low constraint violations
    - High >= Open/Close violations
    - Low <= Open/Close violations
    - Negative prices or volumes
    - Zero volume days
    - Duplicate timestamps

    Args:
        df: DataFrame containing OHLCV data.
        open_col: Column name for open prices.
        high_col: Column name for high prices.
        low_col: Column name for low prices.
        close_col: Column name for close prices.
        volume_col: Column name for volume data.

    Returns:
        List of QualityIssue objects describing any problems found.

    Example:
        >>> df = pl.DataFrame({
        ...     "open": [100, 102], "high": [105, 98],  # 98 < open is error
        ...     "low": [99, 97], "close": [103, 100], "volume": [1000, 2000]
        ... })
        >>> issues = validate_ohlcv(df)
        >>> len(issues) > 0
        True
    """
    issues: List[QualityIssue] = []

    # Check for required columns
    required_cols = [open_col, high_col, low_col, close_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(QualityIssue(
            severity="error",
            category="schema",
            message=f"Missing required columns: {', '.join(missing_cols)}",
        ))
        return issues

    # High >= Low check
    invalid_hl = df.filter(pl.col(high_col) < pl.col(low_col)).height
    if invalid_hl > 0:
        issues.append(QualityIssue(
            severity="error",
            category="price_integrity",
            message=f"High < Low in {invalid_hl} rows",
            affected_rows=invalid_hl,
        ))

    # High >= Open and Close
    invalid_high = df.filter(
        (pl.col(high_col) < pl.col(open_col)) | (pl.col(high_col) < pl.col(close_col))
    ).height
    if invalid_high > 0:
        issues.append(QualityIssue(
            severity="error",
            category="price_integrity",
            message=f"High < Open or Close in {invalid_high} rows",
            affected_rows=invalid_high,
        ))

    # Low <= Open and Close
    invalid_low = df.filter(
        (pl.col(low_col) > pl.col(open_col)) | (pl.col(low_col) > pl.col(close_col))
    ).height
    if invalid_low > 0:
        issues.append(QualityIssue(
            severity="error",
            category="price_integrity",
            message=f"Low > Open or Close in {invalid_low} rows",
            affected_rows=invalid_low,
        ))

    # Negative prices
    for col in required_cols:
        negative_count = df.filter(pl.col(col) < 0).height
        if negative_count > 0:
            issues.append(QualityIssue(
                severity="error",
                category="invalid_value",
                message=f"Negative values in {col}: {negative_count} rows",
                affected_rows=negative_count,
            ))

    # Zero prices (suspicious)
    for col in [open_col, close_col]:
        zero_count = df.filter(pl.col(col) == 0).height
        if zero_count > 0:
            issues.append(QualityIssue(
                severity="warning",
                category="suspicious_value",
                message=f"Zero values in {col}: {zero_count} rows",
                affected_rows=zero_count,
            ))

    # Volume checks if present
    if volume_col in df.columns:
        negative_vol = df.filter(pl.col(volume_col) < 0).height
        if negative_vol > 0:
            issues.append(QualityIssue(
                severity="error",
                category="invalid_value",
                message=f"Negative volume: {negative_vol} rows",
                affected_rows=negative_vol,
            ))

        zero_vol = df.filter(pl.col(volume_col) == 0).height
        if zero_vol > 0:
            pct = zero_vol / df.height * 100
            issues.append(QualityIssue(
                severity="info" if pct < 5 else "warning",
                category="zero_volume",
                message=f"Zero volume: {zero_vol} rows ({pct:.1f}%)",
                affected_rows=zero_vol,
            ))

    return issues


def validate_returns(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    max_daily_return: float = 0.5,
    min_daily_return: float = -0.5,
) -> List[QualityIssue]:
    """Validate return data for outliers and anomalies.

    Checks for extreme returns that may indicate data errors, stock splits,
    or other corporate actions that need attention.

    Args:
        df: DataFrame containing return data.
        return_col: Column name for return values.
        max_daily_return: Maximum expected daily return (default: 50%).
        min_daily_return: Minimum expected daily return (default: -50%).

    Returns:
        List of QualityIssue objects for any anomalies found.

    Example:
        >>> df = pl.DataFrame({"return_1d": [0.01, -0.02, 2.5, -0.01]})
        >>> issues = validate_returns(df)  # Will flag 2.5 as extreme
    """
    issues: List[QualityIssue] = []

    if return_col not in df.columns:
        issues.append(QualityIssue(
            severity="error",
            category="schema",
            message=f"Missing required column: {return_col}",
        ))
        return issues

    returns = df[return_col].drop_nulls()

    # Check for extreme positive returns
    extreme_up = returns.filter(returns > max_daily_return)
    if extreme_up.len() > 0:
        issues.append(QualityIssue(
            severity="warning",
            category="extreme_return",
            message=f"Extreme positive returns (>{max_daily_return:.0%}): {extreme_up.len()} rows",
            affected_rows=extreme_up.len(),
            details={"max_value": float(extreme_up.max()) if extreme_up.max() else None},
        ))

    # Check for extreme negative returns
    extreme_down = returns.filter(returns < min_daily_return)
    if extreme_down.len() > 0:
        issues.append(QualityIssue(
            severity="warning",
            category="extreme_return",
            message=f"Extreme negative returns (<{min_daily_return:.0%}): {extreme_down.len()} rows",
            affected_rows=extreme_down.len(),
            details={"min_value": float(extreme_down.min()) if extreme_down.min() else None},
        ))

    # Check for null returns
    null_count = df.filter(pl.col(return_col).is_null()).height
    if null_count > 0:
        issues.append(QualityIssue(
            severity="warning",
            category="missing_data",
            message=f"Null return values: {null_count} rows",
            affected_rows=null_count,
        ))

    # Check for infinite values
    inf_count = df.filter(pl.col(return_col).is_infinite()).height
    if inf_count > 0:
        issues.append(QualityIssue(
            severity="error",
            category="invalid_value",
            message=f"Infinite return values: {inf_count} rows",
            affected_rows=inf_count,
        ))

    return issues


def check_missing_dates(
    df: pl.DataFrame,
    date_col: str = "date",
    asset_col: Optional[str] = None,
    expected_frequency: str = "daily",
) -> List[QualityIssue]:
    """Identify gaps in time series data.

    Detects missing dates that could indicate data collection issues
    or corporate events. Accounts for weekends and common market holidays.

    Args:
        df: DataFrame containing time series data.
        date_col: Column name for dates.
        asset_col: Optional column to check gaps per asset.
        expected_frequency: Expected data frequency ('daily', 'hourly').

    Returns:
        List of QualityIssue objects for any date gaps found.

    Example:
        >>> df = pl.DataFrame({
        ...     "date": [date(2024, 1, 1), date(2024, 1, 5)],  # Gap Jan 2-4
        ...     "close": [100, 105]
        ... })
        >>> issues = check_missing_dates(df)
    """
    issues: List[QualityIssue] = []

    if date_col not in df.columns:
        issues.append(QualityIssue(
            severity="error",
            category="schema",
            message=f"Missing required column: {date_col}",
        ))
        return issues

    def find_gaps(sub_df: pl.DataFrame) -> List[Tuple[date, date]]:
        """Find date gaps in a dataframe."""
        dates = sub_df[date_col].unique().sort()
        if dates.len() < 2:
            return []

        gaps = []
        date_list = dates.to_list()

        for i in range(1, len(date_list)):
            prev_date = date_list[i - 1]
            curr_date = date_list[i]

            if isinstance(prev_date, str):
                prev_date = date.fromisoformat(prev_date)
            if isinstance(curr_date, str):
                curr_date = date.fromisoformat(curr_date)

            expected_next = prev_date + timedelta(days=1)

            # Skip weekends
            while expected_next.weekday() >= 5:  # Saturday=5, Sunday=6
                expected_next = expected_next + timedelta(days=1)

            if curr_date > expected_next + timedelta(days=2):  # Allow for holidays
                gaps.append((prev_date, curr_date))

        return gaps

    if asset_col and asset_col in df.columns:
        all_gaps: Dict[str, List[Tuple[date, date]]] = {}
        for asset in df[asset_col].unique().to_list():
            asset_df = df.filter(pl.col(asset_col) == asset)
            gaps = find_gaps(asset_df)
            if gaps:
                all_gaps[str(asset)] = gaps

        if all_gaps:
            total_gaps = sum(len(g) for g in all_gaps.values())
            issues.append(QualityIssue(
                severity="warning",
                category="missing_dates",
                message=f"Found {total_gaps} date gaps across {len(all_gaps)} assets",
                affected_rows=total_gaps,
                details={"gaps_by_asset": {k: [(str(g[0]), str(g[1])) for g in v]
                                           for k, v in list(all_gaps.items())[:5]}},
            ))
    else:
        gaps = find_gaps(df)
        if gaps:
            issues.append(QualityIssue(
                severity="warning",
                category="missing_dates",
                message=f"Found {len(gaps)} date gaps in time series",
                affected_rows=len(gaps),
                details={"sample_gaps": [(str(g[0]), str(g[1])) for g in gaps[:5]]},
            ))

    return issues


def detect_outliers(
    df: pl.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5,
) -> List[QualityIssue]:
    """Detect statistical outliers in specified columns.

    Uses IQR (Interquartile Range) or Z-score methods to identify
    potential outliers that may indicate data quality issues.

    Args:
        df: DataFrame containing numeric data.
        columns: List of column names to check for outliers.
        method: Detection method ('iqr' or 'zscore'). Defaults to 'iqr'.
        threshold: Sensitivity threshold. For IQR, multiplier for IQR
            (default 1.5). For Z-score, number of standard deviations.

    Returns:
        List of QualityIssue objects for any outliers detected.

    Example:
        >>> df = pl.DataFrame({"price": [100, 102, 101, 500, 99]})
        >>> issues = detect_outliers(df, ["price"])
    """
    issues: List[QualityIssue] = []

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].drop_nulls()
        if series.len() == 0:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            if q1 is None or q3 is None:
                continue

            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = df.filter(
                (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
            )

        else:  # zscore
            mean = series.mean()
            std = series.std()
            if mean is None or std is None or std == 0:
                continue

            outliers = df.filter(
                ((pl.col(col) - mean) / std).abs() > threshold
            )

        if outliers.height > 0:
            pct = outliers.height / df.height * 100
            issues.append(QualityIssue(
                severity="info" if pct < 1 else "warning",
                category="outlier",
                message=f"Outliers in {col}: {outliers.height} rows ({pct:.2f}%)",
                affected_rows=outliers.height,
                details={
                    "method": method,
                    "threshold": threshold,
                },
            ))

    return issues


def generate_quality_report(
    df: pl.DataFrame,
    date_col: str = "date",
    asset_col: Optional[str] = None,
    ohlcv_cols: Optional[Dict[str, str]] = None,
    return_col: str = "return_1d",
) -> Dict[str, Any]:
    """Generate comprehensive data quality assessment report.

    Runs all validation checks and produces a summary report with
    an overall quality score and detailed issue breakdown.

    Args:
        df: DataFrame to assess.
        date_col: Column name for dates.
        asset_col: Optional column for asset identifiers.
        ohlcv_cols: Dict mapping OHLCV columns (keys: open, high, low, close, volume).
        return_col: Column name for return data.

    Returns:
        Dictionary containing:
        - quality_score: Overall score (0-1)
        - total_rows: Number of rows analyzed
        - issues: List of all QualityIssue objects
        - summary: Counts by severity
        - recommendations: Suggested actions

    Example:
        >>> df = pl.read_parquet("data/bronze/market.parquet")
        >>> report = generate_quality_report(df)
        >>> print(f"Score: {report['quality_score']:.1%}")
    """
    all_issues: List[QualityIssue] = []

    # Run OHLCV validation if columns exist
    if ohlcv_cols:
        all_issues.extend(validate_ohlcv(df, **ohlcv_cols))
    elif all(c in df.columns for c in ["open", "high", "low", "close"]):
        all_issues.extend(validate_ohlcv(df))

    # Run return validation
    if return_col in df.columns:
        all_issues.extend(validate_returns(df, return_col))

    # Check for missing dates
    if date_col in df.columns:
        all_issues.extend(check_missing_dates(df, date_col, asset_col))

    # Detect outliers in numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    if numeric_cols:
        all_issues.extend(detect_outliers(df, numeric_cols[:10]))  # Limit to first 10

    # Calculate quality score
    error_count = sum(1 for i in all_issues if i.severity == "error")
    warning_count = sum(1 for i in all_issues if i.severity == "warning")
    info_count = sum(1 for i in all_issues if i.severity == "info")

    # Weighted score (errors are 10x warnings, warnings are 2x info)
    penalty = error_count * 0.1 + warning_count * 0.02 + info_count * 0.005
    quality_score = max(0, 1 - penalty)

    # Generate recommendations
    recommendations: List[str] = []
    categories: Set[str] = {i.category for i in all_issues}

    if "price_integrity" in categories:
        recommendations.append("Review and clean OHLCV data - check for data provider issues")
    if "extreme_return" in categories:
        recommendations.append("Investigate extreme returns - may indicate splits or data errors")
    if "missing_dates" in categories:
        recommendations.append("Fill date gaps or adjust for market holidays")
    if "outlier" in categories:
        recommendations.append("Consider outlier treatment (winsorization or removal)")

    return {
        "quality_score": quality_score,
        "total_rows": df.height,
        "total_columns": len(df.columns),
        "issues": [
            {
                "severity": i.severity,
                "category": i.category,
                "message": i.message,
                "affected_rows": i.affected_rows,
                "details": i.details,
            }
            for i in all_issues
        ],
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "total_issues": len(all_issues),
        },
        "recommendations": recommendations,
    }
