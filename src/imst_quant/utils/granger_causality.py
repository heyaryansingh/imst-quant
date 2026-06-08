"""Granger causality testing for identifying lead-lag relationships.

This module provides tools for testing Granger causality between time series,
which helps identify predictive relationships between assets, signals, and
market variables. Particularly useful for sentiment-price lead-lag analysis.

Functions:
    granger_causality_test: Test if one series Granger-causes another
    bidirectional_granger_test: Test causality in both directions
    pairwise_granger_matrix: Test all pairs in a DataFrame
    sentiment_price_causality: Specialized sentiment-returns analysis
    causality_summary: Summarize test results

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.granger_causality import granger_causality_test
    >>> # Test if sentiment Granger-causes returns
    >>> sentiment = pl.Series("sentiment", [0.1, 0.2, 0.15, 0.3, 0.25, ...])
    >>> returns = pl.Series("returns", [0.01, 0.02, 0.015, 0.025, 0.02, ...])
    >>> results = granger_causality_test(sentiment, returns, max_lag=5)
    >>> for r in results:
    ...     if r.is_significant:
    ...         print(f"Lag {r.lag}: F={r.f_statistic:.2f}, p={r.p_value:.4f}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class GrangerResult:
    """Result from a Granger causality test.

    Attributes:
        asset_pair: Tuple of (cause, effect) series names.
        lag: Number of lags tested.
        f_statistic: F-test statistic comparing restricted vs unrestricted models.
        p_value: P-value for the F-test (lower = more significant).
        is_significant: Whether the test is significant at the given level.
        direction: String describing causality direction (e.g., "X -> Y").
    """

    asset_pair: Tuple[str, str]
    lag: int
    f_statistic: float
    p_value: float
    is_significant: bool
    direction: str


def _prepare_series(series: pl.Series) -> np.ndarray:
    """Prepare a polars Series for regression by handling nulls and converting to numpy.

    Args:
        series: Input polars Series.

    Returns:
        Numpy array with nulls removed.
    """
    return series.drop_nulls().to_numpy().astype(np.float64)


def _create_lagged_matrix(
    y: np.ndarray,
    x: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lagged regressor matrices for Granger causality test.

    Args:
        y: Dependent variable array.
        x: Potential causal variable array.
        lag: Number of lags to include.

    Returns:
        Tuple of (y_current, y_lags, x_lags) aligned arrays.
    """
    n = len(y)
    if n <= lag:
        return np.array([]), np.array([]), np.array([])

    # Create lagged values
    y_current = y[lag:]
    y_lags = np.column_stack([y[lag - i - 1 : n - i - 1] for i in range(lag)])
    x_lags = np.column_stack([x[lag - i - 1 : n - i - 1] for i in range(lag)])

    return y_current, y_lags, x_lags


def _ols_sse(y: np.ndarray, X: np.ndarray) -> float:
    """Compute sum of squared errors from OLS regression.

    Args:
        y: Dependent variable.
        X: Design matrix (regressors).

    Returns:
        Sum of squared errors (residuals).
    """
    if len(y) == 0 or X.shape[0] == 0:
        return float("inf")

    try:
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(y)), X])

        # OLS: beta = (X'X)^-1 X'y
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

        # Calculate residuals manually if not returned
        if len(residuals) == 0:
            y_pred = X_with_const @ coeffs
            sse = np.sum((y - y_pred) ** 2)
        else:
            sse = residuals[0] if len(residuals) > 0 else np.sum((y - X_with_const @ coeffs) ** 2)

        return float(sse)
    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


def granger_causality_test(
    x: pl.Series,
    y: pl.Series,
    max_lag: int = 5,
    significance: float = 0.05,
) -> List[GrangerResult]:
    """Test if x Granger-causes y using OLS regression and F-test.

    Granger causality tests whether past values of x help predict y beyond
    what past values of y alone can predict. This is done by comparing:
    - Restricted model: y_t = c + sum(a_i * y_{t-i})
    - Unrestricted model: y_t = c + sum(a_i * y_{t-i}) + sum(b_i * x_{t-i})

    The F-statistic tests if the x_lag coefficients are jointly zero.

    Args:
        x: Potential causal series (the "cause").
        y: Dependent series (the "effect").
        max_lag: Maximum number of lags to test (tests each lag 1..max_lag).
        significance: Significance level for hypothesis testing.

    Returns:
        List of GrangerResult for each lag from 1 to max_lag.

    Example:
        >>> sentiment = pl.Series("sentiment", [0.1, 0.2, 0.15, 0.3, 0.25])
        >>> returns = pl.Series("returns", [0.01, 0.02, 0.015, 0.025, 0.02])
        >>> results = granger_causality_test(sentiment, returns, max_lag=2)
    """
    results: List[GrangerResult] = []

    # Get series names
    x_name = x.name if x.name else "X"
    y_name = y.name if y.name else "Y"

    # Convert to numpy
    x_arr = _prepare_series(x)
    y_arr = _prepare_series(y)

    # Align series lengths
    min_len = min(len(x_arr), len(y_arr))
    if min_len < max_lag + 2:
        # Insufficient data for any meaningful test
        for lag in range(1, max_lag + 1):
            results.append(
                GrangerResult(
                    asset_pair=(x_name, y_name),
                    lag=lag,
                    f_statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    direction=f"{x_name} -> {y_name}",
                )
            )
        return results

    x_arr = x_arr[:min_len]
    y_arr = y_arr[:min_len]

    # Check for constant series (no variation)
    if np.std(x_arr) < 1e-10 or np.std(y_arr) < 1e-10:
        for lag in range(1, max_lag + 1):
            results.append(
                GrangerResult(
                    asset_pair=(x_name, y_name),
                    lag=lag,
                    f_statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    direction=f"{x_name} -> {y_name}",
                )
            )
        return results

    # Test each lag
    for lag in range(1, max_lag + 1):
        y_current, y_lags, x_lags = _create_lagged_matrix(y_arr, x_arr, lag)

        n = len(y_current)
        if n < 2 * lag + 2:
            # Not enough observations for this lag
            results.append(
                GrangerResult(
                    asset_pair=(x_name, y_name),
                    lag=lag,
                    f_statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    direction=f"{x_name} -> {y_name}",
                )
            )
            continue

        # Restricted model: y ~ y_lags only
        sse_restricted = _ols_sse(y_current, y_lags)

        # Unrestricted model: y ~ y_lags + x_lags
        X_unrestricted = np.column_stack([y_lags, x_lags])
        sse_unrestricted = _ols_sse(y_current, X_unrestricted)

        # F-test
        # F = ((SSE_r - SSE_u) / q) / (SSE_u / (n - k))
        # where q = number of restrictions (lag), k = number of params in unrestricted
        q = lag  # Number of x_lag parameters
        k = 2 * lag + 1  # Unrestricted params: constant + lag y's + lag x's

        if sse_unrestricted <= 0 or sse_unrestricted >= sse_restricted:
            # No improvement or numerical issues
            f_stat = 0.0
            p_value = 1.0
        else:
            denominator = sse_unrestricted / (n - k)
            if denominator <= 0:
                f_stat = 0.0
                p_value = 1.0
            else:
                f_stat = ((sse_restricted - sse_unrestricted) / q) / denominator
                # P-value from F-distribution
                try:
                    p_value = 1.0 - stats.f.cdf(f_stat, q, n - k)
                except (ValueError, RuntimeWarning):
                    p_value = 1.0

        is_significant = p_value < significance

        results.append(
            GrangerResult(
                asset_pair=(x_name, y_name),
                lag=lag,
                f_statistic=float(f_stat),
                p_value=float(p_value),
                is_significant=is_significant,
                direction=f"{x_name} -> {y_name}",
            )
        )

    return results


def bidirectional_granger_test(
    x: pl.Series,
    y: pl.Series,
    x_name: str = "X",
    y_name: str = "Y",
    max_lag: int = 5,
    significance: float = 0.05,
) -> Dict:
    """Test Granger causality in both directions between two series.

    This function tests both x -> y and y -> x causality, identifies
    feedback loops, and determines the dominant causal direction.

    Args:
        x: First time series.
        y: Second time series.
        x_name: Name for the first series (for labeling).
        y_name: Name for the second series (for labeling).
        max_lag: Maximum number of lags to test.
        significance: Significance level for hypothesis testing.

    Returns:
        Dictionary containing:
        - x_causes_y: List of GrangerResult for x -> y tests.
        - y_causes_x: List of GrangerResult for y -> x tests.
        - feedback: Boolean indicating bidirectional causality.
        - dominant_direction: String describing the dominant direction.
        - best_lag_x_to_y: Lag with lowest p-value for x -> y.
        - best_lag_y_to_x: Lag with lowest p-value for y -> x.

    Example:
        >>> result = bidirectional_granger_test(sentiment, returns)
        >>> if result["feedback"]:
        ...     print("Bidirectional causality detected!")
        >>> print(f"Dominant: {result['dominant_direction']}")
    """
    # Rename series for consistent naming
    x_renamed = x.alias(x_name)
    y_renamed = y.alias(y_name)

    # Test x -> y
    x_causes_y = granger_causality_test(x_renamed, y_renamed, max_lag, significance)

    # Test y -> x
    y_causes_x = granger_causality_test(y_renamed, x_renamed, max_lag, significance)

    # Check for significant causality in each direction
    x_to_y_significant = any(r.is_significant for r in x_causes_y)
    y_to_x_significant = any(r.is_significant for r in y_causes_x)

    # Feedback loop exists if both directions are significant
    feedback = x_to_y_significant and y_to_x_significant

    # Find best lags (lowest p-value)
    best_x_to_y = min(x_causes_y, key=lambda r: r.p_value) if x_causes_y else None
    best_y_to_x = min(y_causes_x, key=lambda r: r.p_value) if y_causes_x else None

    # Determine dominant direction
    if not x_to_y_significant and not y_to_x_significant:
        dominant_direction = "no_causality"
    elif x_to_y_significant and not y_to_x_significant:
        dominant_direction = f"{x_name} -> {y_name}"
    elif y_to_x_significant and not x_to_y_significant:
        dominant_direction = f"{y_name} -> {x_name}"
    else:
        # Both significant - compare p-values
        if best_x_to_y and best_y_to_x:
            if best_x_to_y.p_value < best_y_to_x.p_value:
                dominant_direction = f"{x_name} -> {y_name} (stronger)"
            elif best_y_to_x.p_value < best_x_to_y.p_value:
                dominant_direction = f"{y_name} -> {x_name} (stronger)"
            else:
                dominant_direction = "bidirectional_equal"
        else:
            dominant_direction = "bidirectional"

    return {
        "x_causes_y": x_causes_y,
        "y_causes_x": y_causes_x,
        "feedback": feedback,
        "dominant_direction": dominant_direction,
        "best_lag_x_to_y": best_x_to_y.lag if best_x_to_y else None,
        "best_lag_y_to_x": best_y_to_x.lag if best_y_to_x else None,
        "min_p_x_to_y": best_x_to_y.p_value if best_x_to_y else 1.0,
        "min_p_y_to_x": best_y_to_x.p_value if best_y_to_x else 1.0,
    }


def pairwise_granger_matrix(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    max_lag: int = 5,
    significance: float = 0.05,
) -> Dict:
    """Test Granger causality for all column pairs in a DataFrame.

    Performs pairwise Granger causality tests between all specified columns,
    building a causality matrix similar to a correlation matrix.

    Args:
        df: DataFrame with time series data in columns.
        columns: List of column names to test. If None, uses all numeric columns.
        max_lag: Maximum number of lags to test.
        significance: Significance level for hypothesis testing.

    Returns:
        Dictionary containing:
        - results: List of all GrangerResult objects.
        - causality_matrix: Dict of dicts with minimum p-values.
        - significant_pairs: List of (cause, effect, best_lag, p_value) tuples.
        - num_tests: Total number of tests performed.
        - num_significant: Number of significant results.

    Example:
        >>> df = pl.DataFrame({
        ...     "asset_a": [0.01, 0.02, 0.015, ...],
        ...     "asset_b": [0.02, 0.01, 0.025, ...],
        ...     "sentiment": [0.1, 0.2, 0.15, ...],
        ... })
        >>> result = pairwise_granger_matrix(df, max_lag=3)
        >>> print(result["significant_pairs"])
    """
    # Select columns to test
    if columns is None:
        columns = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    if len(columns) < 2:
        return {
            "results": [],
            "causality_matrix": {},
            "significant_pairs": [],
            "num_tests": 0,
            "num_significant": 0,
        }

    all_results: List[GrangerResult] = []
    causality_matrix: Dict[str, Dict[str, float]] = {col: {} for col in columns}
    significant_pairs: List[Tuple[str, str, int, float]] = []

    # Test all pairs (directed, so test both A->B and B->A)
    for i, col_x in enumerate(columns):
        for j, col_y in enumerate(columns):
            if i == j:
                causality_matrix[col_x][col_y] = 1.0  # Self-causality
                continue

            x = df[col_x]
            y = df[col_y]

            results = granger_causality_test(x, y, max_lag, significance)
            all_results.extend(results)

            # Find minimum p-value across all lags
            if results:
                best_result = min(results, key=lambda r: r.p_value)
                causality_matrix[col_x][col_y] = best_result.p_value

                if best_result.is_significant:
                    significant_pairs.append(
                        (col_x, col_y, best_result.lag, best_result.p_value)
                    )

    # Sort significant pairs by p-value
    significant_pairs.sort(key=lambda x: x[3])

    return {
        "results": all_results,
        "causality_matrix": causality_matrix,
        "significant_pairs": significant_pairs,
        "num_tests": len(all_results),
        "num_significant": sum(1 for r in all_results if r.is_significant),
    }


def sentiment_price_causality(
    sentiment: pl.Series,
    returns: pl.Series,
    max_lag: int = 10,
    significance: float = 0.05,
) -> Dict:
    """Analyze lead-lag relationships between sentiment and price returns.

    This specialized function tests the predictive relationship between
    sentiment signals and price movements, which is crucial for sentiment-
    based trading strategies.

    Args:
        sentiment: Sentiment signal series (e.g., aggregated polarity scores).
        returns: Price return series (e.g., daily log returns).
        max_lag: Maximum lag to test (in same time units as data).
        significance: Significance level for hypothesis testing.

    Returns:
        Dictionary containing:
        - sentiment_causes_returns: Boolean if sentiment predicts returns.
        - returns_cause_sentiment: Boolean if returns predict sentiment.
        - optimal_lag: Best lag for sentiment -> returns (if significant).
        - lead_lag_relationship: Human-readable description.
        - sentiment_to_returns: List of GrangerResult.
        - returns_to_sentiment: List of GrangerResult.
        - strongest_f_statistic: Highest F-statistic found.
        - recommendation: Trading recommendation based on results.

    Example:
        >>> result = sentiment_price_causality(sentiment, returns, max_lag=5)
        >>> print(result["lead_lag_relationship"])
        >>> if result["sentiment_causes_returns"]:
        ...     print(f"Use sentiment with {result['optimal_lag']} lag")
    """
    # Rename series for clarity
    sentiment_named = sentiment.alias("sentiment")
    returns_named = returns.alias("returns")

    # Test sentiment -> returns
    sent_to_ret = granger_causality_test(
        sentiment_named, returns_named, max_lag, significance
    )

    # Test returns -> sentiment
    ret_to_sent = granger_causality_test(
        returns_named, sentiment_named, max_lag, significance
    )

    # Analyze results
    sentiment_causes_returns = any(r.is_significant for r in sent_to_ret)
    returns_cause_sentiment = any(r.is_significant for r in ret_to_sent)

    # Find optimal lag for sentiment -> returns
    optimal_lag = None
    strongest_f = 0.0

    if sent_to_ret:
        best_sent_to_ret = min(sent_to_ret, key=lambda r: r.p_value)
        if best_sent_to_ret.is_significant:
            optimal_lag = best_sent_to_ret.lag
            strongest_f = best_sent_to_ret.f_statistic

    # Also check returns -> sentiment for strongest F
    if ret_to_sent:
        best_ret_to_sent = min(ret_to_sent, key=lambda r: r.p_value)
        if best_ret_to_sent.f_statistic > strongest_f:
            strongest_f = best_ret_to_sent.f_statistic

    # Determine lead-lag relationship
    if sentiment_causes_returns and not returns_cause_sentiment:
        lead_lag_relationship = (
            f"Sentiment leads returns by ~{optimal_lag} periods. "
            "Sentiment signals may predict future price movements."
        )
        recommendation = (
            f"Consider using sentiment signals with a {optimal_lag}-period "
            "lookahead for trade entry decisions."
        )
    elif returns_cause_sentiment and not sentiment_causes_returns:
        best_ret = min(ret_to_sent, key=lambda r: r.p_value)
        lead_lag_relationship = (
            f"Returns lead sentiment by ~{best_ret.lag} periods. "
            "Sentiment appears to react to price movements."
        )
        recommendation = (
            "Sentiment is lagging indicator. Consider momentum-based strategies "
            "instead of sentiment-based entry signals."
        )
    elif sentiment_causes_returns and returns_cause_sentiment:
        lead_lag_relationship = (
            "Bidirectional causality detected. Sentiment and returns "
            "influence each other in a feedback loop."
        )
        recommendation = (
            "Use sentiment as confirming indicator rather than primary signal. "
            f"Sentiment has stronger predictive power at lag {optimal_lag}."
            if optimal_lag
            else "Both signals provide information; combine for better accuracy."
        )
    else:
        lead_lag_relationship = (
            "No significant Granger causality detected between sentiment "
            "and returns at the tested lags."
        )
        recommendation = (
            "Sentiment may not be predictive at these frequencies. "
            "Consider different aggregation windows or additional features."
        )

    return {
        "sentiment_causes_returns": sentiment_causes_returns,
        "returns_cause_sentiment": returns_cause_sentiment,
        "optimal_lag": optimal_lag,
        "lead_lag_relationship": lead_lag_relationship,
        "sentiment_to_returns": sent_to_ret,
        "returns_to_sentiment": ret_to_sent,
        "strongest_f_statistic": float(strongest_f),
        "recommendation": recommendation,
    }


def causality_summary(results: List[GrangerResult]) -> Dict:
    """Generate summary statistics from a list of Granger causality results.

    Args:
        results: List of GrangerResult objects from causality tests.

    Returns:
        Dictionary containing:
        - total_tests: Total number of tests.
        - significant_count: Number of significant results.
        - significant_pct: Percentage of significant results.
        - strongest_pair: The pair with lowest p-value.
        - weakest_p_value: Lowest p-value found.
        - avg_f_statistic: Average F-statistic across all tests.
        - significant_lags: Distribution of significant lags.

    Example:
        >>> results = granger_causality_test(x, y, max_lag=5)
        >>> summary = causality_summary(results)
        >>> print(f"{summary['significant_pct']:.1%} of tests significant")
    """
    if not results:
        return {
            "total_tests": 0,
            "significant_count": 0,
            "significant_pct": 0.0,
            "strongest_pair": None,
            "weakest_p_value": 1.0,
            "avg_f_statistic": 0.0,
            "significant_lags": {},
        }

    total_tests = len(results)
    significant_results = [r for r in results if r.is_significant]
    significant_count = len(significant_results)
    significant_pct = significant_count / total_tests if total_tests > 0 else 0.0

    # Find strongest result (lowest p-value)
    best_result = min(results, key=lambda r: r.p_value)
    strongest_pair = {
        "cause": best_result.asset_pair[0],
        "effect": best_result.asset_pair[1],
        "lag": best_result.lag,
        "f_statistic": best_result.f_statistic,
        "p_value": best_result.p_value,
    }

    # Average F-statistic
    f_stats = [r.f_statistic for r in results if r.f_statistic > 0]
    avg_f_statistic = float(np.mean(f_stats)) if f_stats else 0.0

    # Distribution of significant lags
    significant_lags: Dict[int, int] = {}
    for r in significant_results:
        significant_lags[r.lag] = significant_lags.get(r.lag, 0) + 1

    return {
        "total_tests": total_tests,
        "significant_count": significant_count,
        "significant_pct": significant_pct,
        "strongest_pair": strongest_pair,
        "weakest_p_value": best_result.p_value,
        "avg_f_statistic": avg_f_statistic,
        "significant_lags": significant_lags,
    }
