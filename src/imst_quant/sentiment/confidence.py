"""Sentiment confidence scoring for improved signal quality.

This module provides confidence metrics for sentiment analysis to help
filter low-quality signals and improve trading decision reliability.

Confidence factors:
1. Model agreement - multiple sentiment models agreeing
2. Sample size - sufficient data points for reliability
3. Author credibility - influence scores and post history
4. Text quality - length, coherence, and information content
5. Temporal consistency - sentiment stability over time

Example:
    >>> from imst_quant.sentiment.confidence import calculate_confidence_score
    >>> confidence = calculate_confidence_score(
    ...     sentiment_df,
    ...     min_samples=10,
    ...     credibility_weight=0.3
    ... )
    >>> high_confidence = confidence.filter(pl.col("confidence") > 0.7)
"""

from typing import Dict, Optional

import polars as pl
import numpy as np

import structlog

logger = structlog.get_logger()


def calculate_text_quality_score(
    df: pl.DataFrame,
    text_col: str = "text",
    min_length: int = 50,
    max_length: int = 5000,
) -> pl.DataFrame:
    """Calculate text quality score based on content characteristics.

    Args:
        df: DataFrame with text content.
        text_col: Name of text column (default: "text").
        min_length: Minimum characters for quality text (default: 50).
        max_length: Maximum characters before penalizing (default: 5000).

    Returns:
        DataFrame with additional 'text_quality' column (0-1 scale).

    Example:
        >>> df = pl.DataFrame({"text": ["Short", "This is a well-written analysis..."]})
        >>> result = calculate_text_quality_score(df)
        >>> print(result["text_quality"])
    """
    # Calculate text length
    df = df.with_columns([
        pl.col(text_col).str.len_chars().alias("text_length"),
    ])

    # Quality score based on length (bell curve centered at optimal length)
    optimal_length = (min_length + max_length) / 2
    df = df.with_columns([
        pl.when(pl.col("text_length") < min_length)
        .then(pl.col("text_length") / min_length)
        .when(pl.col("text_length") > max_length)
        .then(max_length / pl.col("text_length"))
        .otherwise(
            1.0 - ((pl.col("text_length") - optimal_length).abs() / optimal_length) * 0.3
        )
        .clip(0, 1)
        .alias("text_quality"),
    ])

    return df


def calculate_sample_size_confidence(
    df: pl.DataFrame,
    group_cols: list[str],
    min_samples: int = 5,
    optimal_samples: int = 30,
) -> pl.DataFrame:
    """Calculate confidence score based on sample size per group.

    Args:
        df: DataFrame with sentiment data.
        group_cols: Columns to group by (e.g., ["asset_id", "date"]).
        min_samples: Minimum samples for any confidence (default: 5).
        optimal_samples: Sample size for maximum confidence (default: 30).

    Returns:
        DataFrame with 'sample_size_confidence' column (0-1 scale).

    Example:
        >>> df = pl.DataFrame({
        ...     "asset_id": ["AAPL"] * 50,
        ...     "sentiment": [0.5] * 50
        ... })
        >>> result = calculate_sample_size_confidence(df, ["asset_id"], min_samples=10)
        >>> print(result["sample_size_confidence"].mean())
    """
    # Count samples per group
    sample_counts = df.group_by(group_cols).agg([
        pl.len().alias("sample_count"),
    ])

    # Calculate confidence based on sample size
    sample_counts = sample_counts.with_columns([
        pl.when(pl.col("sample_count") < min_samples)
        .then(pl.col("sample_count") / min_samples * 0.5)  # Low confidence
        .when(pl.col("sample_count") >= optimal_samples)
        .then(1.0)  # Full confidence
        .otherwise(
            0.5 + (pl.col("sample_count") - min_samples) / (optimal_samples - min_samples) * 0.5
        )
        .clip(0, 1)
        .alias("sample_size_confidence"),
    ])

    # Join back to original dataframe
    df = df.join(sample_counts, on=group_cols, how="left")

    return df


def calculate_author_credibility_score(
    df: pl.DataFrame,
    influence_scores: Optional[Dict[str, float]] = None,
    author_col: str = "author_id",
) -> pl.DataFrame:
    """Calculate author credibility based on influence scores and activity.

    Args:
        df: DataFrame with author information.
        influence_scores: Optional dict mapping author_id to influence score.
        author_col: Name of author column (default: "author_id").

    Returns:
        DataFrame with 'author_credibility' column (0-1 scale).

    Example:
        >>> df = pl.DataFrame({"author_id": ["user1", "user2"]})
        >>> influence = {"user1": 2.5, "user2": 0.8}
        >>> result = calculate_author_credibility_score(df, influence)
        >>> print(result["author_credibility"])
    """
    if influence_scores is None:
        # Default to equal credibility
        df = df.with_columns([
            pl.lit(0.5).alias("author_credibility"),
        ])
        return df

    # Create mapping DataFrame
    influence_df = pl.DataFrame({
        author_col: list(influence_scores.keys()),
        "raw_influence": list(influence_scores.values()),
    })

    # Normalize influence scores to 0-1 range
    max_influence = max(influence_scores.values()) if influence_scores else 1.0
    min_influence = min(influence_scores.values()) if influence_scores else 0.0

    influence_df = influence_df.with_columns([
        ((pl.col("raw_influence") - min_influence) / (max_influence - min_influence)).alias("author_credibility"),
    ])

    # Join to main dataframe
    df = df.join(
        influence_df.select([author_col, "author_credibility"]),
        on=author_col,
        how="left",
    ).with_columns([
        pl.col("author_credibility").fill_null(0.3),  # Default for unknown authors
    ])

    return df


def calculate_temporal_consistency(
    df: pl.DataFrame,
    group_cols: list[str],
    sentiment_col: str = "polarity",
    window_size: int = 7,
) -> pl.DataFrame:
    """Calculate temporal consistency of sentiment over time.

    More stable sentiment over time indicates higher confidence.

    Args:
        df: DataFrame with time-series sentiment data.
        group_cols: Columns to group by (e.g., ["asset_id"]).
        sentiment_col: Name of sentiment column (default: "polarity").
        window_size: Rolling window size for consistency calculation (default: 7).

    Returns:
        DataFrame with 'temporal_consistency' column (0-1 scale).

    Example:
        >>> df = pl.DataFrame({
        ...     "asset_id": ["AAPL"] * 10,
        ...     "date": pl.date_range(start="2024-01-01", end="2024-01-10", interval="1d"),
        ...     "polarity": [0.5, 0.52, 0.51, 0.49, 0.50, 0.48, 0.51, 0.50, 0.49, 0.51]
        ... })
        >>> result = calculate_temporal_consistency(df, ["asset_id"])
        >>> print(result["temporal_consistency"].mean())
    """
    # Sort by group and date
    df = df.sort(group_cols + ["date"])

    # Calculate rolling std of sentiment
    df = df.with_columns([
        pl.col(sentiment_col)
        .rolling_std(window_size=window_size)
        .over(group_cols)
        .alias("sentiment_std"),
    ])

    # Convert std to consistency score (lower std = higher consistency)
    # Normalize assuming std typically ranges from 0 to 0.5
    df = df.with_columns([
        (1.0 - pl.col("sentiment_std").clip(0, 0.5) / 0.5).alias("temporal_consistency"),
    ])

    # Fill nulls (first few rows don't have rolling std)
    df = df.with_columns([
        pl.col("temporal_consistency").fill_null(0.5),
    ])

    return df


def calculate_model_agreement(
    df: pl.DataFrame,
    model_cols: list[str],
) -> pl.DataFrame:
    """Calculate agreement between multiple sentiment models.

    Args:
        df: DataFrame with sentiment scores from multiple models.
        model_cols: List of column names for different sentiment models
            (e.g., ["textblob_polarity", "finbert_polarity", "vader_polarity"]).

    Returns:
        DataFrame with 'model_agreement' column (0-1 scale).

    Example:
        >>> df = pl.DataFrame({
        ...     "textblob": [0.5, 0.6, -0.3],
        ...     "finbert": [0.52, 0.58, -0.28],
        ...     "vader": [0.48, 0.62, -0.35]
        ... })
        >>> result = calculate_model_agreement(df, ["textblob", "finbert", "vader"])
        >>> print(result["model_agreement"])
    """
    if len(model_cols) < 2:
        # Need at least 2 models for agreement
        df = df.with_columns([pl.lit(0.5).alias("model_agreement")])
        return df

    # Calculate pairwise correlation/agreement
    # Use coefficient of variation (std/mean) as inverse measure of agreement
    model_mean = sum(pl.col(col) for col in model_cols) / len(model_cols)
    model_std = pl.concat_list([pl.col(col) for col in model_cols]).list.eval(
        pl.element().std()
    ).list.first()

    df = df.with_columns([
        model_mean.alias("_model_mean"),
        model_std.alias("_model_std"),
    ])

    # Agreement score: higher when std is low relative to mean
    # Handle case where mean is close to 0
    df = df.with_columns([
        pl.when(pl.col("_model_mean").abs() < 0.01)
        .then(
            1.0 - pl.col("_model_std").clip(0, 0.5) / 0.5  # If sentiment near 0, just use std
        )
        .otherwise(
            1.0 - (pl.col("_model_std") / (pl.col("_model_mean").abs() + 0.1)).clip(0, 2) / 2
        )
        .clip(0, 1)
        .alias("model_agreement"),
    ])

    # Clean up temporary columns
    df = df.drop(["_model_mean", "_model_std"])

    return df


def calculate_confidence_score(
    df: pl.DataFrame,
    group_cols: list[str] = ["asset_id", "date"],
    sentiment_col: str = "polarity",
    author_col: str = "author_id",
    text_col: Optional[str] = "text",
    influence_scores: Optional[Dict[str, float]] = None,
    model_cols: Optional[list[str]] = None,
    min_samples: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Calculate comprehensive confidence score for sentiment analysis.

    Combines multiple confidence factors into a single weighted score:
    - Sample size confidence
    - Author credibility
    - Text quality (if text available)
    - Model agreement (if multiple models available)
    - Temporal consistency

    Args:
        df: DataFrame with sentiment data.
        group_cols: Columns to group by (default: ["asset_id", "date"]).
        sentiment_col: Name of sentiment column (default: "polarity").
        author_col: Name of author column (default: "author_id").
        text_col: Name of text column (default: "text"). Set to None if not available.
        influence_scores: Optional dict of author influence scores.
        model_cols: Optional list of sentiment model columns for agreement calculation.
        min_samples: Minimum samples for confidence (default: 5).
        weights: Custom weights for each factor. Default:
            {"sample_size": 0.25, "author": 0.25, "text": 0.15, "model": 0.20, "temporal": 0.15}

    Returns:
        DataFrame with confidence score column and individual factor columns.

    Example:
        >>> df = pl.read_parquet("data/sentiment/sentiment.parquet")
        >>> confidence_df = calculate_confidence_score(
        ...     df,
        ...     influence_scores=author_influence,
        ...     model_cols=["textblob_polarity", "finbert_polarity"],
        ...     min_samples=10
        ... )
        >>> high_conf = confidence_df.filter(pl.col("confidence_score") > 0.7)
        >>> print(f"High confidence signals: {len(high_conf)}")
    """
    if weights is None:
        weights = {
            "sample_size": 0.25,
            "author": 0.25,
            "text": 0.15,
            "model": 0.20,
            "temporal": 0.15,
        }

    # Calculate individual confidence factors
    logger.info("calculating_confidence_factors", group_cols=group_cols)

    # 1. Sample size confidence
    df = calculate_sample_size_confidence(df, group_cols, min_samples=min_samples)

    # 2. Author credibility
    if author_col in df.columns:
        df = calculate_author_credibility_score(df, influence_scores, author_col)
    else:
        df = df.with_columns([pl.lit(0.5).alias("author_credibility")])

    # 3. Text quality (if text column available)
    if text_col and text_col in df.columns:
        df = calculate_text_quality_score(df, text_col)
    else:
        df = df.with_columns([pl.lit(0.5).alias("text_quality")])
        weights["sample_size"] += weights["text"] / 2
        weights["author"] += weights["text"] / 2
        weights["text"] = 0

    # 4. Model agreement (if multiple models available)
    if model_cols and len(model_cols) >= 2:
        df = calculate_model_agreement(df, model_cols)
    else:
        df = df.with_columns([pl.lit(0.5).alias("model_agreement")])
        # Redistribute model agreement weight
        weights["sample_size"] += weights["model"] / 2
        weights["temporal"] += weights["model"] / 2
        weights["model"] = 0

    # 5. Temporal consistency
    if "date" in df.columns:
        df = calculate_temporal_consistency(df, [col for col in group_cols if col != "date"], sentiment_col)
    else:
        df = df.with_columns([pl.lit(0.5).alias("temporal_consistency")])
        # Redistribute temporal weight
        weights["sample_size"] += weights["temporal"] / 2
        weights["author"] += weights["temporal"] / 2
        weights["temporal"] = 0

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted confidence score
    confidence_expr = (
        pl.col("sample_size_confidence") * weights["sample_size"]
        + pl.col("author_credibility") * weights["author"]
        + pl.col("text_quality") * weights["text"]
        + pl.col("model_agreement") * weights["model"]
        + pl.col("temporal_consistency") * weights["temporal"]
    )

    df = df.with_columns([
        confidence_expr.alias("confidence_score"),
    ])

    logger.info(
        "confidence_calculated",
        rows=len(df),
        mean_confidence=df["confidence_score"].mean(),
        weights=weights,
    )

    return df


def filter_by_confidence(
    df: pl.DataFrame,
    min_confidence: float = 0.6,
    confidence_col: str = "confidence_score",
) -> pl.DataFrame:
    """Filter sentiment data by minimum confidence threshold.

    Args:
        df: DataFrame with confidence scores.
        min_confidence: Minimum confidence threshold (0-1 scale, default: 0.6).
        confidence_col: Name of confidence column (default: "confidence_score").

    Returns:
        Filtered DataFrame with only high-confidence signals.

    Example:
        >>> high_conf = filter_by_confidence(sentiment_df, min_confidence=0.7)
        >>> print(f"Kept {len(high_conf)} / {len(sentiment_df)} signals")
    """
    original_len = len(df)
    filtered = df.filter(pl.col(confidence_col) >= min_confidence)

    logger.info(
        "confidence_filter_applied",
        original_rows=original_len,
        filtered_rows=len(filtered),
        retention_rate=f"{len(filtered)/original_len*100:.1f}%",
        min_confidence=min_confidence,
    )

    return filtered
