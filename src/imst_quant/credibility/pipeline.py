"""Credibility score pipeline for author reliability assessment.

This module implements the credibility scoring pipeline (CRED-01 to CRED-04)
which evaluates the reliability of social media authors based on their
posting history, engagement patterns, and behavior anomalies.

The pipeline includes:
    - CRED-01: Author profile building from historical posts
    - CRED-02: Credibility score computation per author
    - CRED-03: Brigade detection for coordinated manipulation
    - CRED-04: Aggregate manipulation risk scoring

Functions:
    compute_credibility_scores: Calculate per-author credibility scores
    detect_brigade: Identify coordinated posting patterns
    get_manipulation_risk: Compute overall manipulation risk

Example:
    >>> from pathlib import Path
    >>> from imst_quant.credibility.pipeline import compute_credibility_scores
    >>> scores = compute_credibility_scores(
    ...     silver_dir=Path("data/silver"),
    ...     output_path=Path("data/credibility/scores.parquet"),
    ...     date="2026-04-01"
    ... )
    >>> print(f"Scored {len(scores)} authors")
"""

from pathlib import Path
from typing import Dict

import polars as pl
import structlog

from .profiler import AuthorProfiler

logger = structlog.get_logger()


def compute_credibility_scores(
    silver_dir: Path,
    output_path: Path | None = None,
    date: str | None = None,
) -> Dict[str, float]:
    """Compute credibility scores per author from silver-layer posts.

    Reads enriched posts from the silver layer, builds author profiles
    using the AuthorProfiler, and calculates credibility scores based
    on posting patterns, engagement metrics, and historical reliability.

    Args:
        silver_dir: Path to silver layer data directory containing
            reddit/date=*/posts_enriched.parquet files.
        output_path: Optional path to write credibility scores as Parquet.
            Parent directories are created if needed.
        date: Optional date filter (YYYY-MM-DD format). If provided,
            only processes posts from that specific date.

    Returns:
        Dictionary mapping author_id (str) to credibility_score (float).
        Scores range from 0.0 (low credibility) to 1.0 (high credibility).
        Returns empty dict if no data found.

    Example:
        >>> scores = compute_credibility_scores(Path("silver"), date="2026-04-01")
        >>> low_cred = [a for a, s in scores.items() if s < 0.3]
    """
    silver_dir = Path(silver_dir)
    parts = sorted((silver_dir / "reddit").glob("date=*/posts_enriched.parquet"))
    if date:
        parts = [p for p in parts if f"date={date}" in str(p)]
    if not parts:
        return {}

    df = pl.concat([pl.read_parquet(p) for p in parts])
    if len(df) == 0:
        return {}

    profiler = AuthorProfiler()
    authors = df["author_id"].unique().to_list()
    scores = {}
    for aid in authors:
        profile = profiler.build_profile(str(aid), df)
        scores[str(aid)] = profile.credibility_score

    if output_path:
        out_df = pl.DataFrame([
            {"author_id": aid, "credibility_score": s}
            for aid, s in scores.items()
        ])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(output_path)
        logger.info("credibility_written", path=str(output_path), authors=len(scores))

    return scores


def detect_brigade(df: pl.DataFrame, time_window_min: int = 60) -> Dict[str, float]:
    """Detect coordinated posting patterns indicative of brigading (CRED-03).

    Identifies authors who post about the same asset within a narrow time
    window, which may indicate coordinated manipulation attempts.

    Args:
        df: DataFrame with columns 'author_id', 'asset', 'created_utc'.
        time_window_min: Time window in minutes to detect clustering.
            Authors posting about the same asset within this window
            are flagged as potential brigade participants.

    Returns:
        Dictionary mapping author_id to brigade_score (0.0-1.0).
        Higher scores indicate stronger brigade participation signals.
        Currently returns empty dict (placeholder implementation).
    """
    return {}


def get_manipulation_risk(
    credibility_scores: Dict[str, float],
    brigade_flags: Dict[str, float],
) -> float:
    """Calculate aggregate manipulation risk score (CRED-04).

    Combines individual author credibility scores and brigade detection
    flags to produce an overall manipulation risk assessment for a
    given time period or asset.

    Args:
        credibility_scores: Dictionary of author_id -> credibility_score.
            Values should be 0.0-1.0 with higher meaning more credible.
        brigade_flags: Dictionary of author_id -> brigade_score from
            detect_brigade(). Currently unused in implementation.

    Returns:
        Manipulation risk score from 0.0 (low risk) to 1.0 (high risk).
        Calculated as 1 - average_credibility. Returns 0.0 if no
        credibility scores provided.

    Example:
        >>> scores = {"author1": 0.8, "author2": 0.6, "author3": 0.4}
        >>> risk = get_manipulation_risk(scores, {})
        >>> print(f"Risk: {risk:.2f}")  # Risk: 0.40
    """
    if not credibility_scores:
        return 0.0
    avg_cred = sum(credibility_scores.values()) / len(credibility_scores)
    return 1.0 - avg_cred
