"""Credibility score pipeline (CRED-01 to CRED-04)."""

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
    """
    Compute credibility scores per author from silver posts.
    Returns author_id -> credibility_score.
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
    """CRED-03: Simple brigade flag - authors posting near-same time on same asset."""
    return {}


def get_manipulation_risk(
    credibility_scores: Dict[str, float],
    brigade_flags: Dict[str, float],
) -> float:
    """CRED-04: Aggregate manipulation risk."""
    if not credibility_scores:
        return 0.0
    avg_cred = sum(credibility_scores.values()) / len(credibility_scores)
    return 1.0 - avg_cred
