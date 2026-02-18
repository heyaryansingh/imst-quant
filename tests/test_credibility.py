"""Tests for credibility (Phase 5)."""

import polars as pl
import pytest

from imst_quant.credibility import AuthorProfiler, AuthorProfile, compute_credibility_scores


def test_author_profiler():
    """CRED-01: Author profile has credibility score."""
    df = pl.DataFrame({
        "author_id": ["a1", "a1", "a1"],
        "created_utc": [1704067200, 1704153600, 1704240000],
        "subreddit": ["stocks", "stocks", "wallstreetbets"],
        "cleaned_text": ["$AAPL moon", "good stock", "buy buy"],
        "url": ["", "", "https://example.com"],
    })
    profiler = AuthorProfiler()
    p = profiler.build_profile("a1", df)
    assert p.author_id == "a1"
    assert p.total_posts == 3
    assert 0 <= p.credibility_score <= 1
    assert 0 <= p.bot_probability <= 1


def test_compute_credibility_scores(tmp_path):
    """CRED-01: Pipeline produces author -> credibility dict."""
    silver = tmp_path / "reddit" / "date=2024-01-15"
    silver.mkdir(parents=True)
    df = pl.DataFrame({
        "author_id": ["a1", "a2"],
        "subreddit": ["stocks", "stocks"],
        "cleaned_text": ["text", "more text"],
    })
    df.write_parquet(silver / "posts_enriched.parquet")
    scores = compute_credibility_scores(tmp_path, output_path=tmp_path / "cred.parquet")
    assert "a1" in scores
    assert "a2" in scores
    assert all(0 <= v <= 1 for v in scores.values())
