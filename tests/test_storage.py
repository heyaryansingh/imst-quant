"""Tests for storage layer modules: raw storage and bronze transformation.

Tests cover:
- Raw JSON storage with causality-preserving timestamps
- Bronze layer Parquet transformation from raw JSON
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Raw storage tests
# ---------------------------------------------------------------------------

from imst_quant.storage.raw import store_reddit_post


class TestRawStorage:
    """Tests for raw Reddit post storage."""

    def _make_submission(self, post_id="abc123", author="testuser",
                         created_utc=1700000000.0, subreddit="wallstreetbets",
                         score=42, title="Test post", selftext="Body text"):
        """Create a mock PRAW Submission object."""
        sub = MagicMock()
        sub.id = post_id
        sub.author = author
        sub.created_utc = created_utc
        sub.subreddit.display_name = subreddit
        sub.title = title
        sub.selftext = selftext
        sub.score = score
        sub.num_comments = 10
        sub.url = "https://reddit.com/r/test"
        sub.permalink = "/r/test/comments/abc123"
        return sub

    def test_stores_json_file(self, tmp_path):
        """Should write a JSON file in the correct directory structure."""
        sub = self._make_submission()
        store_reddit_post(sub, tmp_path)

        # Expect: tmp_path/wallstreetbets/2023-11-14/abc123.json
        files = list(tmp_path.rglob("*.json"))
        assert len(files) == 1
        assert "abc123.json" in files[0].name

    def test_json_content_has_required_fields(self, tmp_path):
        """Stored JSON should contain all required fields including timestamps."""
        sub = self._make_submission()
        store_reddit_post(sub, tmp_path)

        json_file = next(tmp_path.rglob("*.json"))
        with open(json_file) as f:
            record = json.load(f)

        assert record["id"] == "abc123"
        assert record["created_utc"] == 1700000000.0
        assert "retrieved_at" in record
        assert record["subreddit"] == "wallstreetbets"
        assert record["title"] == "Test post"
        assert record["score"] == 42

    def test_author_hashed(self, tmp_path):
        """Author should be stored along with a SHA-256 hash of the name."""
        sub = self._make_submission(author="realuser")
        store_reddit_post(sub, tmp_path)

        json_file = next(tmp_path.rglob("*.json"))
        with open(json_file) as f:
            record = json.load(f)

        assert record["author"] == "realuser"
        assert "author_id" in record
        assert len(record["author_id"]) == 64  # SHA-256 hex digest length

    def test_deleted_author_handled(self, tmp_path):
        """Deleted authors (None) should be stored as [deleted]."""
        sub = self._make_submission(author=None)
        sub.author = None
        store_reddit_post(sub, tmp_path)

        json_file = next(tmp_path.rglob("*.json"))
        with open(json_file) as f:
            record = json.load(f)

        assert record["author"] == "[deleted]"

    def test_directory_structure_by_date(self, tmp_path):
        """Files should be organized by subreddit/date/post_id.json."""
        sub = self._make_submission(
            subreddit="stocks",
            created_utc=1700000000.0  # 2023-11-14
        )
        store_reddit_post(sub, tmp_path)

        # Check directory hierarchy
        subreddit_dir = tmp_path / "stocks"
        assert subreddit_dir.exists()
        date_dirs = list(subreddit_dir.iterdir())
        assert len(date_dirs) == 1
        assert date_dirs[0].name == "2023-11-14"

    def test_multiple_posts_same_date(self, tmp_path):
        """Multiple posts on the same date should coexist."""
        for i, pid in enumerate(["post1", "post2", "post3"]):
            sub = self._make_submission(post_id=pid, created_utc=1700000000.0)
            store_reddit_post(sub, tmp_path)

        files = list(tmp_path.rglob("*.json"))
        assert len(files) == 3


# ---------------------------------------------------------------------------
# Bronze transformation tests
# ---------------------------------------------------------------------------

from imst_quant.storage.bronze import raw_to_bronze_reddit, raw_to_bronze_market


class TestBronzeReddit:
    """Tests for raw-to-bronze Reddit transformation."""

    def _create_raw_reddit(self, raw_dir, subreddit="test", date="2024-01-15",
                           records=None):
        """Create raw Reddit JSON files for testing."""
        if records is None:
            records = [
                {
                    "id": f"post_{i}",
                    "author_id": f"hash_{i}",
                    "created_utc": 1705276800.0 + i,
                    "retrieved_at": "2024-01-15T00:00:00+00:00",
                    "subreddit": subreddit,
                    "author": f"user_{i}",
                    "title": f"Test post {i}",
                    "selftext": f"Body {i}",
                    "score": i * 10,
                    "num_comments": i,
                    "url": "",
                    "permalink": "",
                }
                for i in range(3)
            ]

        reddit_dir = raw_dir / "reddit" / subreddit / date
        reddit_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            with open(reddit_dir / f"{rec['id']}.json", "w") as f:
                json.dump(rec, f)

        return records

    def test_produces_parquet(self, tmp_path):
        """Should create a Parquet file in the bronze directory."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        self._create_raw_reddit(raw_dir)

        raw_to_bronze_reddit(raw_dir, bronze_dir)

        parquet_files = list(bronze_dir.rglob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_preserves_record_count(self, tmp_path):
        """Bronze Parquet should contain same number of records as raw JSON."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        self._create_raw_reddit(raw_dir)

        raw_to_bronze_reddit(raw_dir, bronze_dir)

        parquet_file = next(bronze_dir.rglob("*.parquet"))
        df = pl.read_parquet(parquet_file)
        assert len(df) == 3

    def test_date_partitioning(self, tmp_path):
        """Output should be partitioned by date."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        self._create_raw_reddit(raw_dir, date="2024-01-15")

        raw_to_bronze_reddit(raw_dir, bronze_dir)

        # Should have date=2024-01-15 directory
        date_dirs = [d for d in bronze_dir.rglob("date=*") if d.is_dir()]
        assert len(date_dirs) == 1
        assert "2024-01-15" in date_dirs[0].name

    def test_type_casting(self, tmp_path):
        """Numeric columns should be properly cast."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        self._create_raw_reddit(raw_dir)

        raw_to_bronze_reddit(raw_dir, bronze_dir)

        parquet_file = next(bronze_dir.rglob("*.parquet"))
        df = pl.read_parquet(parquet_file)

        if "score" in df.columns:
            assert df["score"].dtype == pl.Int64
        if "created_utc" in df.columns:
            assert df["created_utc"].dtype == pl.Float64

    def test_missing_raw_dir_no_error(self, tmp_path):
        """Should handle missing raw directory gracefully."""
        raw_dir = tmp_path / "nonexistent"
        bronze_dir = tmp_path / "bronze"
        # Should not raise
        raw_to_bronze_reddit(raw_dir, bronze_dir)

    def test_date_filter(self, tmp_path):
        """When date filter is provided, only that date should be processed."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        self._create_raw_reddit(raw_dir, date="2024-01-15")
        self._create_raw_reddit(raw_dir, date="2024-01-16")

        raw_to_bronze_reddit(raw_dir, bronze_dir, date="2024-01-15")

        parquet_files = list(bronze_dir.rglob("*.parquet"))
        assert len(parquet_files) == 1
        assert "2024-01-15" in str(parquet_files[0])


class TestBronzeMarket:
    """Tests for raw-to-bronze market data transformation."""

    def test_missing_market_dir_no_error(self, tmp_path):
        """Should handle missing market directory gracefully."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        bronze_dir = tmp_path / "bronze"
        # Should not raise
        raw_to_bronze_market(raw_dir, bronze_dir)
