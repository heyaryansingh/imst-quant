"""Tests for feature engineering (Phase 6)."""

import polars as pl
import pytest

from imst_quant.features import build_daily_features


def test_build_daily_features(tmp_path):
    """FEAT-01: Feature builder produces gold parquet."""
    bronze = tmp_path / "market" / "date=2024-01-14"
    bronze.mkdir(parents=True)
    (tmp_path / "market" / "date=2024-01-10").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-11").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-12").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-13").mkdir(parents=True, exist_ok=True)
    for d, c in [("2024-01-10", 100.0), ("2024-01-11", 101.0), ("2024-01-12", 102.0),
                 ("2024-01-13", 101.5), ("2024-01-14", 103.0)]:
        p = tmp_path / "market" / f"date={d}"
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "ticker": ["AAPL"], "date": [d], "close": [c],
        }).write_parquet(p / "data.parquet")

    sent = tmp_path / "sentiment.parquet"
    pl.DataFrame({
        "date": ["2024-01-15"],
        "asset_id": ["AAPL"],
        "sentiment_index": [0.2],
        "post_count": [10],
    }).write_parquet(sent)

    out = tmp_path / "gold" / "features.parquet"
    df = build_daily_features(
        tmp_path, sent, out,
        assets=["AAPL"],
        start_date="2024-01-15",
        end_date="2024-01-15",
    )
    assert len(df) >= 1
    assert "return_1d" in df.columns
    assert "sentiment_index" in df.columns
