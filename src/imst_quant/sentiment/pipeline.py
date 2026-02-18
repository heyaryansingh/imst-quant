"""Silver to sentiment aggregates pipeline."""

import json
from pathlib import Path
from typing import Dict

import polars as pl
import structlog

from .aggregation import aggregate_daily_sentiment
from .textblob import BaselineSentimentAnalyzer

logger = structlog.get_logger()


def silver_to_sentiment(
    silver_dir: Path,
    output_path: Path,
    date: str | None = None,
    influence_scores: Dict[str, float] | None = None,
) -> None:
    """
    Read silver posts, compute TextBlob polarity, aggregate daily per asset.

    Output: Parquet with columns date, asset_id, sentiment_index, post_count.
    """
    silver_dir = Path(silver_dir)
    output_path = Path(output_path)
    silver_reddit = silver_dir / "reddit"
    if not silver_reddit.exists():
        logger.warning("silver_reddit_missing", path=str(silver_reddit))
        return

    parts = sorted(silver_reddit.glob("date=*/posts_enriched.parquet"))
    if date:
        parts = [p for p in parts if f"date={date}" in str(p)]

    analyzer = BaselineSentimentAnalyzer()
    all_rows = []

    for part_path in parts:
        df = pl.read_parquet(part_path)
        if len(df) == 0:
            continue

        for r in df.iter_rows(named=True):
            try:
                links = json.loads(r.get("entity_links", "[]"))
            except (json.JSONDecodeError, TypeError):
                links = []
            if not links:
                continue
            text = r.get("cleaned_text", "") or f"{r.get('title','')} {r.get('selftext','')}"
            score = analyzer.analyze(text)
            author_id = str(r.get("author_id", ""))
            row_date = r.get("date", "")
            for link in links:
                asset_id = link.get("asset_id", "")
                if not asset_id:
                    continue
                all_rows.append({
                    "date": row_date,
                    "asset_id": asset_id,
                    "author_id": author_id,
                    "polarity": score.polarity,
                })

    if not all_rows:
        logger.info("no_posts_with_entities")
        return

    posts_df = pl.DataFrame(all_rows)
    dates = posts_df["date"].unique().to_list()
    results = []
    for d in dates:
        idx = aggregate_daily_sentiment(posts_df, d, influence_scores)
        count_df = posts_df.filter(pl.col("date") == d).group_by("asset_id").agg(pl.len().alias("post_count"))
        for asset, sent in idx.items():
            count_row = count_df.filter(pl.col("asset_id") == asset)
            post_count = count_row["post_count"][0] if len(count_row) > 0 else 0
            results.append({
                "date": d,
                "asset_id": asset,
                "sentiment_index": sent,
                "post_count": post_count,
            })

    if not results:
        return
    out_df = pl.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path, compression="zstd")
    logger.info("sentiment_written", path=str(output_path), rows=len(out_df))
