"""Silver layer: bronze Parquet + text processing + entity linking."""

import json
from pathlib import Path

import polars as pl
import structlog

from imst_quant.entities.linker import EntityLinker
from imst_quant.processing.deduplication import Deduplicator
from imst_quant.processing.language import LanguageDetector
from imst_quant.processing.normalizer import TextNormalizer

logger = structlog.get_logger()


def bronze_to_silver_reddit(
    bronze_dir: Path,
    silver_dir: Path,
    date: str | None = None,
) -> None:
    """
    Convert bronze Reddit Parquet to silver with entity links.

    Applies: language filter, deduplication, text normalization, entity linking.
    Output: silver_dir/reddit/date={date}/posts_enriched.parquet
    """
    bronze_dir = Path(bronze_dir)
    silver_dir = Path(silver_dir)
    bronze_reddit = bronze_dir / "reddit"
    if not bronze_reddit.exists():
        logger.warning("bronze_reddit_missing", path=str(bronze_reddit))
        return

    parts = sorted(bronze_reddit.glob("date=*/data.parquet"))
    if date:
        parts = [p for p in parts if p.parent.name == f"date={date}"]

    detector = LanguageDetector()
    dedup = Deduplicator()
    normalizer = TextNormalizer()
    linker = EntityLinker(confidence_threshold=0.35)

    for part_path in parts:
        date_val = part_path.parent.name.replace("date=", "")
        df = pl.read_parquet(part_path)
        if len(df) == 0:
            continue

        rows = []
        for r in df.iter_rows(named=True):
            text = f"{r.get('title', '') or ''} {r.get('selftext', '') or ''}".strip()
            is_en, _ = detector.is_english(text)
            if not is_en:
                continue
            is_dup, _ = dedup.is_duplicate(str(r.get("id", "")), text)
            if is_dup:
                continue
            norm = normalizer.normalize(text)
            links = linker.link_entities(norm.cleaned, r.get("subreddit", "") or "")
            entity_links = [
                {"asset_id": l.asset_id, "confidence": l.confidence}
                for l in links
            ]
            rows.append({
                "id": r.get("id"),
                "created_utc": r.get("created_utc"),
                "retrieved_at": r.get("retrieved_at"),
                "subreddit": r.get("subreddit"),
                "author_id": r.get("author_id"),
                "title": r.get("title"),
                "selftext": r.get("selftext"),
                "cleaned_text": norm.cleaned,
                "is_english": True,
                "entity_links": json.dumps(entity_links),
                "score": r.get("score"),
                "num_comments": r.get("num_comments"),
                "url": r.get("url"),
                "permalink": r.get("permalink"),
                "date": r.get("date", date_val),
            })

        if not rows:
            logger.info("silver_no_rows_after_filter", date=date_val)
            continue

        out_df = pl.DataFrame(rows)
        out_path = silver_dir / "reddit" / f"date={date_val}"
        out_path.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(
            out_path / "posts_enriched.parquet",
            compression="zstd",
            use_pyarrow=True,
        )
        logger.info("silver_reddit_written", date=date_val, rows=len(out_df))
