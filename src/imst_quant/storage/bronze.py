"""Bronze layer: raw JSON to Parquet."""

import json
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger()


def raw_to_bronze_reddit(
    raw_dir: Path,
    bronze_dir: Path,
    date: str | None = None,
) -> None:
    """Convert raw Reddit JSON to bronze Parquet."""
    raw_dir = Path(raw_dir)
    bronze_dir = Path(bronze_dir)
    reddit_raw = raw_dir / "reddit"
    if not reddit_raw.exists():
        logger.warning("raw_reddit_missing", path=str(reddit_raw))
        return

    records = []
    dates_seen = set()
    for subdir in reddit_raw.iterdir():
        if not subdir.is_dir():
            continue
        for date_dir in subdir.iterdir():
            if not date_dir.is_dir():
                continue
            date_val = date_dir.name
            if date and date_val != date:
                continue
            dates_seen.add(date_val)
            for jf in date_dir.glob("*.json"):
                try:
                    with open(jf) as f:
                        rec = json.load(f)
                    rec["date"] = date_val
                    records.append(rec)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("skip_json", path=str(jf), error=str(e))

    if not records:
        logger.info("no_reddit_records", raw_dir=str(raw_dir))
        return

    df = pl.DataFrame(records)
    for col, dtype in [
        ("created_utc", pl.Float64),
        ("score", pl.Int64),
        ("num_comments", pl.Int64),
    ]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))

    for d in df["date"].unique().to_list():
        part = df.filter(pl.col("date") == d)
        out_path = bronze_dir / "reddit" / f"date={d}"
        out_path.mkdir(parents=True, exist_ok=True)
        part.write_parquet(out_path / "data.parquet", compression="zstd", use_pyarrow=True)
        logger.info("bronze_reddit_written", date=d, rows=len(part))


def raw_to_bronze_market(raw_dir: Path, bronze_dir: Path) -> None:
    """Convert raw market and crypto JSON to bronze Parquet."""
    raw_dir = Path(raw_dir)
    bronze_dir = Path(bronze_dir)

    market_dir = raw_dir / "market"
    if market_dir.exists():
        for jf in market_dir.glob("*.json"):
            try:
                df = pl.read_json(jf)
                if len(df) == 0:
                    continue
                for d in df["date"].unique().to_list():
                    part = df.filter(pl.col("date") == d)
                    out = bronze_dir / "market" / f"date={d}"
                    out.mkdir(parents=True, exist_ok=True)
                    part.write_parquet(out / "data.parquet", compression="zstd", use_pyarrow=True)
            except Exception as e:
                logger.warning("skip_market_json", path=str(jf), error=str(e))

    crypto_dir = raw_dir / "crypto"
    if crypto_dir.exists():
        for jf in crypto_dir.glob("*.json"):
            try:
                df = pl.read_json(jf)
                if len(df) == 0:
                    continue
                for d in df["date"].unique().to_list():
                    part = df.filter(pl.col("date") == d)
                    out = bronze_dir / "crypto" / f"date={d}"
                    out.mkdir(parents=True, exist_ok=True)
                    part.write_parquet(out / "data.parquet", compression="zstd", use_pyarrow=True)
            except Exception as e:
                logger.warning("skip_crypto_json", path=str(jf), error=str(e))
