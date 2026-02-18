#!/usr/bin/env python
"""Convert raw JSON to bronze Parquet.

Usage:
    python scripts/raw_to_bronze.py [--reddit-only] [--market-only] [--date YYYY-MM-DD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from imst_quant.config.settings import Settings
from imst_quant.storage.bronze import raw_to_bronze_market, raw_to_bronze_reddit

logger = structlog.get_logger()


def main():
    parser = argparse.ArgumentParser(description="Convert raw to bronze Parquet")
    parser.add_argument("--reddit-only", action="store_true")
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--date", help="Process only this date YYYY-MM-DD")
    args = parser.parse_args()

    settings = Settings()
    raw_dir = Path(settings.data.raw_dir)
    bronze_dir = Path(settings.data.bronze_dir)

    if not raw_dir.exists():
        logger.warning("raw_dir_missing", path=str(raw_dir))
        return 0

    if not args.market_only:
        raw_to_bronze_reddit(raw_dir, bronze_dir, date=args.date)
    if not args.reddit_only:
        raw_to_bronze_market(raw_dir, bronze_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
