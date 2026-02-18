#!/usr/bin/env python
"""Convert silver Parquet to sentiment aggregates.

Usage:
    python scripts/silver_to_sentiment.py [--date YYYY-MM-DD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.config.settings import Settings
from imst_quant.sentiment.pipeline import silver_to_sentiment


def main():
    parser = argparse.ArgumentParser(description="Silver to sentiment aggregates")
    parser.add_argument("--date", help="Process only this date YYYY-MM-DD")
    args = parser.parse_args()

    settings = Settings()
    silver_dir = Path(settings.data.silver_dir)
    output_path = Path(settings.data.sentiment_dir) / "sentiment_aggregates.parquet"

    silver_to_sentiment(silver_dir, output_path, date=args.date)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
