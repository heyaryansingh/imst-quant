#!/usr/bin/env python
"""Build gold feature vectors.

Usage:
    python scripts/build_features.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.config.settings import Settings
from imst_quant.features import build_daily_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    s = Settings()
    build_daily_features(
        s.data.bronze_dir,
        s.data.sentiment_dir / "sentiment_aggregates.parquet",
        s.data.gold_dir / "features.parquet",
        start_date=args.start,
        end_date=args.end,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
