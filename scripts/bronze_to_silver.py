#!/usr/bin/env python
"""Convert bronze Parquet to silver with entity links.

Usage:
    python scripts/bronze_to_silver.py [--date YYYY-MM-DD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.config.settings import Settings
from imst_quant.storage.silver import bronze_to_silver_reddit


def main():
    parser = argparse.ArgumentParser(description="Bronze to silver Parquet")
    parser.add_argument("--date", help="Process only this date YYYY-MM-DD")
    args = parser.parse_args()

    settings = Settings()
    bronze_dir = Path(settings.data.bronze_dir)
    silver_dir = Path(settings.data.silver_dir)

    bronze_to_silver_reddit(bronze_dir, silver_dir, date=args.date)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
