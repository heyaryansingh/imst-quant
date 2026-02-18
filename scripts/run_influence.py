#!/usr/bin/env python
"""Build influence graph and train GCN for a month.

Usage:
    python scripts/run_influence.py --year YYYY --month MM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.config.settings import Settings
from imst_quant.influence.pipeline import run_influence_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--min-interactions", type=int, default=50)
    args = parser.parse_args()

    settings = Settings()
    out = run_influence_month(
        settings.data.silver_dir,
        settings.data.influence_dir,
        args.year,
        args.month,
        min_interactions=args.min_interactions,
    )
    print("Done." if out else "Skipped (no data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
