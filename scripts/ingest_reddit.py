#!/usr/bin/env python
"""Run Reddit ingestion.

Usage:
    python scripts/ingest_reddit.py [--limit N]

Requires .env with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.
Writes raw JSON to data/raw/reddit/{subreddit}/YYYY-MM-DD/.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.ingestion.reddit import main

if __name__ == "__main__":
    main()
