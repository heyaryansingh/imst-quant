#!/usr/bin/env python
"""Run market data ingestion (equity + crypto).

Usage:
    python scripts/ingest_market.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--equity-only] [--crypto-only]

Fetches OHLCV for paper stocks and crypto pairs. Writes to data/raw/market/ and data/raw/crypto/.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imst_quant.config.settings import Settings
from imst_quant.ingestion.crypto import ingest_crypto_ohlcv
from imst_quant.ingestion.market import ingest_equity_ohlcv


def main():
    parser = argparse.ArgumentParser(description="Ingest market data (equity + crypto)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--equity-only", action="store_true", help="Only fetch equity")
    parser.add_argument("--crypto-only", action="store_true", help="Only fetch crypto")
    args = parser.parse_args()

    settings = Settings()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=90)
    start_date = args.start or start_dt.strftime("%Y-%m-%d")
    end_date = args.end or end_dt.strftime("%Y-%m-%d")

    raw_dir = Path(settings.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.crypto_only:
        df_eq = ingest_equity_ohlcv(
            settings.market.equity_tickers,
            start_date,
            end_date,
            raw_dir,
        )
        print(f"Equity: {len(df_eq)} rows")

    if not args.equity_only:
        df_crypto = ingest_crypto_ohlcv(
            "binance",
            settings.market.crypto_pairs,
            since_days=min(90, (end_dt - start_dt).days),
            output_dir=raw_dir,
        )
        print(f"Crypto: {len(df_crypto)} rows")

    print("Done.")


if __name__ == "__main__":
    main()
