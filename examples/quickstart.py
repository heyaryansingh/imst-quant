#!/usr/bin/env python
"""IMST-Quant Quickstart Example

This script demonstrates the end-to-end pipeline:
1. Ingest market data (equity prices)
2. Process through bronze/silver layers
3. Build feature vectors
4. Train a forecasting model
5. Run a backtest

Prerequisites:
    pip install -e ".[dev]"

For Reddit ingestion, also configure .env with Reddit API credentials.

Usage:
    python examples/quickstart.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for development
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    """Run the quickstart demo."""
    print("=" * 60)
    print("IMST-Quant Quickstart Demo")
    print("=" * 60)
    print()

    # --- Step 1: Configuration ---
    print("[1/6] Loading configuration...")
    from imst_quant.config.settings import Settings

    settings = Settings()

    # Define paths
    raw_dir = Path(settings.data.raw_dir)
    bronze_dir = Path(settings.data.bronze_dir)
    silver_dir = Path(settings.data.silver_dir)
    sentiment_dir = Path(settings.data.sentiment_dir)
    gold_dir = Path(settings.data.gold_dir)

    # Create directories
    for d in [raw_dir, bronze_dir, silver_dir, sentiment_dir, gold_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"      Raw dir:       {raw_dir}")
    print(f"      Gold dir:      {gold_dir}")
    print()

    # --- Step 2: Ingest Market Data ---
    print("[2/6] Ingesting market data...")
    from imst_quant.ingestion.market import ingest_equity_ohlcv

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    try:
        df_market = ingest_equity_ohlcv(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir=raw_dir,
        )
        print(f"      Ingested {len(df_market)} market data rows")
        print(f"      Tickers: {tickers}")
        print(f"      Date range: {start_date} to {end_date}")
    except Exception as e:
        print(f"      Warning: Market ingestion failed: {e}")
        print("      Continuing with existing data...")
    print()

    # --- Step 3: Process to Bronze ---
    print("[3/6] Processing raw -> bronze...")
    from imst_quant.storage.bronze import raw_to_bronze_market

    try:
        raw_to_bronze_market(raw_dir, bronze_dir)
        parquet_files = list(bronze_dir.rglob("*.parquet"))
        print(f"      Bronze parquet files: {len(parquet_files)}")
    except Exception as e:
        print(f"      Warning: Bronze processing failed: {e}")
    print()

    # --- Step 4: Build Features ---
    print("[4/6] Building gold feature vectors...")
    from imst_quant.features import build_daily_features

    features_path = gold_dir / "features.parquet"
    sentiment_path = sentiment_dir / "sentiment_aggregates.parquet"

    # Create empty sentiment file if not exists (for demo purposes)
    if not sentiment_path.exists():
        import polars as pl
        sentiment_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "date": ["2024-01-01"],
            "asset_id": ["AAPL"],
            "sentiment_index": [0.0],
            "post_count": [0],
        }).write_parquet(sentiment_path)
        print("      Created placeholder sentiment file")

    try:
        df_features = build_daily_features(
            bronze_dir=bronze_dir,
            sentiment_path=sentiment_path,
            output_path=features_path,
            assets=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"      Feature rows: {len(df_features)}")
        print(f"      Features: {df_features.columns}")
    except Exception as e:
        print(f"      Warning: Feature building failed: {e}")
        print("      Creating synthetic features for demo...")

        import polars as pl
        import random

        # Create synthetic features for demonstration
        dates = []
        current = datetime.now() - timedelta(days=60)
        for _ in range(60):
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        rows = []
        for ticker in tickers:
            for d in dates:
                rows.append({
                    "date": d,
                    "asset_id": ticker,
                    "return_1d": random.gauss(0.001, 0.02),
                    "return_5d": random.gauss(0.005, 0.05),
                    "volatility_30d": abs(random.gauss(0.02, 0.01)),
                    "related_return_1": random.gauss(0.001, 0.02),
                    "related_return_2": random.gauss(0.001, 0.02),
                    "related_return_3": random.gauss(0.001, 0.02),
                    "sentiment_index": random.gauss(0, 0.3),
                    "post_count": random.randint(0, 100),
                })

        df_features = pl.DataFrame(rows)
        df_features.write_parquet(features_path)
        print(f"      Created {len(df_features)} synthetic feature rows")
    print()

    # --- Step 5: Train Model ---
    print("[5/6] Training LSTM forecaster...")
    try:
        from imst_quant.models.train import train_forecaster

        model = train_forecaster(
            features_path=features_path,
            model_type="lstm",
            output_path=gold_dir / "lstm_model.pt",
            window=5,
            epochs=10,  # Reduced for demo
        )
        print("      Model trained successfully")
        print(f"      Model saved to: {gold_dir / 'lstm_model.pt'}")
    except ValueError as e:
        print(f"      Warning: Training skipped - {e}")
        print("      (Need more data for training)")
    except Exception as e:
        print(f"      Warning: Training failed: {e}")
    print()

    # --- Step 6: Run Backtest ---
    print("[6/6] Running backtest...")
    from imst_quant.trading.backtest import run_backtest

    try:
        results = run_backtest(
            features_path=features_path,
            predictions=None,  # Use simple strategy
            transaction_cost=0.001,
        )

        print()
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total PnL:      {results['total_pnl']:+.4f}")
        print(f"  Sharpe Ratio:   {results['sharpe']:.4f}")
        print(f"  Total Trades:   {results['trades']}")
        print("=" * 60)
    except Exception as e:
        print(f"      Warning: Backtest failed: {e}")

    print()
    print("Demo complete!")
    print()
    print("Next steps:")
    print("  1. Configure Reddit API in .env for social sentiment")
    print("  2. Run 'imst ingest --reddit --limit 1000' to get posts")
    print("  3. Run 'imst process --all' to build full pipeline")
    print("  4. Run 'imst analyze --sentiment' for sentiment scores")
    print("  5. Run 'imst backtest --model transformer --epochs 50'")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
