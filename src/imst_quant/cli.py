#!/usr/bin/env python
"""IMST-Quant CLI: Unified command-line interface for the trading pipeline.

Usage:
    imst --help                     Show all commands
    imst ingest --reddit --market   Ingest data from sources
    imst process --all              Run full data processing pipeline
    imst analyze --sentiment        Run sentiment analysis
    imst backtest --model lstm      Train model and run backtest
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all CLI subcommands.

    Returns:
        Configured ArgumentParser with ingest, process, analyze, backtest,
        and status subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="imst",
        description="IMST-Quant: Influence-aware Multi-Source Sentiment Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  imst ingest --reddit --limit 500      Ingest 500 Reddit posts per subreddit
  imst ingest --market --start 2024-01-01
  imst process --all                    Run full Raw -> Bronze -> Silver pipeline
  imst analyze --sentiment              Generate sentiment aggregates
  imst analyze --influence --year 2024 --month 1
  imst backtest --model lstm --epochs 20
        """,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest subcommand ---
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest data from Reddit, equity markets, and crypto exchanges"
    )
    ingest_parser.add_argument(
        "--reddit", action="store_true", help="Ingest Reddit posts"
    )
    ingest_parser.add_argument(
        "--market", action="store_true", help="Ingest equity market data (yfinance)"
    )
    ingest_parser.add_argument(
        "--crypto", action="store_true", help="Ingest crypto data (CCXT)"
    )
    ingest_parser.add_argument(
        "--all", action="store_true", help="Ingest from all sources"
    )
    ingest_parser.add_argument(
        "--limit", type=int, default=100, help="Max posts per subreddit (default: 100)"
    )
    ingest_parser.add_argument(
        "--start", help="Start date YYYY-MM-DD (default: 90 days ago)"
    )
    ingest_parser.add_argument(
        "--end", help="End date YYYY-MM-DD (default: today)"
    )

    # --- process subcommand ---
    process_parser = subparsers.add_parser(
        "process", help="Process data through the medallion architecture"
    )
    process_parser.add_argument(
        "--raw-to-bronze", action="store_true", help="Convert raw JSON to bronze Parquet"
    )
    process_parser.add_argument(
        "--bronze-to-silver", action="store_true", help="Convert bronze to silver (entity linking)"
    )
    process_parser.add_argument(
        "--build-features", action="store_true", help="Build gold feature vectors"
    )
    process_parser.add_argument(
        "--all", action="store_true", help="Run all processing steps"
    )
    process_parser.add_argument(
        "--date", help="Process only this date YYYY-MM-DD"
    )

    # --- analyze subcommand ---
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run analysis: sentiment, influence, or credibility"
    )
    analyze_parser.add_argument(
        "--sentiment", action="store_true", help="Run sentiment analysis pipeline"
    )
    analyze_parser.add_argument(
        "--influence", action="store_true", help="Train GNN influence model"
    )
    analyze_parser.add_argument(
        "--credibility", action="store_true", help="Run credibility profiling"
    )
    analyze_parser.add_argument(
        "--year", type=int, help="Year for influence analysis"
    )
    analyze_parser.add_argument(
        "--month", type=int, help="Month for influence analysis"
    )
    analyze_parser.add_argument(
        "--date", help="Process only this date YYYY-MM-DD"
    )

    # --- backtest subcommand ---
    backtest_parser = subparsers.add_parser(
        "backtest", help="Train forecasting models and run backtests"
    )
    backtest_parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "transformer", "lightgbm"],
        default="lstm",
        help="Model type (default: lstm)",
    )
    backtest_parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20)"
    )
    backtest_parser.add_argument(
        "--window", type=int, default=5, help="Lookback window (default: 5)"
    )
    backtest_parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost as decimal (default: 0.001)",
    )
    backtest_parser.add_argument(
        "--train-only", action="store_true", help="Only train, skip backtest"
    )
    backtest_parser.add_argument(
        "--backtest-only", action="store_true", help="Only backtest with existing features"
    )

    # --- status subcommand ---
    status_parser = subparsers.add_parser(
        "status", help="Show pipeline status and data statistics"
    )

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest subcommand for data collection.

    Ingests data from configured sources: Reddit posts, equity market OHLCV,
    and cryptocurrency data via CCXT.

    Args:
        args: Parsed command-line arguments including source flags and date range.

    Returns:
        Exit code (0 for success).
    """
    from imst_quant.config.settings import Settings

    settings = Settings()
    raw_dir = Path(settings.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ingest_all = args.all
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=90)
    start_date = args.start or start_dt.strftime("%Y-%m-%d")
    end_date = args.end or end_dt.strftime("%Y-%m-%d")

    if args.reddit or ingest_all:
        print("Ingesting Reddit posts...")
        try:
            from imst_quant.ingestion.reddit import main as reddit_main

            # Temporarily override sys.argv for the script
            old_argv = sys.argv
            sys.argv = ["ingest_reddit", f"--limit={args.limit}"]
            try:
                reddit_main()
            finally:
                sys.argv = old_argv
        except Exception as e:
            logger.error("reddit_ingestion_failed", error=str(e))
            print(f"Warning: Reddit ingestion failed: {e}")

    if args.market or ingest_all:
        print("Ingesting equity market data...")
        try:
            from imst_quant.ingestion.market import ingest_equity_ohlcv

            df = ingest_equity_ohlcv(
                settings.market.equity_tickers,
                start_date,
                end_date,
                raw_dir,
            )
            print(f"  Equity: {len(df)} rows ingested")
        except Exception as e:
            logger.error("market_ingestion_failed", error=str(e))
            print(f"Warning: Market ingestion failed: {e}")

    if args.crypto or ingest_all:
        print("Ingesting crypto market data...")
        try:
            from imst_quant.ingestion.crypto import ingest_crypto_ohlcv

            df = ingest_crypto_ohlcv(
                "binance",
                settings.market.crypto_pairs,
                since_days=min(90, (end_dt - start_dt).days),
                output_dir=raw_dir,
            )
            print(f"  Crypto: {len(df)} rows ingested")
        except Exception as e:
            logger.error("crypto_ingestion_failed", error=str(e))
            print(f"Warning: Crypto ingestion failed: {e}")

    print("Ingestion complete.")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    """Handle the process subcommand for medallion architecture pipeline.

    Processes data through Raw -> Bronze -> Silver -> Gold layers with
    normalization, entity linking, and feature engineering.

    Args:
        args: Parsed command-line arguments including processing stage flags.

    Returns:
        Exit code (0 for success).
    """
    from imst_quant.config.settings import Settings

    settings = Settings()
    raw_dir = Path(settings.data.raw_dir)
    bronze_dir = Path(settings.data.bronze_dir)
    silver_dir = Path(settings.data.silver_dir)
    gold_dir = Path(settings.data.gold_dir)
    sentiment_path = Path(settings.data.sentiment_dir) / "sentiment_aggregates.parquet"

    process_all = args.all

    if args.raw_to_bronze or process_all:
        print("Converting raw to bronze...")
        try:
            from imst_quant.storage.bronze import raw_to_bronze_market, raw_to_bronze_reddit

            if raw_dir.exists():
                raw_to_bronze_reddit(raw_dir, bronze_dir, date=args.date)
                raw_to_bronze_market(raw_dir, bronze_dir)
                print("  Raw -> Bronze complete")
            else:
                print("  Warning: raw_dir does not exist, skipping")
        except Exception as e:
            logger.error("raw_to_bronze_failed", error=str(e))
            print(f"Warning: Raw to bronze failed: {e}")

    if args.bronze_to_silver or process_all:
        print("Converting bronze to silver...")
        try:
            from imst_quant.storage.silver import bronze_to_silver_reddit

            bronze_to_silver_reddit(bronze_dir, silver_dir, date=args.date)
            print("  Bronze -> Silver complete")
        except Exception as e:
            logger.error("bronze_to_silver_failed", error=str(e))
            print(f"Warning: Bronze to silver failed: {e}")

    if args.build_features or process_all:
        print("Building gold feature vectors...")
        try:
            from imst_quant.features import build_daily_features

            output_path = gold_dir / "features.parquet"
            build_daily_features(
                bronze_dir,
                sentiment_path,
                output_path,
            )
            print(f"  Features written to {output_path}")
        except Exception as e:
            logger.error("build_features_failed", error=str(e))
            print(f"Warning: Feature building failed: {e}")

    print("Processing complete.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the analyze subcommand for sentiment, influence, and credibility.

    Runs analysis pipelines including FinBERT sentiment, GNN influence modeling,
    and user credibility profiling.

    Args:
        args: Parsed command-line arguments including analysis type and date filters.

    Returns:
        Exit code (0 for success, 1 for missing required arguments).
    """
    from imst_quant.config.settings import Settings

    settings = Settings()
    silver_dir = Path(settings.data.silver_dir)
    sentiment_dir = Path(settings.data.sentiment_dir)
    influence_dir = Path(settings.data.influence_dir)

    if args.sentiment:
        print("Running sentiment analysis...")
        try:
            from imst_quant.sentiment.pipeline import silver_to_sentiment

            output_path = sentiment_dir / "sentiment_aggregates.parquet"
            silver_to_sentiment(silver_dir, output_path, date=args.date)
            print(f"  Sentiment aggregates written to {output_path}")
        except Exception as e:
            logger.error("sentiment_analysis_failed", error=str(e))
            print(f"Warning: Sentiment analysis failed: {e}")

    if args.influence:
        if not args.year or not args.month:
            print("Error: --year and --month are required for influence analysis")
            return 1

        print(f"Training GNN influence model for {args.year}-{args.month:02d}...")
        try:
            from imst_quant.influence.pipeline import run_influence_month

            result = run_influence_month(
                silver_dir,
                influence_dir,
                args.year,
                args.month,
                min_interactions=50,
            )
            if result:
                print("  Influence model trained successfully")
            else:
                print("  Skipped (insufficient data)")
        except Exception as e:
            logger.error("influence_analysis_failed", error=str(e))
            print(f"Warning: Influence analysis failed: {e}")

    if args.credibility:
        print("Running credibility profiling...")
        try:
            from imst_quant.credibility.pipeline import run_credibility_pipeline

            credibility_dir = Path(settings.data.credibility_dir)
            run_credibility_pipeline(silver_dir, credibility_dir)
            print("  Credibility profiling complete")
        except Exception as e:
            logger.error("credibility_analysis_failed", error=str(e))
            print(f"Warning: Credibility profiling failed: {e}")

    print("Analysis complete.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Handle the backtest subcommand for model training and evaluation.

    Trains forecasting models (LSTM, CNN, Transformer, LightGBM) on feature
    vectors and runs historical backtests with configurable transaction costs.

    Args:
        args: Parsed command-line arguments including model type, epochs, and flags.

    Returns:
        Exit code (0 for success, 1 for missing features or training failure).
    """
    from imst_quant.config.settings import Settings

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)
    features_path = gold_dir / "features.parquet"

    if not features_path.exists() and not args.train_only:
        print(f"Error: Features file not found at {features_path}")
        print("Run 'imst process --build-features' first.")
        return 1

    model = None

    if not args.backtest_only:
        print(f"Training {args.model.upper()} forecaster...")
        try:
            from imst_quant.models.train import train_forecaster

            model_path = gold_dir / f"{args.model}_model.pt"
            model = train_forecaster(
                features_path=features_path,
                model_type=args.model,
                output_path=model_path,
                window=args.window,
                epochs=args.epochs,
            )
            print(f"  Model saved to {model_path}")
        except Exception as e:
            logger.error("training_failed", error=str(e))
            print(f"Warning: Training failed: {e}")
            return 1

    if not args.train_only:
        print("Running backtest...")
        try:
            from imst_quant.trading.backtest import run_backtest

            results = run_backtest(
                features_path=features_path,
                predictions=None,  # Use simple signal strategy
                transaction_cost=args.transaction_cost,
            )
            print("\n--- Backtest Results ---")
            print(f"  Total PnL:    {results['total_pnl']:.4f}")
            print(f"  Sharpe Ratio: {results['sharpe']:.4f}")
            print(f"  Total Trades: {results['trades']}")
        except Exception as e:
            logger.error("backtest_failed", error=str(e))
            print(f"Warning: Backtest failed: {e}")
            return 1

    print("\nBacktest complete.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Handle the status subcommand for pipeline overview.

    Displays data layer statistics (file counts, sizes) and environment
    configuration status.

    Args:
        args: Parsed command-line arguments (currently unused).

    Returns:
        Exit code (0 for success).
    """
    from imst_quant.config.settings import Settings

    settings = Settings()

    print("\n=== IMST-Quant Pipeline Status ===\n")

    data_layers = [
        ("Raw", settings.data.raw_dir),
        ("Bronze", settings.data.bronze_dir),
        ("Silver", settings.data.silver_dir),
        ("Sentiment", settings.data.sentiment_dir),
        ("Influence", settings.data.influence_dir),
        ("Gold", settings.data.gold_dir),
    ]

    for name, path in data_layers:
        path = Path(path)
        if path.exists():
            files = list(path.rglob("*"))
            parquet_files = [f for f in files if f.suffix == ".parquet"]
            json_files = [f for f in files if f.suffix == ".json"]
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"{name:12} {'OK':6} {len(parquet_files):4} parquet, {len(json_files):4} json ({size_mb:.1f} MB)")
        else:
            print(f"{name:12} {'EMPTY':6}")

    # Check environment
    print("\n--- Environment ---")
    if settings.reddit.client_id:
        print("Reddit API:    Configured")
    else:
        print("Reddit API:    Not configured (set REDDIT_CLIENT_ID in .env)")

    print(f"Equity tickers: {', '.join(settings.market.equity_tickers)}")
    print(f"Crypto pairs:   {', '.join(settings.market.crypto_pairs)}")

    print()
    return 0


def main() -> int:
    """Main CLI entry point for IMST-Quant.

    Parses command-line arguments and dispatches to the appropriate
    subcommand handler.

    Returns:
        Exit code from the executed subcommand (0 for success).
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "ingest": cmd_ingest,
        "process": cmd_process,
        "analyze": cmd_analyze,
        "backtest": cmd_backtest,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
