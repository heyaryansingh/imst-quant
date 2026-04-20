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

    # --- portfolio subcommand ---
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Analyze portfolio performance and risk metrics"
    )
    portfolio_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    portfolio_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Daily risk-free rate for Sharpe/Sortino (default: 0)",
    )
    portfolio_parser.add_argument(
        "--var-confidence",
        type=float,
        default=0.95,
        help="Confidence level for VaR calculation (default: 0.95)",
    )
    portfolio_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    portfolio_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- correlation subcommand ---
    correlation_parser = subparsers.add_parser(
        "correlation", help="Analyze asset correlations and relationships"
    )
    correlation_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    correlation_parser.add_argument(
        "--asset1", help="First asset for pairwise correlation analysis"
    )
    correlation_parser.add_argument(
        "--asset2", help="Second asset for pairwise correlation analysis"
    )
    correlation_parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Window size for rolling correlation (default: 60)",
    )
    correlation_parser.add_argument(
        "--method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    correlation_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- validate subcommand ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate data quality and integrity"
    )
    validate_parser.add_argument(
        "--data", help="Path to data file to validate", required=True
    )
    validate_parser.add_argument(
        "--type",
        choices=["ohlcv", "returns", "all"],
        default="all",
        help="Type of validation to run (default: all)",
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- report subcommand ---
    report_parser = subparsers.add_parser(
        "report", help="Generate and export backtest reports"
    )
    report_parser.add_argument(
        "--features", help="Path to features file for backtest"
    )
    report_parser.add_argument(
        "--strategy", default="default", help="Strategy name for report"
    )
    report_parser.add_argument(
        "--output", help="Output path for report (without extension)"
    )
    report_parser.add_argument(
        "--format",
        choices=["json", "csv", "html", "all"],
        default="all",
        help="Export format (default: all)",
    )

    # --- montecarlo subcommand ---
    montecarlo_parser = subparsers.add_parser(
        "montecarlo", help="Run Monte Carlo simulations for risk assessment"
    )
    montecarlo_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    montecarlo_parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    montecarlo_parser.add_argument(
        "--horizon",
        type=int,
        default=252,
        help="Simulation horizon in trading days (default: 252)",
    )
    montecarlo_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for VaR/ES (default: 0.95)",
    )
    montecarlo_parser.add_argument(
        "--method",
        choices=["historical", "parametric", "gbm"],
        default="historical",
        help="Simulation method (default: historical)",
    )
    montecarlo_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )
    montecarlo_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- benchmark subcommand ---
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Compare strategy performance against benchmarks"
    )
    benchmark_parser.add_argument(
        "--strategy", help="Path to strategy returns file", required=True
    )
    benchmark_parser.add_argument(
        "--benchmark", help="Path to benchmark returns file or ticker (e.g., SPY)"
    )
    benchmark_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Daily risk-free rate for alpha calculation (default: 0)",
    )
    benchmark_parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Window for rolling metrics (default: 60)",
    )
    benchmark_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- regime subcommand ---
    regime_parser = subparsers.add_parser(
        "regime", help="Detect market regimes (volatility, trend, combined)"
    )
    regime_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    regime_parser.add_argument(
        "--type",
        choices=["volatility", "trend", "combined"],
        default="combined",
        help="Type of regime detection (default: combined)",
    )
    regime_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    regime_parser.add_argument(
        "--volatility-window",
        type=int,
        default=20,
        help="Window for volatility calculation (default: 20)",
    )
    regime_parser.add_argument(
        "--trend-window",
        type=int,
        default=50,
        help="Window for trend calculation (default: 50)",
    )
    regime_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- journal subcommand ---
    journal_parser = subparsers.add_parser(
        "journal", help="Trade journal for logging and analyzing trades"
    )
    journal_parser.add_argument(
        "--action",
        choices=["log", "close", "list", "stats", "report", "export"],
        default="list",
        help="Journal action (default: list)",
    )
    journal_parser.add_argument(
        "--symbol", help="Trading symbol for new trade"
    )
    journal_parser.add_argument(
        "--direction",
        choices=["long", "short"],
        help="Trade direction",
    )
    journal_parser.add_argument(
        "--entry-price", type=float, help="Entry price"
    )
    journal_parser.add_argument(
        "--quantity", type=float, help="Position size"
    )
    journal_parser.add_argument(
        "--reason", help="Trade rationale"
    )
    journal_parser.add_argument(
        "--trade-id", help="Trade ID for close action"
    )
    journal_parser.add_argument(
        "--exit-price", type=float, help="Exit price for close action"
    )
    journal_parser.add_argument(
        "--setup", help="Setup type (e.g., breakout, pullback)"
    )
    journal_parser.add_argument(
        "--stop-loss", type=float, help="Stop loss price"
    )
    journal_parser.add_argument(
        "--take-profit", type=float, help="Take profit price"
    )
    journal_parser.add_argument(
        "--output", help="Output path for export"
    )
    journal_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- seasonality subcommand ---
    seasonality_parser = subparsers.add_parser(
        "seasonality", help="Analyze seasonal patterns in returns"
    )
    seasonality_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    seasonality_parser.add_argument(
        "--type",
        choices=["dow", "monthly", "tom", "holiday", "all"],
        default="all",
        help="Type of seasonality analysis (default: all)",
    )
    seasonality_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    seasonality_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- liquidity subcommand ---
    liquidity_parser = subparsers.add_parser(
        "liquidity", help="Analyze market liquidity and trading costs"
    )
    liquidity_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    liquidity_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    liquidity_parser.add_argument(
        "--order-size",
        type=float,
        help="Order size for market impact estimation",
    )
    liquidity_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- pairs subcommand ---
    pairs_parser = subparsers.add_parser(
        "pairs", help="Cointegration analysis for pairs trading"
    )
    pairs_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    pairs_parser.add_argument(
        "--asset1", help="First asset for pair analysis"
    )
    pairs_parser.add_argument(
        "--asset2", help="Second asset for pair analysis"
    )
    pairs_parser.add_argument(
        "--scan", action="store_true", help="Scan all pairs for cointegration"
    )
    pairs_parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Significance level for cointegration test (default: 0.05)",
    )
    pairs_parser.add_argument(
        "--min-half-life",
        type=float,
        default=1.0,
        help="Minimum half-life in periods (default: 1)",
    )
    pairs_parser.add_argument(
        "--max-half-life",
        type=float,
        default=60.0,
        help="Maximum half-life in periods (default: 60)",
    )
    pairs_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- optimize subcommand ---
    optimize_parser = subparsers.add_parser(
        "optimize", help="Portfolio optimization (mean-variance, risk parity, HRP)"
    )
    optimize_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    optimize_parser.add_argument(
        "--method",
        choices=["sharpe", "variance", "risk-parity", "hrp"],
        default="sharpe",
        help="Optimization method (default: sharpe)",
    )
    optimize_parser.add_argument(
        "--target-return",
        type=float,
        help="Target annual return (for target-return optimization)",
    )
    optimize_parser.add_argument(
        "--target-volatility",
        type=float,
        help="Target annual volatility (for target-risk optimization)",
    )
    optimize_parser.add_argument(
        "--allow-short", action="store_true", help="Allow short positions"
    )
    optimize_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- orderflow subcommand ---
    orderflow_parser = subparsers.add_parser(
        "orderflow", help="Order flow analysis (VPIN, volume imbalance, toxicity)"
    )
    orderflow_parser.add_argument(
        "--data", help="Path to tick/trade data file", required=True
    )
    orderflow_parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window for calculations (default: 50)",
    )
    orderflow_parser.add_argument(
        "--vpin-bucket-size",
        type=float,
        default=10000.0,
        help="Volume per bucket for VPIN (default: 10000)",
    )
    orderflow_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
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


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Handle the portfolio subcommand for risk metrics analysis.

    Calculates and displays comprehensive portfolio risk metrics including
    Sharpe ratio, Sortino ratio, max drawdown, VaR, and Calmar ratio.

    Args:
        args: Parsed command-line arguments including features path and filters.

    Returns:
        Exit code (0 for success, 1 for missing data).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.risk_metrics import calculate_all_metrics

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Run 'imst process --build-features' first.")
        return 1

    df = pl.read_parquet(features_path)

    if "return_1d" not in df.columns:
        print("Error: Features file must contain 'return_1d' column")
        return 1

    # Filter by asset if specified
    if args.asset:
        if "asset_id" in df.columns:
            df = df.filter(pl.col("asset_id") == args.asset)
        elif "ticker" in df.columns:
            df = df.filter(pl.col("ticker") == args.asset)

        if df.height == 0:
            print(f"Error: No data found for asset '{args.asset}'")
            return 1

    returns = df["return_1d"].drop_nulls()

    if returns.len() == 0:
        print("Error: No valid return data found")
        return 1

    metrics = calculate_all_metrics(
        returns,
        risk_free_rate=args.risk_free_rate,
        var_confidence=args.var_confidence,
    )

    if args.json:
        print(json_module.dumps(metrics, indent=2))
    else:
        print("\n=== Portfolio Risk Metrics ===\n")
        print(f"Data points:        {returns.len()}")
        if args.asset:
            print(f"Asset:              {args.asset}")
        print()
        print(f"Total Return:       {metrics['total_return']:+.2%}")
        print(f"Annualized Return:  {metrics['annualized_return']:+.2%}")
        print(f"Volatility (Ann.):  {metrics['volatility']:.2%}")
        print()
        print(f"Sharpe Ratio:       {metrics['sharpe']:.4f}")
        print(f"Sortino Ratio:      {metrics['sortino']:.4f}")
        print(f"Calmar Ratio:       {metrics['calmar']:.4f}")
        print()
        print(f"Max Drawdown:       {metrics['max_drawdown']:.2%}")
        print(f"VaR ({args.var_confidence:.0%}):          {metrics['var']:.2%}")
        print()

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


def cmd_correlation(args: argparse.Namespace) -> int:
    """Handle the correlation subcommand for asset relationship analysis.

    Calculates and displays correlation metrics between assets including
    pairwise correlations and rolling correlation time series.

    Args:
        args: Parsed command-line arguments including feature path and options.

    Returns:
        Exit code (0 for success, 1 for missing data).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.correlation import (
        calculate_asset_correlation,
        calculate_correlation_summary,
        calculate_rolling_correlation,
    )

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if "return_1d" not in df.columns:
        print("Error: Features file must contain 'return_1d' column")
        return 1

    # Determine asset column
    asset_col = "asset_id" if "asset_id" in df.columns else "ticker"
    if asset_col not in df.columns:
        print("Error: Features file must contain 'asset_id' or 'ticker' column")
        return 1

    if args.asset1 and args.asset2:
        # Pairwise rolling correlation
        rolling_corr = calculate_rolling_correlation(
            df,
            args.asset1,
            args.asset2,
            return_col="return_1d",
            asset_col=asset_col,
            window=args.rolling_window,
        )

        if rolling_corr.height == 0:
            print(f"Error: Could not calculate correlation for {args.asset1} and {args.asset2}")
            return 1

        valid_corr = rolling_corr.filter(pl.col("rolling_correlation").is_not_null())
        if valid_corr.height > 0:
            avg_corr = valid_corr["rolling_correlation"].mean()
            min_corr = valid_corr["rolling_correlation"].min()
            max_corr = valid_corr["rolling_correlation"].max()
            latest_corr = valid_corr["rolling_correlation"].to_list()[-1]

            if args.json:
                result = {
                    "asset1": args.asset1,
                    "asset2": args.asset2,
                    "window": args.rolling_window,
                    "avg_correlation": float(avg_corr) if avg_corr else None,
                    "min_correlation": float(min_corr) if min_corr else None,
                    "max_correlation": float(max_corr) if max_corr else None,
                    "latest_correlation": latest_corr,
                }
                print(json_module.dumps(result, indent=2))
            else:
                print(f"\n=== Rolling Correlation: {args.asset1} vs {args.asset2} ===\n")
                print(f"Window:        {args.rolling_window} periods")
                print(f"Data points:   {valid_corr.height}")
                print()
                print(f"Average:       {avg_corr:.4f}" if avg_corr else "Average:       N/A")
                print(f"Minimum:       {min_corr:.4f}" if min_corr else "Minimum:       N/A")
                print(f"Maximum:       {max_corr:.4f}" if max_corr else "Maximum:       N/A")
                print(f"Latest:        {latest_corr:.4f}")
                print()
    else:
        # Full correlation summary
        summary = calculate_correlation_summary(
            df, return_col="return_1d", asset_col=asset_col
        )

        if args.json:
            # Convert correlation matrix to dict for JSON
            corr_matrix = summary["correlation_matrix"]
            matrix_dict = {
                "assets": corr_matrix["asset"].to_list(),
                "correlations": {
                    col: corr_matrix[col].to_list()
                    for col in corr_matrix.columns if col != "asset"
                },
            }
            result = {
                "avg_correlation": summary["avg_correlation"],
                "min_correlation": summary["min_correlation"],
                "max_correlation": summary["max_correlation"],
                "highly_correlated_pairs": [
                    {"asset1": p[0], "asset2": p[1], "correlation": p[2]}
                    for p in summary["highly_correlated"]
                ],
                "negatively_correlated_pairs": [
                    {"asset1": p[0], "asset2": p[1], "correlation": p[2]}
                    for p in summary["negatively_correlated"]
                ],
                "total_pairs": summary["total_pairs"],
                "correlation_matrix": matrix_dict,
            }
            print(json_module.dumps(result, indent=2, default=str))
        else:
            print("\n=== Correlation Analysis Summary ===\n")
            print(f"Total asset pairs: {summary['total_pairs']}")
            print(f"Average correlation: {summary['avg_correlation']:.4f}")
            print()

            if summary["max_correlation"]:
                pair = summary["max_correlation"]["pair"]
                val = summary["max_correlation"]["value"]
                print(f"Highest correlation: {pair[0]} / {pair[1]} = {val:.4f}")

            if summary["min_correlation"]:
                pair = summary["min_correlation"]["pair"]
                val = summary["min_correlation"]["value"]
                print(f"Lowest correlation:  {pair[0]} / {pair[1]} = {val:.4f}")

            if summary["highly_correlated"]:
                print(f"\nHighly correlated pairs (>0.7): {len(summary['highly_correlated'])}")
                for p in summary["highly_correlated"][:5]:
                    print(f"  {p[0]} / {p[1]}: {p[2]:.4f}")

            if summary["negatively_correlated"]:
                print(f"\nNegatively correlated pairs (<-0.3): {len(summary['negatively_correlated'])}")
                for p in summary["negatively_correlated"][:5]:
                    print(f"  {p[0]} / {p[1]}: {p[2]:.4f}")

            print()

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle the validate subcommand for data quality checks.

    Runs comprehensive data quality validation and reports any issues
    found in the specified data file.

    Args:
        args: Parsed command-line arguments including data path and validation type.

    Returns:
        Exit code (0 for success/no issues, 1 for errors found).
    """
    import json as json_module

    import polars as pl

    from imst_quant.utils.data_quality import generate_quality_report

    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return 1

    # Load data based on extension
    if data_path.suffix == ".parquet":
        df = pl.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pl.read_csv(data_path)
    else:
        print(f"Error: Unsupported file format: {data_path.suffix}")
        return 1

    # Run quality checks
    report = generate_quality_report(df)

    if args.json:
        print(json_module.dumps(report, indent=2, default=str))
    else:
        print(f"\n=== Data Quality Report: {data_path.name} ===\n")
        print(f"Rows:          {report['total_rows']:,}")
        print(f"Columns:       {report['total_columns']}")
        print(f"Quality Score: {report['quality_score']:.1%}")
        print()
        print(f"Issues Found:  {report['summary']['total_issues']}")
        print(f"  Errors:      {report['summary']['errors']}")
        print(f"  Warnings:    {report['summary']['warnings']}")
        print(f"  Info:        {report['summary']['info']}")
        print()

        if report["issues"]:
            print("--- Issues ---")
            for issue in report["issues"]:
                severity_icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}
                icon = severity_icon.get(issue["severity"], "•")
                print(f"  {icon} [{issue['severity'].upper()}] {issue['message']}")
            print()

        if report["recommendations"]:
            print("--- Recommendations ---")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
            print()

    # Return 1 if there are errors
    return 1 if report["summary"]["errors"] > 0 else 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle the report subcommand for generating backtest reports.

    Runs a backtest and exports results in specified formats (JSON, CSV, HTML).

    Args:
        args: Parsed command-line arguments including output path and format.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.trading.backtest import run_backtest
    from imst_quant.trading.report import (
        export_csv,
        export_html,
        export_json,
        generate_report,
    )
    from imst_quant.utils.risk_metrics import calculate_all_metrics

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    print(f"Running backtest on {features_path}...")

    # Run backtest
    backtest_results = run_backtest(features_path, transaction_cost=0.001)

    # Load data for additional metrics
    df = pl.read_parquet(features_path)
    if "return_1d" in df.columns:
        returns = df["return_1d"].drop_nulls()
        risk_metrics = calculate_all_metrics(returns)
        backtest_results.update(risk_metrics)

    # Generate report
    report = generate_report(
        backtest_results,
        strategy_name=args.strategy,
        parameters={"transaction_cost": 0.001},
        daily_pnl=returns if "return_1d" in df.columns else None,
    )

    # Determine output path
    output_base = Path(args.output) if args.output else gold_dir / f"report_{args.strategy}"

    # Export in requested formats
    created_files = []

    if args.format in ["json", "all"]:
        json_path = export_json(report, output_base.with_suffix(".json"))
        created_files.append(json_path)
        print(f"  JSON report: {json_path}")

    if args.format in ["csv", "all"]:
        csv_paths = export_csv(report, output_base)
        created_files.extend(csv_paths)
        for p in csv_paths:
            print(f"  CSV report:  {p}")

    if args.format in ["html", "all"]:
        html_path = export_html(report, output_base.with_suffix(".html"), title=f"Backtest: {args.strategy}")
        created_files.append(html_path)
        print(f"  HTML report: {html_path}")

    print(f"\nGenerated {len(created_files)} report file(s).")
    return 0


def cmd_montecarlo(args: argparse.Namespace) -> int:
    """Handle the montecarlo subcommand for risk simulation.

    Runs Monte Carlo simulations to estimate portfolio risk metrics including
    VaR, Expected Shortfall, and return distributions.

    Args:
        args: Parsed command-line arguments including simulation parameters.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.monte_carlo import MonteCarloSimulator

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if "return_1d" not in df.columns:
        print("Error: Features file must contain 'return_1d' column")
        return 1

    returns = df["return_1d"].drop_nulls()

    if returns.len() < 30:
        print("Error: Insufficient data for Monte Carlo simulation (need at least 30 observations)")
        return 1

    print(f"Running {args.simulations:,} Monte Carlo simulations...")
    print(f"Method: {args.method}, Horizon: {args.horizon} days, Confidence: {args.confidence:.0%}")

    simulator = MonteCarloSimulator(
        returns,
        n_simulations=args.simulations,
        seed=args.seed,
    )

    if args.method == "historical":
        result = simulator.run_historical_simulation(horizon=args.horizon)
    elif args.method == "parametric":
        result = simulator.run_parametric_simulation(horizon=args.horizon)
    else:  # gbm
        result = simulator.run_gbm_simulation(horizon=args.horizon)

    var = simulator.var_simulation(confidence=args.confidence)
    es = simulator.expected_shortfall(confidence=args.confidence)

    if args.json:
        output = {
            "method": args.method,
            "simulations": args.simulations,
            "horizon": args.horizon,
            "confidence": args.confidence,
            "var": float(var),
            "expected_shortfall": float(es),
            "mean_return": float(result.mean_return),
            "std_return": float(result.std_return),
            "percentiles": {str(k): float(v) for k, v in result.percentiles.items()},
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Monte Carlo Simulation Results ===\n")
        print(f"Simulations:       {args.simulations:,}")
        print(f"Horizon:           {args.horizon} trading days")
        print(f"Method:            {args.method}")
        print()
        print(f"VaR ({args.confidence:.0%}):        {var:.2%}")
        print(f"Expected Shortfall: {es:.2%}")
        print()
        print(f"Mean Return:       {result.mean_return:.2%}")
        print(f"Std Deviation:     {result.std_return:.2%}")
        print()
        print("Percentiles:")
        for pct, val in sorted(result.percentiles.items()):
            print(f"  {pct}th:           {val:.2%}")
        print()

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the benchmark subcommand for relative performance analysis.

    Compares strategy returns against a benchmark to calculate alpha, beta,
    tracking error, information ratio, and capture ratios.

    Args:
        args: Parsed command-line arguments including strategy and benchmark paths.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    import polars as pl

    from imst_quant.utils.benchmark import BenchmarkAnalyzer

    strategy_path = Path(args.strategy)

    if not strategy_path.exists():
        print(f"Error: Strategy file not found at {strategy_path}")
        return 1

    # Load strategy returns
    if strategy_path.suffix == ".parquet":
        strat_df = pl.read_parquet(strategy_path)
    else:
        strat_df = pl.read_csv(strategy_path)

    return_col = "return_1d" if "return_1d" in strat_df.columns else "return"
    if return_col not in strat_df.columns:
        print("Error: Strategy file must contain 'return_1d' or 'return' column")
        return 1

    strategy_returns = strat_df[return_col].drop_nulls()

    # Load or generate benchmark returns
    if args.benchmark:
        bench_path = Path(args.benchmark)
        if bench_path.exists():
            if bench_path.suffix == ".parquet":
                bench_df = pl.read_parquet(bench_path)
            else:
                bench_df = pl.read_csv(bench_path)
            bench_return_col = "return_1d" if "return_1d" in bench_df.columns else "return"
            benchmark_returns = bench_df[bench_return_col].drop_nulls()
        else:
            # Try to fetch benchmark data using ticker
            print(f"Fetching benchmark data for {args.benchmark}...")
            try:
                import yfinance as yf
                ticker = yf.Ticker(args.benchmark)
                hist = ticker.history(period="5y")
                benchmark_returns = pl.Series(hist["Close"].pct_change().dropna().values)
            except Exception as e:
                print(f"Error: Could not fetch benchmark data: {e}")
                return 1
    else:
        # Use equal-weight benchmark from strategy data if available
        print("Warning: No benchmark specified, using zero benchmark (absolute returns)")
        benchmark_returns = pl.Series([0.0] * strategy_returns.len())

    # Align series lengths
    min_len = min(strategy_returns.len(), benchmark_returns.len())
    strategy_returns = strategy_returns.tail(min_len)
    benchmark_returns = benchmark_returns.tail(min_len)

    analyzer = BenchmarkAnalyzer(
        strategy_returns,
        benchmark_returns,
        risk_free_rate=args.risk_free_rate,
    )

    metrics = analyzer.calculate_all_metrics()

    if args.json:
        output = {
            "alpha": float(metrics.alpha),
            "beta": float(metrics.beta),
            "r_squared": float(metrics.r_squared),
            "tracking_error": float(metrics.tracking_error),
            "information_ratio": float(metrics.information_ratio),
            "up_capture": float(metrics.up_capture),
            "down_capture": float(metrics.down_capture),
            "capture_ratio": float(metrics.capture_ratio),
            "excess_return": float(metrics.excess_return),
            "correlation": float(metrics.correlation),
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Benchmark Comparison ===\n")
        print(f"Data points:       {min_len}")
        print()
        print(f"Alpha (annualized): {metrics.alpha:.2%}")
        print(f"Beta:               {metrics.beta:.2f}")
        print(f"R-squared:          {metrics.r_squared:.2%}")
        print()
        print(f"Tracking Error:     {metrics.tracking_error:.2%}")
        print(f"Information Ratio:  {metrics.information_ratio:.2f}")
        print()
        print(f"Up Capture:         {metrics.up_capture:.1%}")
        print(f"Down Capture:       {metrics.down_capture:.1%}")
        print(f"Capture Ratio:      {metrics.capture_ratio:.2f}")
        print()
        print(f"Excess Return:      {metrics.excess_return:.2%}")
        print(f"Correlation:        {metrics.correlation:.2f}")
        print()

    return 0


def cmd_regime(args: argparse.Namespace) -> int:
    """Handle the regime subcommand for market regime detection.

    Analyzes price/return data to detect market regimes including volatility
    states, trend direction, and combined regime classifications.

    Args:
        args: Parsed command-line arguments including regime type and parameters.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.regime_detection import (
        detect_combined_regime,
        detect_trend_regime,
        detect_volatility_regime,
        regime_statistics,
        regime_transition_matrix,
    )

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if "return_1d" not in df.columns:
        print("Error: Features file must contain 'return_1d' column")
        return 1

    # Filter by asset if specified
    asset_col = "asset_id" if "asset_id" in df.columns else "ticker"
    if args.asset and asset_col in df.columns:
        df = df.filter(pl.col(asset_col) == args.asset)
        if df.height == 0:
            print(f"Error: No data found for asset {args.asset}")
            return 1

    returns = df["return_1d"].drop_nulls()

    print(f"Detecting {args.type} regime(s)...")
    print(f"Data points: {returns.len()}")

    if args.type == "volatility":
        regimes = detect_volatility_regime(returns, window=args.volatility_window)
        regime_col = "volatility_regime"
    elif args.type == "trend":
        regimes = detect_trend_regime(returns, window=args.trend_window)
        regime_col = "trend_regime"
    else:  # combined
        regimes = detect_combined_regime(
            returns,
            vol_window=args.volatility_window,
            trend_window=args.trend_window,
        )
        regime_col = "combined_regime"

    stats = regime_statistics(regimes, regime_col)
    transitions = regime_transition_matrix(regimes, regime_col)

    if args.json:
        # Get current regime
        current = regimes[regime_col].to_list()[-1] if regimes.height > 0 else None

        output = {
            "type": args.type,
            "current_regime": str(current) if current else None,
            "statistics": {
                str(k): {
                    "count": int(v["count"]),
                    "percentage": float(v["percentage"]),
                    "avg_duration": float(v["avg_duration"]) if v.get("avg_duration") else None,
                }
                for k, v in stats.items()
            },
            "transition_matrix": {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in transitions.items()
            },
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Market Regime Analysis ({args.type}) ===\n")

        # Current regime
        if regimes.height > 0:
            current = regimes[regime_col].to_list()[-1]
            print(f"Current Regime:    {current}")
            print()

        # Regime statistics
        print("Regime Distribution:")
        for regime, data in sorted(stats.items(), key=lambda x: -x[1]["percentage"]):
            pct = data["percentage"]
            count = data["count"]
            duration = data.get("avg_duration")
            duration_str = f", avg duration: {duration:.1f} days" if duration else ""
            print(f"  {regime:15} {pct:6.1%} ({count} periods{duration_str})")
        print()

        # Transition probabilities
        print("Transition Probabilities:")
        for from_regime, to_probs in sorted(transitions.items()):
            probs_str = ", ".join(
                f"{to_r}: {p:.1%}" for to_r, p in sorted(to_probs.items())
            )
            print(f"  {from_regime:15} -> {probs_str}")
        print()

    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    """Handle the journal subcommand for trade logging and analysis.

    Provides trade journaling functionality including logging new trades,
    closing positions, viewing statistics, and exporting data.

    Args:
        args: Parsed command-line arguments including action and trade details.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    from imst_quant.config.settings import Settings
    from imst_quant.utils.trade_journal import TradeEntry, TradeJournal

    settings = Settings()
    journal_path = Path(settings.data.gold_dir) / "trade_journal.json"
    journal = TradeJournal(journal_path)

    if args.action == "log":
        if not args.symbol or not args.direction or not args.entry_price or not args.quantity:
            print("Error: --symbol, --direction, --entry-price, and --quantity are required for logging")
            return 1

        entry = TradeEntry(
            symbol=args.symbol.upper(),
            direction=args.direction,
            entry_price=args.entry_price,
            quantity=args.quantity,
            entry_reason=args.reason or "",
            setup_type=args.setup or "",
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
        )

        # Calculate risk if stop loss provided
        if args.stop_loss:
            if args.direction == "long":
                risk_per_share = args.entry_price - args.stop_loss
            else:
                risk_per_share = args.stop_loss - args.entry_price
            entry.risk_amount = abs(risk_per_share * args.quantity)

        trade_id = journal.log_entry(entry)
        print(f"Trade logged: {trade_id}")
        print(f"  Symbol:    {entry.symbol}")
        print(f"  Direction: {entry.direction}")
        print(f"  Entry:     ${entry.entry_price:.2f}")
        print(f"  Quantity:  {entry.quantity}")
        if entry.stop_loss:
            print(f"  Stop Loss: ${entry.stop_loss:.2f}")
        if entry.risk_amount:
            print(f"  Risk:      ${entry.risk_amount:.2f}")

    elif args.action == "close":
        if not args.trade_id or not args.exit_price:
            print("Error: --trade-id and --exit-price are required for closing")
            return 1

        trade = journal.close_trade(
            args.trade_id,
            exit_price=args.exit_price,
            exit_reason=args.reason or "",
        )

        if trade is None:
            print(f"Error: Trade {args.trade_id} not found")
            return 1

        print(f"Trade closed: {trade.trade_id}")
        print(f"  Symbol:   {trade.symbol}")
        print(f"  Entry:    ${trade.entry_price:.2f}")
        print(f"  Exit:     ${trade.exit_price:.2f}")
        print(f"  P&L:      ${trade.pnl:+.2f} ({trade.pnl_percent:+.2%})")
        if trade.r_multiple is not None:
            print(f"  R-Multiple: {trade.r_multiple:+.2f}R")

    elif args.action == "list":
        open_trades = journal.get_open_trades()
        closed_trades = journal.get_closed_trades()

        if args.json:
            output = {
                "open": [t.to_dict() for t in open_trades],
                "closed": [t.to_dict() for t in closed_trades[-10:]],
            }
            print(json_module.dumps(output, indent=2, default=str))
        else:
            print(f"\n=== Open Positions ({len(open_trades)}) ===\n")
            for t in open_trades:
                print(f"  [{t.trade_id}] {t.symbol} {t.direction.upper()} @ ${t.entry_price:.2f} x {t.quantity}")

            print(f"\n=== Recent Closed ({min(10, len(closed_trades))}) ===\n")
            for t in closed_trades[-10:]:
                pnl_str = f"${t.pnl:+.2f}" if t.pnl else "N/A"
                print(f"  [{t.trade_id}] {t.symbol} {t.direction.upper()} P&L: {pnl_str}")

    elif args.action == "stats":
        stats = journal.get_statistics()

        if args.json:
            output = {
                "total_trades": stats.total_trades,
                "winning_trades": stats.winning_trades,
                "losing_trades": stats.losing_trades,
                "win_rate": stats.win_rate,
                "total_pnl": stats.total_pnl,
                "avg_pnl": stats.avg_pnl,
                "profit_factor": stats.profit_factor,
                "expectancy": stats.expectancy,
                "avg_r_multiple": stats.avg_r_multiple,
                "largest_win": stats.largest_win,
                "largest_loss": stats.largest_loss,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(journal.generate_summary_report())

    elif args.action == "report":
        by_symbol = journal.get_performance_by_symbol()
        by_setup = journal.get_performance_by_setup()

        print("\n=== Performance by Symbol ===\n")
        for symbol, stats in sorted(by_symbol.items()):
            print(f"  {symbol:8} Win Rate: {stats.win_rate:.1%}, P&L: ${stats.total_pnl:+.2f}, Trades: {stats.total_trades}")

        print("\n=== Performance by Setup ===\n")
        for setup, stats in sorted(by_setup.items()):
            print(f"  {setup:15} Win Rate: {stats.win_rate:.1%}, P&L: ${stats.total_pnl:+.2f}, Trades: {stats.total_trades}")

    elif args.action == "export":
        output_path = Path(args.output) if args.output else journal_path.with_suffix(".csv")
        journal.export_to_csv(output_path)
        print(f"Journal exported to: {output_path}")

    return 0


def cmd_seasonality(args: argparse.Namespace) -> int:
    """Handle the seasonality subcommand for temporal pattern analysis.

    Analyzes seasonal patterns in returns including day-of-week effects,
    monthly seasonality, and turn-of-month patterns.

    Args:
        args: Parsed command-line arguments including analysis type.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.seasonality import (
        analyze_day_of_week_effect,
        analyze_holiday_effect,
        analyze_monthly_seasonality,
        analyze_turn_of_month_effect,
        generate_seasonality_report,
    )

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if "return_1d" not in df.columns:
        print("Error: Features file must contain 'return_1d' column")
        return 1

    # Filter by asset if specified
    asset_col = "asset_id" if "asset_id" in df.columns else "ticker"
    if args.asset and asset_col in df.columns:
        df = df.filter(pl.col(asset_col) == args.asset)
        if df.height == 0:
            print(f"Error: No data found for asset {args.asset}")
            return 1

    date_col = "date" if "date" in df.columns else "timestamp"

    if args.type == "all":
        if args.json:
            dow = analyze_day_of_week_effect(df, date_col)
            monthly = analyze_monthly_seasonality(df, date_col)
            tom = analyze_turn_of_month_effect(df, date_col)
            holiday = analyze_holiday_effect(df, date_col)

            output = {
                "day_of_week": {
                    "best": str(dow.best_period),
                    "worst": str(dow.worst_period),
                    "spread": dow.spread,
                    "significant": dow.is_significant,
                },
                "monthly": {
                    "best": str(monthly.best_period),
                    "worst": str(monthly.worst_period),
                    "spread": monthly.spread,
                    "significant": monthly.is_significant,
                },
                "turn_of_month": tom,
                "holiday": holiday,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(generate_seasonality_report(df, date_col))
    else:
        if args.type == "dow":
            result = analyze_day_of_week_effect(df, date_col)
            print(f"\n=== Day of Week Effect ===\n")
            print(f"Best Day:    {result.best_period}")
            print(f"Worst Day:   {result.worst_period}")
            print(f"Spread:      {result.spread:.4%}")
            print(f"Significant: {'Yes' if result.is_significant else 'No'}")
        elif args.type == "monthly":
            result = analyze_monthly_seasonality(df, date_col)
            print(f"\n=== Monthly Seasonality ===\n")
            print(f"Best Month:  {result.best_period}")
            print(f"Worst Month: {result.worst_period}")
            print(f"Spread:      {result.spread:.4%}")
            print(f"Significant: {'Yes' if result.is_significant else 'No'}")
        elif args.type == "tom":
            result = analyze_turn_of_month_effect(df, date_col)
            print(f"\n=== Turn of Month Effect ===\n")
            for key, val in result.items():
                if isinstance(val, float):
                    print(f"{key:20}: {val:.4%}" if abs(val) < 1 else f"{key:20}: {val:.1f}")
                else:
                    print(f"{key:20}: {val}")
        elif args.type == "holiday":
            result = analyze_holiday_effect(df, date_col)
            print(f"\n=== Holiday Effect ===\n")
            for key, val in result.items():
                if isinstance(val, float):
                    print(f"{key:20}: {val:.4%}" if abs(val) < 1 else f"{key:20}: {val:.1f}")
                else:
                    print(f"{key:20}: {val}")

    return 0


def cmd_liquidity(args: argparse.Namespace) -> int:
    """Handle the liquidity subcommand for market liquidity analysis.

    Analyzes market liquidity including Amihud illiquidity, spread estimation,
    volume profiles, and market impact estimation.

    Args:
        args: Parsed command-line arguments including order size for impact.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.liquidity_analysis import (
        analyze_liquidity,
        estimate_market_impact,
    )

    settings = Settings()
    gold_dir = Path(settings.data.gold_dir)

    features_path = Path(args.features) if args.features else gold_dir / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    # Filter by asset if specified
    asset_col = "asset_id" if "asset_id" in df.columns else "ticker"
    if args.asset and asset_col in df.columns:
        df = df.filter(pl.col(asset_col) == args.asset)
        if df.height == 0:
            print(f"Error: No data found for asset {args.asset}")
            return 1

    metrics = analyze_liquidity(df)

    if args.json:
        output = {
            "avg_daily_volume": metrics.avg_daily_volume,
            "avg_dollar_volume": metrics.avg_dollar_volume,
            "volume_volatility": metrics.volume_volatility,
            "amihud_illiquidity": metrics.amihud_illiquidity,
            "avg_spread_estimate": metrics.avg_spread_estimate,
            "volume_concentration": metrics.volume_concentration,
            "liquidity_score": metrics.liquidity_score,
            "days_analyzed": metrics.days_analyzed,
        }

        if args.order_size:
            impact = estimate_market_impact(df, args.order_size)
            output["market_impact"] = impact

        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Liquidity Analysis ===\n")
        if args.asset:
            print(f"Asset:              {args.asset}")
        print(f"Days Analyzed:      {metrics.days_analyzed}")
        print()
        print(f"Avg Daily Volume:   {metrics.avg_daily_volume:,.0f}")
        print(f"Avg Dollar Volume:  ${metrics.avg_dollar_volume:,.0f}")
        print(f"Volume Volatility:  {metrics.volume_volatility:.2%}")
        print()
        print(f"Amihud Illiquidity: {metrics.amihud_illiquidity:.2e}")
        print(f"Est. Spread:        {metrics.avg_spread_estimate:.4%}")
        print(f"Vol. Concentration: {metrics.volume_concentration:.1%}")
        print()
        print(f"Liquidity Score:    {metrics.liquidity_score:.1f}/100")

        if args.order_size:
            impact = estimate_market_impact(df, args.order_size)
            print()
            print(f"--- Market Impact (Order: {args.order_size:,.0f} shares) ---")
            print(f"Participation Rate: {impact['participation_rate']:.2%}")
            print(f"Expected Slippage:  {impact['expected_slippage_pct']:.4%} ({impact['expected_slippage_bps']:.1f} bps)")

        print()

    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    """Handle pairs trading cointegration analysis.

    Tests for cointegration between asset pairs and generates trading signals.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module
    import pandas as pd
    from imst_quant.utils.cointegration import (
        test_cointegration,
        find_cointegrated_pairs,
        generate_pairs_signal,
    )

    features_path = args.features or "gold/features.parquet"

    try:
        df = pd.read_parquet(features_path)
    except FileNotFoundError:
        print(f"Error: Features file not found: {features_path}")
        return 1

    if args.scan:
        # Scan all pairs for cointegration
        print("Scanning for cointegrated pairs...")

        # Extract returns columns
        return_cols = [c for c in df.columns if "return" in c.lower()]
        if not return_cols:
            return_cols = df.select_dtypes(include=[float]).columns.tolist()[:10]

        prices = df[return_cols].cumsum() + 100  # Convert returns to prices

        pairs = find_cointegrated_pairs(
            prices,
            significance=args.significance,
            min_half_life=args.min_half_life,
            max_half_life=args.max_half_life,
        )

        if args.json:
            print(json_module.dumps(pairs, indent=2))
        else:
            print(f"\n=== Cointegrated Pairs Found: {len(pairs)} ===\n")
            for pair in pairs[:20]:  # Show top 20
                print(f"{pair['asset_y']:12} / {pair['asset_x']:12}  "
                      f"p={pair['p_value']:.4f}  "
                      f"β={pair['hedge_ratio']:.3f}  "
                      f"HL={pair['half_life']:.1f}d")

    elif args.asset1 and args.asset2:
        # Test specific pair
        if args.asset1 not in df.columns or args.asset2 not in df.columns:
            print(f"Error: Assets not found in data")
            return 1

        y = df[args.asset1]
        x = df[args.asset2]

        result = test_cointegration(y, x, args.significance)

        if args.json:
            output = {
                "asset_y": args.asset1,
                "asset_x": args.asset2,
                "is_cointegrated": result.is_cointegrated,
                "adf_statistic": result.adf_statistic,
                "p_value": result.p_value,
                "hedge_ratio": result.hedge_ratio,
                "half_life": result.half_life,
                "spread_mean": result.spread_mean,
                "spread_std": result.spread_std,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(f"\n=== Cointegration Test: {args.asset1} / {args.asset2} ===\n")
            print(f"Cointegrated:   {'Yes' if result.is_cointegrated else 'No'}")
            print(f"ADF Statistic:  {result.adf_statistic:.4f}")
            print(f"P-Value:        {result.p_value:.4f}")
            print(f"Hedge Ratio:    {result.hedge_ratio:.4f}")
            print(f"Half-Life:      {result.half_life:.1f} periods")
            print(f"Spread Mean:    {result.spread_mean:.4f}")
            print(f"Spread Std:     {result.spread_std:.4f}")

            if result.is_cointegrated:
                signal = generate_pairs_signal(y, x, result.hedge_ratio)
                print(f"\n--- Current Signal ---")
                print(f"Z-Score:        {signal.zscore:.2f}")
                positions = {-1: "Short Pair", 0: "Neutral", 1: "Long Pair"}
                print(f"Position:       {positions[signal.position]}")

    else:
        print("Error: Specify --scan or both --asset1 and --asset2")
        return 1

    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Handle portfolio optimization.

    Optimizes portfolio weights using various methods.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module
    import pandas as pd
    from imst_quant.utils.portfolio_optimization import (
        mean_variance_optimize,
        risk_parity_optimize,
        hierarchical_risk_parity,
        OptimizationObjective,
    )

    features_path = args.features or "gold/features.parquet"

    try:
        df = pd.read_parquet(features_path)
    except FileNotFoundError:
        print(f"Error: Features file not found: {features_path}")
        return 1

    # Extract returns columns
    return_cols = [c for c in df.columns if "return" in c.lower()]
    if not return_cols:
        return_cols = df.select_dtypes(include=[float]).columns.tolist()[:10]

    returns = df[return_cols]

    # Select optimization method
    if args.method == "sharpe":
        result = mean_variance_optimize(
            returns,
            OptimizationObjective.MAX_SHARPE,
            allow_short=args.allow_short,
        )
    elif args.method == "variance":
        result = mean_variance_optimize(
            returns,
            OptimizationObjective.MIN_VARIANCE,
            allow_short=args.allow_short,
        )
    elif args.method == "risk-parity":
        result = risk_parity_optimize(returns)
    elif args.method == "hrp":
        result = hierarchical_risk_parity(returns)
    else:
        print(f"Error: Unknown method: {args.method}")
        return 1

    if args.json:
        output = {
            "method": args.method,
            "weights": result.weights,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "diversification_ratio": result.diversification_ratio,
            "effective_n": result.effective_n,
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Portfolio Optimization ({args.method}) ===\n")
        print(f"Expected Return:      {result.expected_return:.2%}")
        print(f"Volatility:           {result.volatility:.2%}")
        print(f"Sharpe Ratio:         {result.sharpe_ratio:.3f}")
        print(f"Diversification:      {result.diversification_ratio:.2f}")
        print(f"Effective N:          {result.effective_n:.1f}")
        print()
        print("--- Optimal Weights ---")
        for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
            if abs(weight) > 0.001:
                print(f"  {asset:20} {weight:7.2%}")

    return 0


def cmd_orderflow(args: argparse.Namespace) -> int:
    """Handle order flow analysis.

    Analyzes trade flow for informed trading indicators.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module
    import pandas as pd
    from imst_quant.utils.order_flow import analyze_order_flow

    try:
        df = pd.read_parquet(args.data)
    except FileNotFoundError:
        try:
            df = pd.read_csv(args.data)
        except FileNotFoundError:
            print(f"Error: Data file not found: {args.data}")
            return 1

    # Find price and volume columns
    price_col = None
    volume_col = None
    for col in df.columns:
        col_lower = col.lower()
        if price_col is None and "price" in col_lower:
            price_col = col
        if volume_col is None and "volume" in col_lower:
            volume_col = col

    if price_col is None or volume_col is None:
        print("Error: Could not find price and volume columns")
        return 1

    prices = df[price_col]
    volumes = df[volume_col]

    metrics = analyze_order_flow(
        prices,
        volumes,
        window=args.window,
        vpin_bucket_size=args.vpin_bucket_size,
    )

    if args.json:
        output = {
            "volume_imbalance": metrics.volume_imbalance,
            "ofi": metrics.ofi,
            "vpin": metrics.vpin,
            "trade_flow_toxicity": metrics.trade_flow_toxicity,
            "buy_volume_pct": metrics.buy_volume_pct,
            "sell_volume_pct": metrics.sell_volume_pct,
            "large_trade_ratio": metrics.large_trade_ratio,
            "aggressor_imbalance": metrics.aggressor_imbalance,
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== Order Flow Analysis ===\n")
        print(f"Volume Imbalance:    {metrics.volume_imbalance:+.3f}")
        print(f"Order Flow Imbal:    {metrics.ofi:+.3f}")
        print(f"VPIN:                {metrics.vpin:.3f}")
        print(f"Trade Flow Toxicity: {metrics.trade_flow_toxicity:.2f}")
        print()
        print(f"Buy Volume %:        {metrics.buy_volume_pct:.1%}")
        print(f"Sell Volume %:       {metrics.sell_volume_pct:.1%}")
        print(f"Large Trade Ratio:   {metrics.large_trade_ratio:.1%}")
        print(f"Aggressor Imbalance: {metrics.aggressor_imbalance:+.3f}")
        print()

        # Interpretation
        if metrics.vpin > 0.4:
            print("⚠️  High VPIN: Elevated informed trading probability")
        if abs(metrics.volume_imbalance) > 0.5:
            side = "buying" if metrics.volume_imbalance > 0 else "selling"
            print(f"⚠️  Strong {side} pressure detected")

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
        "portfolio": cmd_portfolio,
        "correlation": cmd_correlation,
        "validate": cmd_validate,
        "report": cmd_report,
        "montecarlo": cmd_montecarlo,
        "benchmark": cmd_benchmark,
        "regime": cmd_regime,
        "journal": cmd_journal,
        "seasonality": cmd_seasonality,
        "liquidity": cmd_liquidity,
        "pairs": cmd_pairs,
        "optimize": cmd_optimize,
        "orderflow": cmd_orderflow,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
