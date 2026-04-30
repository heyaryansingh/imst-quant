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

    # --- factors subcommand ---
    factors_parser = subparsers.add_parser(
        "factors", help="Factor exposure and risk decomposition analysis"
    )
    factors_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    factors_parser.add_argument(
        "--type",
        choices=["exposures", "decomposition", "attribution", "rolling"],
        default="exposures",
        help="Type of factor analysis (default: exposures)",
    )
    factors_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    factors_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window for rolling analysis (default: 60)",
    )
    factors_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- execution subcommand ---
    execution_parser = subparsers.add_parser(
        "execution", help="Execution quality and slippage analysis"
    )
    execution_parser.add_argument(
        "--trades", help="Path to trade log file (JSON or CSV)"
    )
    execution_parser.add_argument(
        "--estimate", action="store_true", help="Estimate slippage for hypothetical order"
    )
    execution_parser.add_argument(
        "--order-size", type=float, help="Order size for estimation"
    )
    execution_parser.add_argument(
        "--avg-volume", type=float, help="Average daily volume for estimation"
    )
    execution_parser.add_argument(
        "--spread-bps", type=float, default=5.0, help="Spread in basis points (default: 5)"
    )
    execution_parser.add_argument(
        "--volatility", type=float, default=0.02, help="Daily volatility (default: 0.02)"
    )
    execution_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- streaks subcommand ---
    streaks_parser = subparsers.add_parser(
        "streaks", help="Win/loss streak analysis and statistics"
    )
    streaks_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    streaks_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Return threshold for win classification (default: 0)",
    )
    streaks_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    streaks_parser.add_argument(
        "--ruin-analysis", action="store_true", help="Include gambler's ruin analysis"
    )
    streaks_parser.add_argument(
        "--bankroll",
        type=float,
        default=100.0,
        help="Bankroll units for ruin analysis (default: 100)",
    )
    streaks_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- volatility subcommand ---
    volatility_parser = subparsers.add_parser(
        "volatility", help="Volatility forecasting and analysis (EWMA, GARCH)"
    )
    volatility_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    volatility_parser.add_argument(
        "--method",
        choices=["ewma", "garch", "historical", "compare"],
        default="ewma",
        help="Volatility estimation method (default: ewma)",
    )
    volatility_parser.add_argument(
        "--span",
        type=int,
        default=20,
        help="EWMA span / Historical window (default: 20)",
    )
    volatility_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    volatility_parser.add_argument(
        "--forecast", action="store_true", help="Generate volatility forecasts"
    )
    volatility_parser.add_argument(
        "--cone", action="store_true", help="Generate volatility cone analysis"
    )
    volatility_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- distribution subcommand ---
    distribution_parser = subparsers.add_parser(
        "distribution", help="Returns distribution analysis (skewness, kurtosis, normality)"
    )
    distribution_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    distribution_parser.add_argument(
        "--asset", help="Filter to specific asset (default: all)"
    )
    distribution_parser.add_argument(
        "--normality", action="store_true", help="Run comprehensive normality tests"
    )
    distribution_parser.add_argument(
        "--tails", action="store_true", help="Analyze tail risk"
    )
    distribution_parser.add_argument(
        "--rolling-window",
        type=int,
        default=63,
        help="Window for rolling statistics (default: 63)",
    )
    distribution_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- signal subcommand ---
    signal_parser = subparsers.add_parser(
        "signal", help="Signal backtesting and validation"
    )
    signal_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    signal_parser.add_argument(
        "--signal-col", help="Column name containing the signal", required=True
    )
    signal_parser.add_argument(
        "--returns-col", default="returns", help="Column name containing returns (default: returns)"
    )
    signal_parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost as decimal (default: 0.001)",
    )
    signal_parser.add_argument(
        "--decay", action="store_true", help="Run signal decay analysis"
    )
    signal_parser.add_argument(
        "--max-lag",
        type=int,
        default=20,
        help="Maximum lag for decay analysis (default: 20)",
    )
    signal_parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap simulations (0 to skip)",
    )
    signal_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- health subcommand ---
    health_parser = subparsers.add_parser(
        "health", help="Check data pipeline health status"
    )
    health_parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days to check for recent data (default: 7)",
    )
    health_parser.add_argument(
        "--layer",
        choices=["raw", "bronze", "silver", "sentiment", "influence", "gold"],
        help="Check specific layer only",
    )
    health_parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Max acceptable data age in hours (default: 24)",
    )
    health_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- summary subcommand ---
    summary_parser = subparsers.add_parser(
        "summary", help="Quick portfolio metrics summary"
    )
    summary_parser.add_argument(
        "--features", help="Path to features parquet file (default: gold/features.parquet)"
    )
    summary_parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top holdings to show (default: 10)"
    )
    summary_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- exposure subcommand ---
    exposure_parser = subparsers.add_parser(
        "exposure", help="Analyze portfolio concentration and sector/asset exposure"
    )
    exposure_parser.add_argument(
        "--portfolio", help="Path to portfolio parquet file with [symbol, weight, sector, asset_type]"
    )
    exposure_parser.add_argument(
        "--geography", action="store_true", help="Include geographic exposure analysis"
    )
    exposure_parser.add_argument(
        "--sector-threshold", type=float, default=0.3, help="Sector concentration threshold (default: 0.3)"
    )
    exposure_parser.add_argument(
        "--position-threshold", type=float, default=0.1, help="Single position threshold (default: 0.1)"
    )
    exposure_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- risk-metrics subcommand ---
    risk_parser = subparsers.add_parser(
        "risk-metrics", help="Calculate comprehensive portfolio risk metrics"
    )
    risk_parser.add_argument(
        "--returns", help="Path to returns parquet file (default: gold/returns.parquet)"
    )
    risk_parser.add_argument(
        "--return-col", default="returns", help="Column name for returns (default: returns)"
    )
    risk_parser.add_argument(
        "--risk-free-rate", type=float, default=0.0001, help="Daily risk-free rate (default: 0.0001)"
    )
    risk_parser.add_argument(
        "--periods", type=int, default=252, help="Trading periods per year (default: 252)"
    )
    risk_parser.add_argument(
        "--var-confidence", type=float, default=0.95, help="VaR confidence level (default: 0.95)"
    )
    risk_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- quality-check subcommand ---
    quality_parser = subparsers.add_parser(
        "quality-check", help="Validate data quality and completeness"
    )
    quality_parser.add_argument(
        "--layer",
        choices=["bronze", "silver", "gold"],
        required=True,
        help="Data layer to check"
    )
    quality_parser.add_argument(
        "--date", help="Specific date to check (YYYY-MM-DD)"
    )
    quality_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix detected issues"
    )
    quality_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # --- signal-performance subcommand ---
    signal_perf_parser = subparsers.add_parser(
        "signal-performance", help="Analyze trading signal performance metrics"
    )
    signal_perf_parser.add_argument(
        "--data", required=True, help="Path to parquet file with signal and returns columns"
    )
    signal_perf_parser.add_argument(
        "--signal-col", default="signal", help="Signal column name (default: signal)"
    )
    signal_perf_parser.add_argument(
        "--returns-col", default="returns", help="Returns column name (default: returns)"
    )
    signal_perf_parser.add_argument(
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


def cmd_factors(args: argparse.Namespace) -> int:
    """Handle the factors subcommand for factor exposure analysis.

    Analyzes portfolio factor exposures using regression-based methods,
    including Fama-French style factor models.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.factor_analysis import (
        decompose_risk,
        estimate_factor_exposures,
        factor_attribution,
        generate_synthetic_factors,
        rolling_factor_exposures,
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

    # Generate synthetic factors if market data available
    # In production, would load real factor data
    print("Generating synthetic factor returns for analysis...")
    factor_returns = generate_synthetic_factors(returns, seed=42)

    if args.type == "exposures":
        exposures = estimate_factor_exposures(returns, factor_returns)

        if args.json:
            output = {
                "betas": exposures.betas,
                "alpha": exposures.alpha,
                "r_squared": exposures.r_squared,
                "adj_r_squared": exposures.adj_r_squared,
                "residual_vol": exposures.residual_vol,
                "t_stats": exposures.t_stats,
                "p_values": exposures.p_values,
                "factor_contributions": exposures.factor_contributions,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(f"\n=== Factor Exposures ===\n")
            print(f"Alpha (ann.):       {exposures.alpha:+.2%}")
            print(f"R-squared:          {exposures.r_squared:.2%}")
            print(f"Adj. R-squared:     {exposures.adj_r_squared:.2%}")
            print(f"Idiosyncratic Vol:  {exposures.residual_vol:.2%}")
            print()
            print("--- Factor Betas ---")
            for factor, beta in exposures.betas.items():
                t_stat = exposures.t_stats.get(factor, 0)
                p_val = exposures.p_values.get(factor, 1)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  {factor:8} {beta:+.3f}  (t={t_stat:.2f}) {sig}")
            print()

    elif args.type == "decomposition":
        decomp = decompose_risk(returns, factor_returns)

        if args.json:
            output = {
                "total_risk": decomp.total_risk,
                "systematic_risk": decomp.systematic_risk,
                "idiosyncratic_risk": decomp.idiosyncratic_risk,
                "factor_risks": decomp.factor_risks,
                "diversification_benefit": decomp.diversification_benefit,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(f"\n=== Risk Decomposition ===\n")
            print(f"Total Risk:         {decomp.total_risk:.2%}")
            print(f"Systematic Risk:    {decomp.systematic_risk:.2%}")
            print(f"Idiosyncratic Risk: {decomp.idiosyncratic_risk:.2%}")
            print()
            print("--- Factor Risk Contributions ---")
            for factor, risk in decomp.factor_risks.items():
                pct = risk / decomp.total_risk * 100 if decomp.total_risk > 0 else 0
                print(f"  {factor:8} {risk:.2%}  ({pct:.1f}% of total)")
            print()

    elif args.type == "attribution":
        attrib = factor_attribution(returns, factor_returns)

        if args.json:
            print(json_module.dumps(attrib, indent=2))
        else:
            print(f"\n=== Return Attribution ===\n")
            print(f"Total Return (ann.): {attrib['total']:.2%}")
            print()
            print("--- Sources ---")
            print(f"  Alpha:            {attrib['alpha']:+.2%}")
            for key in ["MKT", "SMB", "HML", "MOM"]:
                if key in attrib:
                    print(f"  {key:16} {attrib[key]:+.2%}")
            print(f"  Residual:         {attrib['residual']:+.2%}")
            print()

    elif args.type == "rolling":
        rolling = rolling_factor_exposures(returns, factor_returns, window=args.window)

        # Show last few values
        if args.json:
            output = rolling.tail(20).to_dicts()
            print(json_module.dumps(output, indent=2))
        else:
            print(f"\n=== Rolling Factor Exposures (last 10) ===\n")
            print(f"Window: {args.window} periods\n")
            tail = rolling.tail(10)
            for i, row in enumerate(tail.iter_rows(named=True)):
                betas_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in row.items()
                    if k not in ["alpha", "r_squared"] and v is not None
                )
                print(f"  [{i+1}] α={row['alpha']:.2%}  {betas_str}")
            print()

    return 0


def cmd_execution(args: argparse.Namespace) -> int:
    """Handle the execution subcommand for trade execution analysis.

    Analyzes execution quality including slippage and market impact.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    from imst_quant.utils.execution_analytics import (
        estimate_expected_slippage,
        generate_execution_report,
    )

    if args.estimate:
        # Estimate slippage for hypothetical order
        if not args.order_size or not args.avg_volume:
            print("Error: --order-size and --avg-volume required for estimation")
            return 1

        estimate = estimate_expected_slippage(
            order_size=args.order_size,
            avg_daily_volume=args.avg_volume,
            avg_spread_bps=args.spread_bps,
            volatility=args.volatility,
        )

        if args.json:
            print(json_module.dumps(estimate, indent=2))
        else:
            print(f"\n=== Execution Cost Estimate ===\n")
            print(f"Order Size:         {args.order_size:,.0f} shares")
            print(f"Avg Daily Volume:   {args.avg_volume:,.0f}")
            print(f"Participation Rate: {estimate['participation_rate']:.2%}")
            print()
            print("--- Cost Breakdown ---")
            print(f"  Spread Cost:      {estimate['spread_cost_bps']:.1f} bps")
            print(f"  Market Impact:    {estimate['market_impact_bps']:.1f} bps")
            print(f"  Timing Cost:      {estimate['timing_cost_bps']:.1f} bps")
            print(f"  -------------------")
            print(f"  Total Expected:   {estimate['total_expected_cost_bps']:.1f} bps")
            print()
            print(f"Recommendation: {estimate['recommendation']}")
            print()

    elif args.trades:
        # Analyze trade log
        import pandas as pd
        from imst_quant.utils.execution_analytics import (
            ExecutedTrade,
            OrderSide,
            OrderType,
            analyze_execution_quality,
        )

        try:
            if args.trades.endswith(".json"):
                trades_data = pd.read_json(args.trades)
            else:
                trades_data = pd.read_csv(args.trades)
        except FileNotFoundError:
            print(f"Error: Trade file not found: {args.trades}")
            return 1

        # Convert to ExecutedTrade objects
        trades = []
        for _, row in trades_data.iterrows():
            trade = ExecutedTrade(
                trade_id=str(row.get("trade_id", "")),
                symbol=str(row.get("symbol", "")),
                side=OrderSide.BUY if str(row.get("side", "")).lower() == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                decision_price=float(row.get("decision_price", row.get("order_price", 0))),
                order_price=float(row.get("order_price", 0)),
                fill_price=float(row.get("fill_price", 0)),
                quantity_ordered=float(row.get("quantity_ordered", row.get("quantity", 0))),
                quantity_filled=float(row.get("quantity_filled", row.get("quantity", 0))),
                commission=float(row.get("commission", 0)),
                venue=str(row.get("venue", "unknown")),
            )
            trades.append(trade)

        if not trades:
            print("Error: No trades found in file")
            return 1

        metrics = analyze_execution_quality(trades)

        if args.json:
            output = {
                "total_trades": metrics.total_trades,
                "fill_rate": metrics.fill_rate,
                "avg_slippage_bps": metrics.avg_slippage_bps,
                "total_slippage_cost": metrics.total_slippage_cost,
                "implementation_shortfall": metrics.implementation_shortfall,
                "commission_cost": metrics.commission_cost,
                "slippage_by_venue": metrics.slippage_by_venue,
                "slippage_by_size_bucket": metrics.slippage_by_size_bucket,
            }
            print(json_module.dumps(output, indent=2))
        else:
            print(generate_execution_report(trades, "Trade Log Analysis"))

    else:
        print("Error: Specify --trades <file> or --estimate with order parameters")
        return 1

    return 0


def cmd_streaks(args: argparse.Namespace) -> int:
    """Handle the streaks subcommand for win/loss streak analysis.

    Analyzes consecutive winning and losing periods in returns.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    import numpy as np
    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.streak_analysis import (
        analyze_streaks,
        calculate_gambler_ruin_prob,
        generate_streak_report,
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

    stats = analyze_streaks(returns, threshold=args.threshold)

    if args.json:
        output = {
            "total_periods": stats.total_periods,
            "win_rate": stats.win_rate,
            "max_win_streak": stats.max_win_streak,
            "max_loss_streak": stats.max_loss_streak,
            "avg_win_streak": stats.avg_win_streak,
            "avg_loss_streak": stats.avg_loss_streak,
            "current_streak": stats.current_streak,
            "expected_max_streak": stats.expected_max_streak,
            "runs_test_p": stats.streak_runs_test_p,
            "max_favorable_excursion": stats.max_favorable_excursion,
            "max_adverse_excursion": stats.max_adverse_excursion,
        }

        if args.ruin_analysis:
            # Calculate average win/loss for ruin analysis
            wins = returns.filter(returns > args.threshold)
            losses = returns.filter(returns <= args.threshold)
            avg_win = float(wins.mean()) if wins.len() > 0 else 0.01
            avg_loss = abs(float(losses.mean())) if losses.len() > 0 else 0.01

            ruin = calculate_gambler_ruin_prob(
                win_rate=stats.win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                bankroll_units=args.bankroll,
            )
            output["ruin_analysis"] = ruin

        print(json_module.dumps(output, indent=2))
    else:
        print(generate_streak_report(stats))

        if args.ruin_analysis:
            wins = returns.filter(returns > args.threshold)
            losses = returns.filter(returns <= args.threshold)
            avg_win = float(wins.mean()) if wins.len() > 0 else 0.01
            avg_loss = abs(float(losses.mean())) if losses.len() > 0 else 0.01

            ruin = calculate_gambler_ruin_prob(
                win_rate=stats.win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                bankroll_units=args.bankroll,
            )

            print("--- Gambler's Ruin Analysis ---")
            print(f"Risk/Reward Ratio:  {ruin['risk_reward']:.2f}")
            print(f"Edge per Trade:     {ruin['edge']:+.4f}")
            print(f"Kelly Fraction:     {ruin['kelly_fraction']:.1%}")
            print(f"Ruin Probability:   {ruin['ruin_probability']:.1%}")
            print(f"Survival Prob:      {ruin['survival_probability']:.1%}")
            print()

    return 0


def cmd_volatility(args: argparse.Namespace) -> int:
    """Handle the volatility subcommand for volatility forecasting.

    Analyzes volatility using EWMA, GARCH, or historical methods with
    optional forecasting and cone analysis.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.volatility_forecast import (
        compare_volatility_methods,
        ewma_volatility,
        garch_volatility,
        historical_volatility,
        volatility_cone,
        volatility_forecast,
    )

    settings = Settings()
    features_path = args.features or str(Path(settings.data.gold_dir) / "features.parquet")

    if not Path(features_path).exists():
        print(f"Error: Features file not found: {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    # Filter by asset if specified
    if args.asset and "asset" in df.columns:
        df = df.filter(pl.col("asset") == args.asset)

    if "returns" not in df.columns:
        print("Error: 'returns' column not found in features file")
        return 1

    returns = df["returns"].drop_nulls().to_pandas()

    output = {}

    if args.method == "compare":
        comparison = compare_volatility_methods(returns)
        current = comparison.iloc[-1].to_dict()
        output["volatility_comparison"] = {k: float(v) for k, v in current.items()}
    else:
        if args.method == "ewma":
            vol = ewma_volatility(returns, span=args.span)
        elif args.method == "garch":
            vol = garch_volatility(returns)
        else:
            vol = historical_volatility(returns, window=args.span)

        output["current_volatility"] = float(vol.iloc[-1])
        output["method"] = args.method

    if args.forecast:
        forecast = volatility_forecast(returns, method=args.method if args.method != "compare" else "ewma")
        output["forecast"] = {
            "current": forecast.current,
            "forecast_1d": forecast.forecast_1d,
            "forecast_5d": forecast.forecast_5d,
            "forecast_21d": forecast.forecast_21d,
            "half_life": forecast.half_life,
        }

    if args.cone:
        cone = volatility_cone(returns)
        output["volatility_cone"] = {
            "windows": cone.windows,
            "current": cone.current,
            "min": cone.min_vol,
            "max": cone.max_vol,
            "median": cone.median_vol,
        }

    if args.json:
        print(json_module.dumps(output, indent=2))
    else:
        print("=== Volatility Analysis ===")
        print(f"Method: {args.method}")
        if "current_volatility" in output:
            print(f"Current Volatility: {output['current_volatility']:.2%}")
        if "volatility_comparison" in output:
            print("\nVolatility Comparison (current):")
            for method, vol in output["volatility_comparison"].items():
                print(f"  {method}: {vol:.2%}")
        if "forecast" in output:
            f = output["forecast"]
            print(f"\nVolatility Forecast:")
            print(f"  1-day:  {f['forecast_1d']:.2%}")
            print(f"  5-day:  {f['forecast_5d']:.2%}")
            print(f"  21-day: {f['forecast_21d']:.2%}")
            if f.get("half_life"):
                print(f"  Half-life: {f['half_life']:.1f} days")
        if "volatility_cone" in output:
            print("\nVolatility Cone:")
            c = output["volatility_cone"]
            for i, w in enumerate(c["windows"]):
                print(f"  {w:3d}d: Current={c['current'][i]:.2%} | Range=[{c['min'][i]:.2%}, {c['max'][i]:.2%}] | Median={c['median'][i]:.2%}")

    return 0


def cmd_distribution(args: argparse.Namespace) -> int:
    """Handle the distribution subcommand for returns distribution analysis.

    Analyzes return distribution statistics, normality, and tail risk.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.returns_distribution import (
        analyze_distribution,
        analyze_tails,
        distribution_summary,
        test_normality,
    )

    settings = Settings()
    features_path = args.features or str(Path(settings.data.gold_dir) / "features.parquet")

    if not Path(features_path).exists():
        print(f"Error: Features file not found: {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if args.asset and "asset" in df.columns:
        df = df.filter(pl.col("asset") == args.asset)

    if "returns" not in df.columns:
        print("Error: 'returns' column not found in features file")
        return 1

    returns = df["returns"].drop_nulls().to_pandas()

    output = {}

    # Basic distribution analysis
    stats = analyze_distribution(returns)
    output["distribution"] = {
        "mean": stats.mean,
        "std": stats.std,
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,
        "min": stats.min_return,
        "max": stats.max_return,
        "n_observations": stats.n_observations,
    }

    if args.normality:
        normality = test_normality(returns)
        output["normality_tests"] = {
            "jarque_bera": {"statistic": normality.jarque_bera[0], "pvalue": normality.jarque_bera[1]},
            "kolmogorov_smirnov": {"statistic": normality.kolmogorov_smirnov[0], "pvalue": normality.kolmogorov_smirnov[1]},
            "is_normal_5pct": normality.is_normal_5pct,
            "is_normal_1pct": normality.is_normal_1pct,
        }
        if normality.shapiro_wilk:
            output["normality_tests"]["shapiro_wilk"] = {"statistic": normality.shapiro_wilk[0], "pvalue": normality.shapiro_wilk[1]}

    if args.tails:
        tails = analyze_tails(returns)
        output["tail_analysis"] = {
            "var_95": tails.var_95,
            "var_99": tails.var_99,
            "expected_shortfall_95": tails.expected_shortfall_95,
            "expected_shortfall_99": tails.expected_shortfall_99,
            "left_tail_ratio": tails.left_tail_ratio,
            "right_tail_ratio": tails.right_tail_ratio,
            "gain_loss_ratio": tails.gain_loss_ratio,
        }

    if args.json:
        print(json_module.dumps(output, indent=2))
    else:
        print("=== Returns Distribution Analysis ===")
        d = output["distribution"]
        print(f"Observations: {d['n_observations']}")
        print(f"Mean Return:  {d['mean']:.6f}")
        print(f"Std Dev:      {d['std']:.6f}")
        print(f"Skewness:     {d['skewness']:.4f}")
        print(f"Kurtosis:     {d['kurtosis']:.4f} (excess)")
        print(f"Range:        [{d['min']:.4f}, {d['max']:.4f}]")

        if "normality_tests" in output:
            print("\n--- Normality Tests ---")
            n = output["normality_tests"]
            print(f"Jarque-Bera:      stat={n['jarque_bera']['statistic']:.2f}, p={n['jarque_bera']['pvalue']:.4f}")
            print(f"K-S Test:         stat={n['kolmogorov_smirnov']['statistic']:.4f}, p={n['kolmogorov_smirnov']['pvalue']:.4f}")
            print(f"Normal at 5%:     {'Yes' if n['is_normal_5pct'] else 'No'}")

        if "tail_analysis" in output:
            print("\n--- Tail Risk Analysis ---")
            t = output["tail_analysis"]
            print(f"VaR 95%:          {t['var_95']:.4f}")
            print(f"VaR 99%:          {t['var_99']:.4f}")
            print(f"ES 95% (CVaR):    {t['expected_shortfall_95']:.4f}")
            print(f"ES 99% (CVaR):    {t['expected_shortfall_99']:.4f}")
            print(f"Gain/Loss Ratio:  {t['gain_loss_ratio']:.2f}")

    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    """Handle the signal subcommand for signal backtesting.

    Backtests trading signals with performance metrics and optional
    decay analysis.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import json as json_module

    import polars as pl

    from imst_quant.config.settings import Settings
    from imst_quant.utils.signal_backtest import (
        backtest_signal,
        bootstrap_signal,
        signal_decay_analysis,
        signal_statistics,
        turnover_analysis,
    )

    settings = Settings()
    features_path = args.features or str(Path(settings.data.gold_dir) / "features.parquet")

    if not Path(features_path).exists():
        print(f"Error: Features file not found: {features_path}")
        return 1

    df = pl.read_parquet(features_path)

    if args.signal_col not in df.columns:
        print(f"Error: Signal column '{args.signal_col}' not found")
        return 1

    if args.returns_col not in df.columns:
        print(f"Error: Returns column '{args.returns_col}' not found")
        return 1

    signal = df[args.signal_col].drop_nulls().to_pandas()
    returns = df[args.returns_col].drop_nulls().to_pandas()

    # Align series
    common_idx = signal.index.intersection(returns.index)
    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]

    output = {}

    # Main backtest
    result = backtest_signal(signal, returns, transaction_cost=args.transaction_cost)
    output["backtest"] = {
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "hit_rate": result.hit_rate,
        "profit_factor": result.profit_factor,
        "n_trades": result.n_trades,
        "avg_holding_period": result.avg_holding_period,
    }

    # Signal statistics
    stats = signal_statistics(signal)
    output["signal_stats"] = stats

    # Turnover analysis
    turnover = turnover_analysis(signal)
    output["turnover"] = turnover

    if args.decay:
        decay = signal_decay_analysis(signal, returns, max_lag=args.max_lag)
        output["decay_analysis"] = decay.to_dict(orient="records")

    if args.bootstrap > 0:
        bootstrap = bootstrap_signal(signal, returns, n_simulations=args.bootstrap)
        output["bootstrap"] = bootstrap

    if args.json:
        print(json_module.dumps(output, indent=2, default=float))
    else:
        print("=== Signal Backtest Results ===")
        b = output["backtest"]
        print(f"Total Return:       {b['total_return']:.2%}")
        print(f"Annualized Return:  {b['annualized_return']:.2%}")
        print(f"Volatility:         {b['volatility']:.2%}")
        print(f"Sharpe Ratio:       {b['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {b['max_drawdown']:.2%}")
        print(f"Hit Rate:           {b['hit_rate']:.2%}")
        print(f"Profit Factor:      {b['profit_factor']:.2f}")
        print(f"Number of Trades:   {b['n_trades']}")
        print(f"Avg Holding Period: {b['avg_holding_period']:.1f} days")

        print("\n--- Signal Statistics ---")
        s = output["signal_stats"]
        print(f"Long %:  {s['pct_long']:.1%}")
        print(f"Short %: {s['pct_short']:.1%}")
        print(f"Flat %:  {s['pct_flat']:.1%}")

        if "decay_analysis" in output:
            print("\n--- Signal Decay (IC by lag) ---")
            for row in output["decay_analysis"][:10]:
                ic = row.get("information_coefficient", 0) or 0
                print(f"  Lag {row['lag']:2d}: IC={ic:.4f}")

        if "bootstrap" in output:
            print("\n--- Bootstrap Confidence Intervals ---")
            bs = output["bootstrap"]
            print(f"Sharpe 95% CI: [{bs['sharpe_ci_lower']:.2f}, {bs['sharpe_ci_upper']:.2f}]")
            print(f"Hit Rate 95% CI: [{bs['hit_rate_ci_lower']:.2%}, {bs['hit_rate_ci_upper']:.2%}]")

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check data pipeline health status.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.config.settings import Settings
    from imst_quant.utils.health_check import (
        check_pipeline_health,
        check_data_freshness,
    )
    import json as json_module

    settings = Settings()
    data_dir = Path(settings.data.raw_dir).parent  # Get data/ directory

    if args.layer:
        # Check specific layer freshness
        freshness = check_data_freshness(
            data_dir,
            layer=args.layer,
            max_age_hours=args.max_age_hours
        )

        if args.json:
            print(json_module.dumps(freshness, indent=2))
        else:
            print(f"=== {args.layer.upper()} Layer Freshness ===")
            if freshness["fresh"]:
                print("✓ Data is fresh")
            else:
                print(f"✗ Data is stale: {freshness.get('reason', 'unknown')}")

            if freshness.get("latest_file"):
                print(f"Latest file: {freshness['latest_file']}")
                print(f"Age: {freshness['age_hours']:.1f} hours")
    else:
        # Check full pipeline health
        health = check_pipeline_health(data_dir, lookback_days=args.lookback_days)

        if args.json:
            print(json_module.dumps(health, indent=2))
        else:
            status_icon = "✓" if health["status"] == "healthy" else "⚠"
            print(f"=== Pipeline Health: {status_icon} {health['status'].upper()} ===")
            print(f"Checked at: {health['checked_at']}")
            print(f"\nData Layers (last {args.lookback_days} days):")

            for layer, info in health["layers"].items():
                if info["exists"]:
                    recent = "✓" if info["has_recent_data"] else "✗"
                    print(f"  {recent} {layer:12s}: {info['file_count']} files ({info['total_size_mb']} MB)")
                else:
                    print(f"  ✗ {layer:12s}: NOT FOUND")

            if health["alerts"]:
                print("\n⚠ Alerts:")
                for alert in health["alerts"]:
                    print(f"  - {alert}")

            if health["recommendations"]:
                print("\n💡 Recommendations:")
                for rec in health["recommendations"]:
                    print(f"  - {rec}")

    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Show quick portfolio metrics summary.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.config.settings import Settings
    import pandas as pd
    import json as json_module

    settings = Settings()

    if args.features:
        features_path = Path(args.features)
    else:
        features_path = Path(settings.data.gold_dir) / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        return 1

    df = pd.read_parquet(features_path)

    # Calculate summary metrics
    summary = {
        "total_records": len(df),
        "date_range": {
            "start": str(df.index.min()) if hasattr(df.index, 'min') else "N/A",
            "end": str(df.index.max()) if hasattr(df.index, 'max') else "N/A",
        },
    }

    # Add returns statistics if available
    if "returns" in df.columns:
        returns = df["returns"].dropna()
        summary["returns_stats"] = {
            "mean_daily": float(returns.mean()),
            "std_daily": float(returns.std()),
            "sharpe_approx": float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0,
            "min": float(returns.min()),
            "max": float(returns.max()),
        }

    # Add ticker-level stats if available
    if "ticker" in df.columns:
        ticker_counts = df["ticker"].value_counts().head(args.top_n)
        summary["top_tickers"] = ticker_counts.to_dict()

    if args.json:
        print(json_module.dumps(summary, indent=2, default=str))
    else:
        print("=== Portfolio Summary ===")
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")

        if "returns_stats" in summary:
            rs = summary["returns_stats"]
            print("\n--- Returns Statistics ---")
            print(f"Mean Daily Return: {rs['mean_daily']:.4%}")
            print(f"Daily Volatility:  {rs['std_daily']:.4%}")
            print(f"Sharpe Ratio:      {rs['sharpe_approx']:.2f}")
            print(f"Min Return:        {rs['min']:.4%}")
            print(f"Max Return:        {rs['max']:.4%}")

        if "top_tickers" in summary:
            print(f"\n--- Top {args.top_n} Tickers by Record Count ---")
            for ticker, count in summary["top_tickers"].items():
                print(f"  {ticker:6s}: {count:,} records")

    return 0


def cmd_exposure(args: argparse.Namespace) -> int:
    """Analyze portfolio concentration and exposure across dimensions.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.config.settings import Settings
    from imst_quant.utils.exposure_analysis import (
        ExposureAnalyzer,
        format_exposure_report,
    )
    import polars as pl
    import json as json_module

    settings = Settings()

    if args.portfolio:
        portfolio_path = Path(args.portfolio)
    else:
        # Default: try to load from gold directory
        portfolio_path = Path(settings.data.gold_dir) / "portfolio.parquet"

    if not portfolio_path.exists():
        print(f"Error: Portfolio file not found: {portfolio_path}")
        print("Expected columns: [symbol, weight, sector, asset_type, geography (optional)]")
        return 1

    try:
        portfolio_df = pl.read_parquet(portfolio_path)
    except Exception as e:
        print(f"Error reading portfolio file: {e}")
        return 1

    try:
        analyzer = ExposureAnalyzer(portfolio_df)
        metrics = analyzer.analyze(include_geography=args.geography)

        # Get concentration risks
        risks = analyzer.identify_concentration_risks(
            sector_threshold=args.sector_threshold,
            single_threshold=args.position_threshold,
        )

        # Get diversification score
        div_score = analyzer.get_diversification_score()

        if args.json:
            output = {
                "sector_exposure": metrics.by_sector,
                "asset_type_exposure": metrics.by_asset_type,
                "geography_exposure": metrics.by_geography,
                "herfindahl_index": metrics.herfindahl_index,
                "max_single_exposure": metrics.max_single_exposure,
                "top5_concentration": metrics.concentration_ratio_top5,
                "diversification_score": div_score,
                "concentration_risks": risks,
            }
            print(json_module.dumps(output, indent=2, default=str))
        else:
            # Print formatted report
            print(format_exposure_report(metrics))
            print()
            print(f"Diversification Score: {div_score:.1f}/100")

            if risks:
                print("\n⚠️  Concentration Risks Identified:")
                for risk in risks:
                    print(f"  • {risk}")
            else:
                print("\n✓ No concentration risks detected")

    except Exception as e:
        print(f"Error analyzing exposure: {e}")
        return 1

    return 0


def cmd_risk_metrics(args: argparse.Namespace) -> int:
    """Calculate comprehensive portfolio risk metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.config.settings import Settings
    from imst_quant.utils.risk_metrics import calculate_all_metrics
    import polars as pl
    import json as json_module

    settings = Settings()

    if args.returns:
        returns_path = Path(args.returns)
    else:
        returns_path = Path(settings.data.gold_dir) / "returns.parquet"

    if not returns_path.exists():
        print(f"Error: Returns file not found: {returns_path}")
        print("Expected columns: [date, returns] or specify with --return-col")
        return 1

    try:
        df = pl.read_parquet(returns_path)
    except Exception as e:
        print(f"Error reading returns file: {e}")
        return 1

    if args.return_col not in df.columns:
        print(f"Error: Column '{args.return_col}' not found in returns file")
        print(f"Available columns: {df.columns}")
        return 1

    try:
        metrics = calculate_all_metrics(
            df,
            risk_free_rate=args.risk_free_rate,
            periods_per_year=args.periods,
            var_confidence=args.var_confidence,
            return_col=args.return_col,
        )

        if args.json:
            print(json_module.dumps(metrics, indent=2))
        else:
            print("\n=== Portfolio Risk Metrics ===\n")
            print(f"Total Return:        {metrics['total_return']:>10.2%}")
            print(f"Annualized Return:   {metrics['annualized_return']:>10.2%}")
            print(f"Annualized Vol:      {metrics['volatility']:>10.2%}")
            print()
            print(f"Sharpe Ratio:        {metrics['sharpe']:>10.3f}")
            print(f"Sortino Ratio:       {metrics['sortino']:>10.3f}")
            print(f"Calmar Ratio:        {metrics['calmar']:>10.3f}")
            print()
            print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
            print(f"VaR ({args.var_confidence:.0%}):        {metrics['var']:>10.2%}")
            print()

            # Risk assessment
            sharpe_rating = "Excellent" if metrics['sharpe'] > 2 else "Good" if metrics['sharpe'] > 1 else "Fair" if metrics['sharpe'] > 0 else "Poor"
            print(f"Risk Rating:         {sharpe_rating} (based on Sharpe ratio)")

    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return 1

    return 0


def cmd_quality_check(args: argparse.Namespace) -> int:
    """Validate data quality and completeness.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.config.settings import Settings
    from imst_quant.utils.data_quality import check_data_quality
    import polars as pl
    import json as json_module
    from datetime import datetime

    settings = Settings()

    # Determine directory based on layer
    layer_dirs = {
        "bronze": settings.data.bronze_dir,
        "silver": settings.data.silver_dir,
        "gold": settings.data.gold_dir,
    }

    data_dir = Path(layer_dirs[args.layer])

    if not data_dir.exists():
        print(f"Error: {args.layer} directory not found: {data_dir}")
        return 1

    # Get files to check
    if args.date:
        # Check specific date
        date_str = args.date.replace("-", "")
        pattern = f"*{date_str}*.parquet"
    else:
        # Check all files
        pattern = "*.parquet"

    files = list(data_dir.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return 1

    print(f"\n=== Data Quality Check: {args.layer.upper()} Layer ===\n")
    print(f"Checking {len(files)} file(s)...\n")

    issues_found = []

    for file in files:
        try:
            df = pl.read_parquet(file)
            result = check_data_quality(df)

            if not result["is_valid"]:
                issues_found.append({
                    "file": file.name,
                    "issues": result["issues"],
                })

            if args.json:
                continue

            # Print results
            status = "✓" if result["is_valid"] else "✗"
            print(f"{status} {file.name}")

            if result["issues"]:
                for issue in result["issues"]:
                    print(f"  ⚠️  {issue}")

        except Exception as e:
            print(f"✗ {file.name}: Error reading file - {e}")
            issues_found.append({
                "file": file.name,
                "issues": [f"Read error: {e}"],
            })

    if args.json:
        output = {
            "layer": args.layer,
            "files_checked": len(files),
            "issues_count": len(issues_found),
            "issues": issues_found,
        }
        print(json_module.dumps(output, indent=2))
    else:
        print()
        if issues_found:
            print(f"❌ Found {len(issues_found)} file(s) with issues")
            if args.fix:
                print("\n⚠️  Auto-fix not yet implemented. Please fix issues manually.")
        else:
            print("✅ All files passed quality checks")

    return 1 if issues_found else 0


def cmd_signal_performance(args: argparse.Namespace) -> int:
    """Analyze trading signal performance metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    from imst_quant.utils.signal_performance import analyze_signal_performance
    import polars as pl
    import json as json_module

    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Expected columns: [signal, returns] or specify with --signal-col and --returns-col")
        return 1

    try:
        df = pl.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return 1

    if args.signal_col not in df.columns or args.returns_col not in df.columns:
        print(f"Error: Required columns not found")
        print(f"Expected: [{args.signal_col}, {args.returns_col}]")
        print(f"Available: {df.columns}")
        return 1

    try:
        metrics = analyze_signal_performance(
            df,
            signal_col=args.signal_col,
            returns_col=args.returns_col,
        )

        if args.json:
            print(json_module.dumps(metrics, indent=2))
        else:
            print("\n=== Signal Performance Analysis ===\n")
            print(f"Total Trades:        {metrics['total_trades']:>10,}")
            print()
            print(f"Win Rate:            {metrics['win_rate']:>10.2%}")
            print(f"Average Win:         {metrics['avg_win']:>10.4f}")
            print(f"Average Loss:        {metrics['avg_loss']:>10.4f}")
            print(f"Win/Loss Ratio:      {metrics['win_loss_ratio']:>10.3f}")
            print()
            print(f"Profit Factor:       {metrics['profit_factor']:>10.3f}")
            print(f"Total PnL:           {metrics['total_pnl']:>10.4f}")
            print(f"Average Trade PnL:   {metrics['avg_trade_pnl']:>10.4f}")
            print()
            print(f"Max Win Streak:      {metrics['max_win_streak']:>10,}")
            print(f"Max Loss Streak:     {metrics['max_loss_streak']:>10,}")
            print()
            print(f"Quality Score:       {metrics['quality_score']:>10.1f}/100")
            print()

            # Performance assessment
            if metrics['quality_score'] >= 70:
                rating = "Excellent"
            elif metrics['quality_score'] >= 50:
                rating = "Good"
            elif metrics['quality_score'] >= 30:
                rating = "Fair"
            else:
                rating = "Poor"

            print(f"Overall Rating:      {rating}")

    except Exception as e:
        print(f"Error analyzing signal performance: {e}")
        return 1

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
        "factors": cmd_factors,
        "execution": cmd_execution,
        "streaks": cmd_streaks,
        "volatility": cmd_volatility,
        "distribution": cmd_distribution,
        "signal": cmd_signal,
        "health": cmd_health,
        "summary": cmd_summary,
        "exposure": cmd_exposure,
        "risk-metrics": cmd_risk_metrics,
        "quality-check": cmd_quality_check,
        "signal-performance": cmd_signal_performance,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
