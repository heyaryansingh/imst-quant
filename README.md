# IMST-Quant

**Influence-aware Multi-Source Sentiment Trading**

A Graph Neural Network (GNN) based sentiment analysis pipeline for quantitative trading. IMST-Quant ingests social media data (Reddit), processes it through a medallion architecture (Raw -> Bronze -> Silver -> Gold), applies NLP sentiment analysis, and uses GNN influence scoring to generate trading signals.

---

## Architecture

```
                              IMST-Quant Pipeline Architecture
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  DATA INGESTION                    DATA PROCESSING                                |
|  +-------------+                   +---------------+                              |
|  |   Reddit    |  Raw JSON         |    Bronze     |  Parquet                     |
|  |   (PRAW)    | ------------->    | (Dedup, Lang) | -----------+                 |
|  +-------------+                   +---------------+            |                 |
|  +-------------+                          |                     |                 |
|  |   Market    |  Raw JSON                v                     |                 |
|  | (yfinance)  | ------------->    +---------------+            |                 |
|  +-------------+                   |    Silver     |            |                 |
|  +-------------+                   | (Entity Link) |            |                 |
|  |   Crypto    |  Raw JSON         +---------------+            |                 |
|  |   (CCXT)    | -------+                |                      |                 |
|  +-------------+        |                v                      |                 |
|                         |         +---------------+             |                 |
|                         +-------> |   Sentiment   | <-----------+                 |
|                                   |   Analysis    |                               |
|                                   +---------------+                               |
|                                          |                                        |
+------------------------------------------|-----------------------------------------+
                                           |
+------------------------------------------|-----------------------------------------+
|                                          v                                        |
|  INFLUENCE & FEATURES              +---------------+                              |
|                                    |    GNN        |                              |
|                                    |  Influence    |                              |
|                                    |   Scoring     |                              |
|                                    +---------------+                              |
|                                          |                                        |
|                                          v                                        |
|                                   +---------------+                               |
|                                   |     Gold      |                               |
|                                   |   Features    |                               |
|                                   +---------------+                               |
|                                          |                                        |
+------------------------------------------|-----------------------------------------+
                                           |
+------------------------------------------|-----------------------------------------+
|                                          v                                        |
|  FORECASTING & TRADING             +---------------+                              |
|                                    |   ML Models   |                              |
|                                    | LSTM/CNN/TF   |                              |
|                                    +---------------+                              |
|                                          |                                        |
|                                          v                                        |
|                                   +---------------+     +---------------+         |
|                                   |    Signals    | --> |   Backtest    |         |
|                                   |   Generator   |     |    Engine     |         |
|                                   +---------------+     +---------------+         |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Medallion Architecture

| Layer | Purpose | Storage |
|-------|---------|---------|
| **Raw** | Unprocessed JSON from APIs | `data/raw/` |
| **Bronze** | Deduplicated, validated Parquet | `data/bronze/` |
| **Silver** | Entity-linked, cleaned data | `data/silver/` |
| **Sentiment** | Sentiment scores and aggregates | `data/sentiment/` |
| **Influence** | GNN influence scores per author | `data/influence/` |
| **Gold** | Feature vectors for ML models | `data/gold/` |

---

## Features

- **Multi-Source Ingestion**: Reddit (via PRAW), equities (yfinance), crypto (CCXT/Binance)
- **Checkpoint-Based Crawling**: Resume interrupted ingestion without duplicates
- **Entity Linking**: Cashtag extraction, company alias resolution, ambiguity handling
- **Sentiment Analysis**: TextBlob + FinBERT support
- **GNN Influence Scoring**: 2-layer GCN to identify influential authors
- **ML Forecasters**: LSTM, CNN, Transformer models for price direction
- **Backtesting**: Event-driven backtester with transaction costs
- **Strict Causality**: Features use T-1 data to prevent lookahead bias

---

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/imst-quant.git
cd imst-quant

# Install with pip
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Reddit API credentials (required for Reddit ingestion)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Optional: Override default data paths
# DATA_RAW_DIR=data/raw
# DATA_BRONZE_DIR=data/bronze
```

To get Reddit API credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Create a "script" application
3. Copy the client ID (under app name) and secret

---

## Quick Start

### Using the CLI

IMST-Quant provides a unified CLI for all operations:

```bash
# Show all available commands
imst --help

# Ingest data from all sources
imst ingest --reddit --market

# Process through the pipeline
imst process --all

# Run sentiment analysis
imst analyze --sentiment

# Train a forecasting model and backtest
imst backtest --model lstm --epochs 20
```

### Using Make

```bash
# Install dependencies
make install

# Run the full pipeline
make reproduce

# Run tests
make test

# Clean all data
make clean
```

### Using Individual Scripts

```bash
# 1. Ingest Reddit posts
python scripts/ingest_reddit.py --limit 1000

# 2. Ingest market data (90 days by default)
python scripts/ingest_market.py

# 3. Convert raw to bronze
python scripts/raw_to_bronze.py

# 4. Convert bronze to silver (entity linking)
python scripts/bronze_to_silver.py

# 5. Generate sentiment aggregates
python scripts/silver_to_sentiment.py

# 6. Build feature vectors
python scripts/build_features.py

# 7. Train GNN influence model (optional)
python scripts/run_influence.py --year 2024 --month 1
```

---

## Usage Examples

### End-to-End Pipeline

```python
from pathlib import Path
from imst_quant.config.settings import Settings
from imst_quant.ingestion.market import ingest_equity_ohlcv
from imst_quant.features import build_daily_features
from imst_quant.models.train import train_forecaster
from imst_quant.trading.backtest import run_backtest

# Initialize settings
settings = Settings()

# Ingest market data
df_market = ingest_equity_ohlcv(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-06-01",
    output_dir=settings.data.raw_dir,
)

# Build features
features = build_daily_features(
    bronze_dir=settings.data.bronze_dir,
    sentiment_path=settings.data.sentiment_dir / "sentiment_aggregates.parquet",
    output_path=settings.data.gold_dir / "features.parquet",
)

# Train model
model = train_forecaster(
    features_path=settings.data.gold_dir / "features.parquet",
    model_type="lstm",
    epochs=20,
)

# Run backtest
results = run_backtest(
    features_path=settings.data.gold_dir / "features.parquet",
    transaction_cost=0.001,
)
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
```

### GNN Influence Scoring

```python
from imst_quant.influence.pipeline import run_influence_month

# Build influence graph and train GCN for a month
scores = run_influence_month(
    silver_dir="data/silver",
    influence_dir="data/influence",
    year=2024,
    month=1,
    min_interactions=50,
)
```

### Custom Sentiment Analysis

```python
from imst_quant.sentiment.textblob import analyze_sentiment
from imst_quant.sentiment.aggregation import aggregate_daily

# Analyze individual post
result = analyze_sentiment("AAPL is going to moon! Great earnings report.")
print(f"Polarity: {result['polarity']}, Subjectivity: {result['subjectivity']}")

# Aggregate daily sentiment for an asset
daily = aggregate_daily(df_silver, asset_id="AAPL")
```

---

## Project Structure

```
imst-quant/
|-- config/                  # Configuration files
|   |-- subreddits.yaml      # Subreddit list for ingestion
|
|-- data/                    # Data storage (gitignored)
|   |-- raw/                 # Raw JSON from APIs
|   |-- bronze/              # Deduplicated Parquet
|   |-- silver/              # Entity-linked data
|   |-- sentiment/           # Sentiment aggregates
|   |-- influence/           # GNN influence scores
|   |-- gold/                # Feature vectors
|
|-- scripts/                 # Standalone pipeline scripts
|   |-- ingest_reddit.py
|   |-- ingest_market.py
|   |-- raw_to_bronze.py
|   |-- bronze_to_silver.py
|   |-- silver_to_sentiment.py
|   |-- run_influence.py
|   |-- build_features.py
|
|-- src/imst_quant/          # Main package
|   |-- config/              # Settings (Pydantic)
|   |-- ingestion/           # Data ingestion modules
|   |-- storage/             # Data layer (raw, bronze, silver)
|   |-- processing/          # Deduplication, language, normalization
|   |-- entities/            # Cashtag extraction, entity linking
|   |-- sentiment/           # TextBlob, FinBERT, aggregation
|   |-- influence/           # GNN models, graph construction
|   |-- features/            # Feature engineering
|   |-- models/              # LSTM, CNN, Transformer forecasters
|   |-- trading/             # Signals, backtest, portfolio
|   |-- cli.py               # CLI entry point
|
|-- tests/                   # Unit and integration tests
|-- examples/                # Example scripts
|-- pyproject.toml           # Project configuration
|-- Makefile                 # Build automation
|-- README.md                # This file
```

---

## Configuration

### Settings Hierarchy

Configuration is loaded from environment variables and `.env` file:

| Setting | Env Variable | Default |
|---------|--------------|---------|
| Reddit Client ID | `REDDIT_CLIENT_ID` | (required) |
| Reddit Secret | `REDDIT_CLIENT_SECRET` | (required) |
| Raw Data Dir | `DATA_RAW_DIR` | `data/raw` |
| Bronze Data Dir | `DATA_BRONZE_DIR` | `data/bronze` |
| Silver Data Dir | `DATA_SILVER_DIR` | `data/silver` |
| Equity Tickers | `MARKET_EQUITY_TICKERS` | `AAPL,JNJ,JPM,XOM` |
| Crypto Pairs | `MARKET_CRYPTO_PAIRS` | `BTC/USDT,ETH/USDT` |

### Subreddit Configuration

Edit `config/subreddits.yaml`:

```yaml
equity_subreddits:
  - wallstreetbets
  - stocks
  - investing

crypto_subreddits:
  - Bitcoin
  - CryptoCurrency
  - ethereum
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/imst_quant --cov-report=html

# Run specific test categories
pytest tests/test_sentiment.py -v
pytest tests/test_influence.py -v
```

---

## Development

### Code Style

This project uses:
- **Black** for formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Adding New Data Sources

1. Create ingestion module in `src/imst_quant/ingestion/`
2. Add storage handlers in `src/imst_quant/storage/`
3. Update entity linking if new asset types
4. Add tests in `tests/`

---

## Roadmap

- [x] Reddit ingestion with checkpointing
- [x] Market data (yfinance + CCXT)
- [x] Medallion architecture (Raw -> Bronze -> Silver)
- [x] Sentiment analysis (TextBlob + FinBERT)
- [x] GNN influence scoring
- [x] Feature engineering with strict causality
- [x] ML forecasters (LSTM, CNN, Transformer)
- [x] Backtesting engine
- [ ] Paper trading integration
- [ ] Real-time monitoring dashboard
- [ ] Advanced portfolio optimization
- [ ] Multi-asset correlation analysis

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- [PRAW](https://praw.readthedocs.io/) for Reddit API access
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN implementation
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [CCXT](https://github.com/ccxt/ccxt) for crypto exchange integration
