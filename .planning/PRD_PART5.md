# PRD Part 5: Architecture and Implementation Blueprint

## 7. Architecture

### 7.1 System Architecture (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMST-QUANT SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA INGESTION LAYER                           │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   Reddit API    │  StockTwits API │    RSS Feeds    │    Market Data APIs   │
│   (PRAW)        │  (Connector)    │   (feedparser)  │  (Polygon/yfinance)   │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬────────────┘
         │                 │                 │                   │
         ▼                 ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAW DATA STORE                                 │
│                     /data/raw/{source}/YYYY/MM/DD/*.jsonl.gz                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING PIPELINE                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   Deduplication │ Language Filter │  Entity Linker  │   Text Normalizer     │
│   (MinHash)     │  (fasttext)     │ (Embeddings)    │                       │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬────────────┘
         │                 │                 │                   │
         ▼                 ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BRONZE/SILVER STORE                              │
│                   /data/bronze/... (parsed) /data/silver/... (enriched)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│   SENTIMENT MODULE    │ │   INFLUENCE MODULE    │ │   CREDIBILITY MODULE  │
├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤
│ • TextBlob (baseline) │ │ • Graph Builder       │ │ • Author Profiler     │
│ • FinBERT (upgrade)   │ │ • GCN Training        │ │ • Bot Detector        │
│ • Stance Classifier   │ │ • Score Inference     │ │ • Brigade Detector    │
│ • Event Tagger        │ │                       │ │                       │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FEATURE STORE (GOLD)                             │
│                          /data/gold/daily_features.parquet                  │
│                          (AS-OF CORRECT, CAUSALITY ENFORCED)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FORECASTING MODULE                                │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ LSTM Classifier │ CNN Classifier  │   Transformer   │  LightGBM/TFT         │
│ (baseline)      │ (baseline)      │   (baseline)    │  (upgrade)            │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬────────────┘
         │                 │                 │                   │
         └─────────────────┴─────────────────┴───────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRADING MODULE                                   │
├─────────────────────────────────────┬───────────────────────────────────────┤
│         BASELINE POLICIES           │          UPGRADE POLICIES             │
│  • Fixed Threshold (LEAKAGE DEMO)   │  • Portfolio Policy                   │
│  • Dynamic Threshold                │  • Risk Management                    │
│                                     │  • Transaction Cost Model             │
└─────────────────────────────────────┴───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│         BACKTEST ENGINE           │   │       PAPER TRADING ENGINE        │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│ • Walk-forward Validation         │   │ • Daily Scheduler                 │
│ • Purging/Embargo                 │   │ • Alpaca Integration              │
│ • Transaction Cost Model          │   │ • Order Management                │
│ • Statistical Tests               │   │ • P&L Tracking                    │
│ • Ablation Framework              │   │ • Kill Switches                   │
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING & OBSERVABILITY                          │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   Prometheus    │    Grafana      │   Alertmanager  │   OpenTelemetry       │
│   (metrics)     │  (dashboards)   │   (alerts)      │   (traces/logs)       │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                  │
├───────────────────────────────────────┬─────────────────────────────────────┤
│           PostgreSQL                  │           Object Store              │
│  • Metadata, configs, logs            │  • Raw data (JSONL.gz)              │
│  • Trading logs, audit trail          │  • Parquet files (bronze/silver/gold│
│  • User tables                        │  • Model artifacts                  │
└───────────────────────────────────────┴─────────────────────────────────────┘
```

### 7.2 Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCY GRAPH                           │
└─────────────────────────────────────────────────────────────────────────────┘

                                   ┌──────────┐
                                   │  config  │
                                   └────┬─────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
        ┌──────────┐             ┌──────────┐             ┌──────────┐
        │ ingestion│             │processing│             │   nlp    │
        └────┬─────┘             └────┬─────┘             └────┬─────┘
             │                        │                        │
             │                        ▼                        │
             │                  ┌──────────┐                   │
             │                  │  graph   │◄──────────────────┤
             │                  └────┬─────┘                   │
             │                       │                         │
             └───────────────────────┼─────────────────────────┘
                                     ▼
                               ┌──────────┐
                               │ features │
                               └────┬─────┘
                                    │
                                    ▼
                               ┌──────────┐
                               │  models  │
                               └────┬─────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              ┌──────────┐                    ┌──────────┐
              │ trading  │                    │ backtest │
              └────┬─────┘                    └────┬─────┘
                   │                               │
                   ▼                               │
              ┌────────────┐                       │
              │paper_trade │                       │
              └────┬───────┘                       │
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                             ┌──────────┐
                             │monitoring│
                             └──────────┘
```

### 7.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                              │
└─────────────────────────────────────────────────────────────────────────────┘

[Reddit API] ──► [Raw JSONL] ──► [Parse/Validate] ──► [Bronze Parquet]
                     │                                       │
                     │                                       ▼
                     │              [Entity Linking] ◄── [Silver Parquet]
                     │                    │                  │
                     │                    ▼                  │
                     │              [Sentiment]              │
                     │                    │                  │
                     │                    ▼                  │
                     │              [Aggregation]            │
                     │                    │                  │
                     │                    ▼                  │
[Market API] ──► [Raw JSON] ──► [OHLCV Parquet] ──► [Gold Features]
                                                          │
                                                          ▼
                                                    [Model Training]
                                                          │
                                                          ▼
                                                    [Predictions DB]
                                                          │
                                    ┌─────────────────────┴──────────────────┐
                                    ▼                                        ▼
                              [Backtest]                              [Paper Trade]
                                    │                                        │
                                    ▼                                        ▼
                              [Results DB]                            [Trading Log DB]
```

---

## 8. Implementation Blueprint

### 8.1 Repository Structure

```
imst-quant/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                      # Lint, test, build
│   │   ├── backtest.yml                # Weekly backtest run
│   │   └── deploy.yml                  # Deployment pipeline
│   └── CODEOWNERS
│
├── configs/
│   ├── default.yaml                    # Default configuration
│   ├── paper_replication.yaml          # Paper replication settings
│   ├── production.yaml                 # Production settings
│   ├── backtest.yaml                   # Backtest configuration
│   └── schemas/
│       ├── config_schema.py            # Pydantic config models
│       └── validate.py                 # Config validation
│
├── data/                               # Gitignored, created at runtime
│   ├── raw/
│   │   ├── reddit/
│   │   ├── market/
│   │   └── crypto/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── docker/
│   ├── Dockerfile                      # Main application
│   ├── Dockerfile.jupyter              # Jupyter for research
│   ├── docker-compose.yml              # Local development
│   └── docker-compose.prod.yml         # Production setup
│
├── docs/
│   ├── architecture.md
│   ├── model_cards/
│   │   ├── lstm_classifier.md
│   │   ├── cnn_classifier.md
│   │   ├── transformer_classifier.md
│   │   └── influence_gnn.md
│   ├── datasheets/
│   │   ├── reddit_posts.md
│   │   └── market_data.md
│   ├── runbooks/
│   │   ├── daily_operations.md
│   │   ├── incident_response.md
│   │   └── model_retraining.md
│   └── api.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_influence_modeling.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_backtest_analysis.ipynb
│   └── 06_ablation_studies.ipynb
│
├── reports/                            # Generated reports
│   ├── daily/
│   ├── weekly/
│   └── monthly/
│
├── scripts/
│   ├── setup_dev.sh                    # Development setup
│   ├── download_models.sh              # Download pretrained models
│   ├── run_backtest.sh                 # Run backtest script
│   └── deploy.sh                       # Deployment script
│
├── src/
│   └── imst_quant/
│       ├── __init__.py
│       ├── __main__.py                 # CLI entry point
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py                 # Typer CLI app
│       │   ├── ingest.py               # Ingestion commands
│       │   ├── train.py                # Training commands
│       │   ├── backtest.py             # Backtest commands
│       │   ├── paper_trade.py          # Paper trading commands
│       │   └── evaluate.py             # Evaluation commands
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py             # Pydantic settings
│       │   └── loader.py               # Config loading
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── base.py                 # Base connector interface
│       │   ├── reddit.py               # Reddit connector
│       │   ├── stocktwits.py           # StockTwits connector
│       │   ├── market.py               # Market data connector
│       │   ├── crypto.py               # Crypto data connector
│       │   └── rss.py                  # RSS connector
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── cleaning.py             # Text cleaning
│       │   ├── deduplication.py        # Near-duplicate detection
│       │   ├── language.py             # Language detection
│       │   └── entity_linking.py       # Entity linking
│       │
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── sentiment/
│       │   │   ├── __init__.py
│       │   │   ├── textblob.py         # TextBlob baseline
│       │   │   ├── finbert.py          # FinBERT sentiment
│       │   │   └── aggregation.py      # Sentiment aggregation
│       │   ├── stance/
│       │   │   ├── __init__.py
│       │   │   └── classifier.py       # Stance classification
│       │   └── events/
│       │       ├── __init__.py
│       │       └── tagger.py           # Event tagging
│       │
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── builder.py              # Graph construction
│       │   ├── models.py               # GNN models
│       │   ├── trainer.py              # GNN training
│       │   └── inference.py            # Influence scoring
│       │
│       ├── credibility/
│       │   ├── __init__.py
│       │   ├── profiler.py             # Author profiling
│       │   ├── bot_detection.py        # Bot detection
│       │   └── brigade_detection.py    # Brigade detection
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── builder.py              # Feature construction
│       │   ├── store.py                # Feature store interface
│       │   ├── causality.py            # Causality enforcement
│       │   └── validation.py           # Feature validation
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                 # Base model interface
│       │   ├── lstm.py                 # LSTM classifier
│       │   ├── cnn.py                  # CNN classifier
│       │   ├── transformer.py          # Transformer classifier
│       │   ├── lightgbm.py             # LightGBM model
│       │   ├── trainer.py              # Training pipeline
│       │   └── ensemble.py             # Model ensemble
│       │
│       ├── trading/
│       │   ├── __init__.py
│       │   ├── policies/
│       │   │   ├── __init__.py
│       │   │   ├── fixed_threshold.py  # Fixed threshold (leakage demo)
│       │   │   ├── dynamic_threshold.py# Dynamic threshold
│       │   │   └── portfolio.py        # Portfolio policy
│       │   ├── risk.py                 # Risk management
│       │   └── costs.py                # Transaction costs
│       │
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py               # Backtest engine
│       │   ├── metrics.py              # Performance metrics
│       │   ├── walk_forward.py         # Walk-forward validation
│       │   ├── statistics.py           # Statistical tests
│       │   └── ablations.py            # Ablation framework
│       │
│       ├── paper_trade/
│       │   ├── __init__.py
│       │   ├── scheduler.py            # Trading scheduler
│       │   ├── brokers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py             # Base broker interface
│       │   │   └── alpaca.py           # Alpaca integration
│       │   ├── execution.py            # Order execution
│       │   └── logging.py              # Trade logging
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── metrics.py              # Prometheus metrics
│       │   ├── drift.py                # Drift detection
│       │   └── alerts.py               # Alert rules
│       │
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── daily.py                # Daily reports
│       │   ├── backtest.py             # Backtest reports
│       │   └── templates/              # Report templates
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py              # Logging setup
│           ├── database.py             # Database utilities
│           └── time.py                 # Time utilities
│
├── tests/
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/
│   │   ├── test_sentiment.py
│   │   ├── test_entity_linking.py
│   │   ├── test_features.py
│   │   ├── test_causality.py           # Causality tests
│   │   ├── test_models.py
│   │   └── test_trading.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_feature_pipeline.py
│   │   └── test_backtest_pipeline.py
│   └── e2e/
│       ├── test_full_pipeline.py
│       └── test_paper_trading.py
│
├── .env.example                        # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml                      # Poetry configuration
├── poetry.lock
└── README.md
```

### 8.2 Configuration Files

#### pyproject.toml
```toml
[tool.poetry]
name = "imst-quant"
version = "0.1.0"
description = "Influence-aware Multi-Source Sentiment Trading"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "imst_quant", from = "src"}]

[tool.poetry.dependencies]
python = "^3.11"

# Core
polars = "^0.20.0"
pandas = "^2.1.0"
numpy = "^1.26.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

# NLP
transformers = "^4.36.0"
tokenizers = "^0.15.0"
sentencepiece = "^0.1.99"
textblob = "^0.17.1"
fasttext-wheel = "^0.9.2"
sentence-transformers = "^2.2.2"

# Graph
torch = "^2.1.0"
torch-geometric = "^2.4.0"
networkx = "^3.2.0"

# ML
pytorch-lightning = "^2.1.0"
scikit-learn = "^1.3.0"
lightgbm = "^4.2.0"
mlflow = "^2.9.0"
optuna = "^3.4.0"

# Data
praw = "^7.7.1"
ccxt = "^4.1.0"
yfinance = "^0.2.33"
feedparser = "^6.0.10"
datasketch = "^1.6.4"

# API
fastapi = "^0.108.0"
uvicorn = "^0.25.0"
httpx = "^0.26.0"

# Database
sqlalchemy = "^2.0.0"
asyncpg = "^0.29.0"
psycopg2-binary = "^2.9.9"

# Orchestration
prefect = "^2.14.0"

# Monitoring
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.21.0"
opentelemetry-sdk = "^1.21.0"

# Broker
alpaca-trade-api = "^3.0.2"

# Utils
typer = "^0.9.0"
rich = "^13.7.0"
python-dotenv = "^1.0.0"
pytz = "^2023.3"
emoji = "^2.9.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^4.1.0"
pytest-mock = "^3.12.0"
black = "^23.12.0"
ruff = "^0.1.8"
mypy = "^1.7.0"
pre-commit = "^3.6.0"
ipython = "^8.19.0"
jupyter = "^1.0.0"
great-expectations = "^0.18.0"

[tool.poetry.scripts]
imst = "imst_quant.cli.main:app"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "D", "UP"]
ignore = ["D100", "D104"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --cov=src/imst_quant --cov-report=term-missing"
```

#### configs/default.yaml
```yaml
# IMST-Quant Default Configuration

project:
  name: "imst-quant"
  version: "0.1.0"
  mode: "paper_replication"  # paper_replication | production

# Data paths
paths:
  data_dir: "./data"
  raw_dir: "${paths.data_dir}/raw"
  bronze_dir: "${paths.data_dir}/bronze"
  silver_dir: "${paths.data_dir}/silver"
  gold_dir: "${paths.data_dir}/gold"
  models_dir: "./models"
  reports_dir: "./reports"

# Ingestion settings
ingestion:
  reddit:
    enabled: true
    subreddits:
      equity:
        - wallstreetbets
        - stocks
        - investing
        - options
        - stockmarket
      crypto:
        - cryptocurrency
        - bitcoin
        - ethereum
        - CryptoMarkets
    crawl_interval_minutes: 15
    backfill_days: 90
    rate_limit_per_minute: 60
    score_observation_window_minutes: 30

  market:
    provider: "yfinance"  # polygon | iex | yfinance
    equities:
      # Paper replication assets
      primary:
        - AAPL
        - JNJ
        - JPM
        - XOM
      # Related stocks for correlation features
      related:
        AAPL: [MSFT, GOOGL, META]
        JNJ: [PFE, MRK, UNH]
        JPM: [BAC, GS, MS]
        XOM: [CVX, COP, SLB]
      # Extended universe (production)
      extended:
        - NVDA
        - TSLA
        - AMD
        - AMZN
        - NFLX
        - DIS
        - BA
        - CAT
        - HD
        - MCD
        - NKE
        - SBUX
        - V
        - MA
        - PYPL
    crypto:
      - BTC/USDT
      - ETH/USDT
      - SOL/USDT

# Processing settings
processing:
  language:
    model: "lid.176.ftz"
    threshold: 0.8
    allowed_languages: ["en"]

  deduplication:
    enabled: true
    method: "minhash"
    num_perm: 128
    threshold: 0.8

  entity_linking:
    cashtag_regex: '\$[A-Z]{1,5}\b'
    confidence_threshold: 0.7
    use_embeddings: true
    embedding_model: "all-MiniLM-L6-v2"

# Sentiment settings
sentiment:
  baseline:
    method: "textblob"
    normalization: "sum_influence"  # sum_influence | post_count | none
    bucket_hours: 3

  upgrade:
    method: "finbert"
    model_name: "ProsusAI/finbert"
    batch_size: 32
    max_length: 512

  stance:
    enabled: true
    model_name: null  # Will use weak supervision + fine-tuning

  events:
    enabled: true
    types:
      - earnings
      - guidance
      - acquisition
      - regulation
      - lawsuit
      - product
      - hack
      - partnership
      - rumor

# Influence settings
influence:
  graph:
    min_interactions: 50
    edge_types:
      - reply
      - co_thread
      - mention

  gnn:
    model: "gcn"
    input_dim: 4
    hidden_dim: 64
    output_dim: 1
    num_layers: 2
    dropout: 0.5

  training:
    epochs: 100
    learning_rate: 0.01
    pseudo_label_method: "pagerank"

# Credibility settings
credibility:
  author_profiling:
    enabled: true
    lookback_days: 90
    min_posts: 5

  bot_detection:
    enabled: true
    posting_entropy_threshold: 2.0
    repetition_threshold: 0.3

  brigade_detection:
    enabled: true
    spike_threshold: 3.0
    similarity_threshold: 0.9
    time_window_minutes: 60

# Feature settings
features:
  window_sizes: [5, 15, 30]
  rolling_volatility_window: 30
  cutoff_time: "15:50"
  timezone: "US/Eastern"

  columns:
    baseline:
      - return_1d
      - volatility_30d
      - related_return_1
      - related_return_2
      - related_return_3
      - sentiment_index
    upgrade:
      - return_1d
      - return_5d
      - volatility_30d
      - related_return_1
      - related_return_2
      - related_return_3
      - sentiment_index
      - sentiment_mean
      - sentiment_std
      - stance_mean
      - stance_entropy
      - post_volume
      - unique_authors
      - influence_weighted_sentiment
      - credibility_weighted_sentiment
      - manipulation_adjusted_sentiment
      - rsi_14
      - macd_signal
      - volume_ratio

# Model settings
models:
  lstm:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.3

  cnn:
    num_filters: 32
    kernel_sizes: [3, 5, 7]
    dropout: 0.3

  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 128
    dropout: 0.3

  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 50
    early_stopping_patience: 5
    val_ratio: 0.2
    rolling_window_days: 90

# Trading settings
trading:
  policies:
    fixed_threshold:
      enabled: true  # For paper replication only
      is_leakage_demo: true
      thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    dynamic_threshold:
      enabled: true
      initial_threshold: 0.5
      learning_rate: 0.1
      volatility_window: 20
      min_threshold: 0.1
      max_threshold: 0.9

    portfolio:
      enabled: true
      max_leverage: 1.0
      max_single_position: 0.1
      max_sector_exposure: 0.3
      max_turnover_daily: 0.5
      target_volatility: 0.15
      min_position_size: 0.01
      long_only: false

  costs:
    commission_bps: 5.0
    spread_bps: 10.0
    market_impact_bps: 5.0
    slippage_vol_mult: 0.5

  risk:
    max_daily_loss_pct: 0.02
    max_drawdown_pct: 0.10
    max_single_loss_pct: 0.05

# Backtest settings
backtest:
  initial_capital: 100000.0
  execution_price: "open"  # open | close | vwap
  walk_forward:
    initial_train_days: 180
    retrain_frequency_days: 30
    test_window_days: 30
    purge_days: 5
    embargo_days: 5
    expanding_window: false

  statistics:
    bootstrap_samples: 10000
    confidence_level: 0.95

# Paper trading settings
paper_trade:
  enabled: true
  broker: "alpaca"
  schedule:
    ingest_time: "07:00"
    feature_time: "09:00"
    inference_time: "09:15"
    portfolio_time: "09:25"
    order_time: "09:30"
    close_time: "16:00"
    diagnostics_time: "18:00"
  timezone: "US/Eastern"

# Monitoring settings
monitoring:
  prometheus:
    enabled: true
    port: 9090

  alerting:
    enabled: true
    channels:
      - slack  # Configure webhook in secrets

  drift:
    enabled: true
    reference_window_days: 30
    threshold: 0.3

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/imst_quant.log"

# Database
database:
  postgres:
    host: "${POSTGRES_HOST:-localhost}"
    port: "${POSTGRES_PORT:-5432}"
    database: "${POSTGRES_DB:-imst_quant}"
    user: "${POSTGRES_USER:-postgres}"
    password: "${POSTGRES_PASSWORD}"
```

### 8.3 CLI Commands

```python
# src/imst_quant/cli/main.py
import typer
from rich.console import Console

app = typer.Typer(
    name="imst",
    help="IMST-Quant: Influence-aware Multi-Source Sentiment Trading",
    add_completion=False
)
console = Console()

# Import sub-commands
from imst_quant.cli import ingest, train, backtest, paper_trade, evaluate

app.add_typer(ingest.app, name="ingest", help="Data ingestion commands")
app.add_typer(train.app, name="train", help="Model training commands")
app.add_typer(backtest.app, name="backtest", help="Backtesting commands")
app.add_typer(paper_trade.app, name="paper", help="Paper trading commands")
app.add_typer(evaluate.app, name="eval", help="Evaluation commands")


@app.command()
def version():
    """Show version information."""
    from imst_quant import __version__
    console.print(f"IMST-Quant version {__version__}")


@app.command()
def init(
    config: str = typer.Option("default", help="Configuration preset"),
    force: bool = typer.Option(False, help="Overwrite existing setup")
):
    """Initialize project with data directories and config."""
    from imst_quant.config import initialize_project
    initialize_project(config, force)
    console.print("[green]Project initialized successfully![/green]")


@app.command()
def run_pipeline(
    config: str = typer.Option("default.yaml", help="Config file path"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
    mode: str = typer.Option("backtest", help="Mode: backtest | paper_trade")
):
    """Run complete pipeline from ingestion to trading."""
    from imst_quant.pipeline import run_full_pipeline
    run_full_pipeline(config, start_date, end_date, mode)


if __name__ == "__main__":
    app()
```

#### CLI Command Reference

```bash
# Project setup
imst init --config default
imst init --config paper_replication --force

# Ingestion
imst ingest reddit --subreddits wallstreetbets,stocks --days 30
imst ingest market --symbols AAPL,MSFT --start 2024-01-01
imst ingest crypto --symbols BTC/USDT,ETH/USDT
imst ingest all --config production.yaml

# Processing
imst process dedupe --input data/bronze/reddit
imst process entities --input data/bronze/reddit
imst process sentiment --method finbert --batch-size 32

# Training
imst train influence --month 2024-01
imst train forecast --model lstm --window 30 --date 2024-02-01
imst train all --config paper_replication.yaml

# Backtesting
imst backtest run --config backtest.yaml --start 2023-01-01 --end 2024-01-01
imst backtest walk-forward --folds 12 --train-days 180
imst backtest ablation --studies price_only,no_credibility
imst backtest report --output reports/backtest_2024.md

# Evaluation
imst eval causality --features data/gold/features.parquet
imst eval statistics --results results/backtest.json
imst eval compare --baseline lstm --candidate transformer

# Paper trading
imst paper start --capital 100000 --config production.yaml
imst paper stop
imst paper status
imst paper logs --date 2024-02-01

# Monitoring
imst monitor start --port 9090
imst monitor alerts --config alerting_rules.yml

# Utilities
imst validate config --file configs/production.yaml
imst validate features --file data/gold/features.parquet
```

### 8.4 Makefile

```makefile
.PHONY: help install dev test lint format clean docker

PYTHON := python
POETRY := poetry

help:
	@echo "IMST-Quant Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  dev           Install development dependencies"
	@echo "  download      Download required models and data"
	@echo ""
	@echo "Development:"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linters"
	@echo "  format        Format code"
	@echo "  typecheck     Run type checker"
	@echo ""
	@echo "Pipeline:"
	@echo "  ingest        Run data ingestion"
	@echo "  process       Run data processing"
	@echo "  train         Train all models"
	@echo "  backtest      Run backtest"
	@echo "  paper-trade   Start paper trading"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start Docker services"
	@echo "  docker-down   Stop Docker services"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Generate documentation"

# Setup
install:
	$(POETRY) install --no-dev

dev:
	$(POETRY) install
	pre-commit install

download:
	$(POETRY) run python scripts/download_models.py
	$(POETRY) run python scripts/download_data.py

# Development
test:
	$(POETRY) run pytest tests/ -v

test-cov:
	$(POETRY) run pytest tests/ -v --cov=src/imst_quant --cov-report=html

test-causality:
	$(POETRY) run pytest tests/unit/test_causality.py -v --tb=long

lint:
	$(POETRY) run ruff check src/ tests/
	$(POETRY) run black --check src/ tests/

format:
	$(POETRY) run ruff check --fix src/ tests/
	$(POETRY) run black src/ tests/

typecheck:
	$(POETRY) run mypy src/

# Pipeline
ingest:
	$(POETRY) run imst ingest all --config configs/$(CONFIG).yaml

process:
	$(POETRY) run imst process all --config configs/$(CONFIG).yaml

train:
	$(POETRY) run imst train all --config configs/$(CONFIG).yaml

backtest:
	$(POETRY) run imst backtest run --config configs/$(CONFIG).yaml \
		--start $(START_DATE) --end $(END_DATE)

backtest-paper:
	$(POETRY) run imst backtest run --config configs/paper_replication.yaml \
		--start 2022-01-01 --end 2023-12-31

backtest-ablation:
	$(POETRY) run imst backtest ablation --config configs/$(CONFIG).yaml \
		--studies price_only,sentiment_no_influence,no_credibility,no_top_influencers

paper-trade:
	$(POETRY) run imst paper start --config configs/production.yaml

# Full pipeline (reproducibility)
reproduce:
	$(MAKE) clean-data
	$(MAKE) ingest CONFIG=paper_replication
	$(MAKE) process CONFIG=paper_replication
	$(MAKE) train CONFIG=paper_replication
	$(MAKE) backtest-paper
	$(POETRY) run imst backtest report --output reports/replication_$(shell date +%Y%m%d).md

# Docker
docker-build:
	docker build -t imst-quant:latest -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Utilities
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-data:
	rm -rf data/bronze/ data/silver/ data/gold/

docs:
	$(POETRY) run mkdocs build

# CI targets
ci-lint:
	$(MAKE) lint
	$(MAKE) typecheck

ci-test:
	$(MAKE) test-cov
	$(MAKE) test-causality

ci-backtest:
	$(MAKE) backtest CONFIG=paper_replication START_DATE=2023-01-01 END_DATE=2023-06-30
```

---

*Continued in Part 6 (Evaluation Plan and Audit Pack)...*
