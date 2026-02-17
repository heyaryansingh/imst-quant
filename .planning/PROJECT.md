# IMST-Quant: Influence-aware Multi-Source Sentiment Trading

## What This Is

A production-grade system that replicates and extends the research paper "GNN-based social media sentiment analysis for stock market forecasting and trading." The system ingests multi-source social media data (Reddit primary, with architecture for X/StockTwits/Threads), builds monthly influence graphs using GNNs, computes sentiment indices, forecasts asset returns, and executes algorithmic trading strategies with full audit trails and risk controls.

## Core Value

**Faithful paper replication with production-grade upgrades**: First reproduce the paper's methodology exactly (TextBlob + GCN influence + LSTM/CNN/Transformer forecasting + threshold trading), then layer modern NLP (FinBERT), credibility filtering, walk-forward evaluation, and live paper trading — all with strict temporal causality guarantees.

## Requirements

### Validated

(None yet — ship to validate)

### Active

#### Data Ingestion
- [ ] **ING-01**: System ingests Reddit posts/comments from configured subreddits via PRAW
- [ ] **ING-02**: System stores raw JSON with retrieval timestamp for causality
- [ ] **ING-03**: System handles rate limits and implements incremental crawling
- [ ] **ING-04**: System fetches daily OHLCV data for configured equity universe
- [ ] **ING-05**: System fetches daily OHLCV data for configured crypto universe
- [ ] **ING-06**: System normalizes raw data to bronze/silver/gold parquet layers

#### Entity Linking
- [ ] **ENT-01**: System extracts cashtags via regex pattern matching
- [ ] **ENT-02**: System maps company names/aliases to tickers via dictionary
- [ ] **ENT-03**: System disambiguates entities using contextual embeddings
- [ ] **ENT-04**: System achieves >90% precision on labeled test set (200 samples)

#### Text Processing
- [ ] **TXT-01**: System detects and filters non-English posts
- [ ] **TXT-02**: System deduplicates near-identical posts via MinHash
- [ ] **TXT-03**: System normalizes text while preserving raw original

#### Baseline Sentiment (Paper Replication)
- [ ] **SENT-01**: System computes TextBlob polarity scores
- [ ] **SENT-02**: System aggregates sentiment into 3-hour buckets then daily
- [ ] **SENT-03**: System weights sentiment by author influence scores

#### Upgraded Sentiment
- [ ] **SENT-04**: System runs FinBERT inference for sentiment classification
- [ ] **SENT-05**: System classifies stance (bullish/bearish/neutral)
- [ ] **SENT-06**: System tags event types (earnings, M&A, etc.)
- [ ] **SENT-07**: System computes credibility-weighted sentiment aggregates

#### Influence Modeling
- [ ] **INF-01**: System builds monthly interaction graphs from Reddit data
- [ ] **INF-02**: System trains 2-layer GCN on interaction graphs
- [ ] **INF-03**: System outputs per-user monthly influence scores
- [ ] **INF-04**: System filters users by minimum interaction threshold (default 50)

#### Credibility/Bot Detection
- [ ] **CRED-01**: System computes author credibility features
- [ ] **CRED-02**: System estimates bot probability per author
- [ ] **CRED-03**: System detects coordinated brigade patterns

#### Forecasting (Baseline)
- [ ] **FORE-01**: System trains LSTM model with 5/15/30 day windows
- [ ] **FORE-02**: System trains CNN model with configurable kernels
- [ ] **FORE-03**: System trains Transformer model for direction prediction
- [ ] **FORE-04**: System retrains daily on rolling 90-day window
- [ ] **FORE-05**: System predicts next-day return direction

#### Forecasting (Upgrade)
- [ ] **FORE-06**: System supports Temporal Fusion Transformer
- [ ] **FORE-07**: System supports LightGBM cross-sectional ranking
- [ ] **FORE-08**: System outputs calibrated prediction probabilities

#### Trading (Baseline)
- [ ] **TRAD-01**: System implements fixed threshold (labeled as leakage demo)
- [ ] **TRAD-02**: System implements dynamic threshold with incremental updates

#### Trading (Upgrade)
- [ ] **TRAD-03**: System implements portfolio policy with constraints
- [ ] **TRAD-04**: System applies transaction cost model
- [ ] **TRAD-05**: System enforces risk limits (max drawdown, exposure caps)

#### Backtesting
- [ ] **BACK-01**: System runs event-driven backtest with daily bars
- [ ] **BACK-02**: System applies configurable transaction costs
- [ ] **BACK-03**: System computes standard performance metrics
- [ ] **BACK-04**: System runs walk-forward validation with purging/embargo
- [ ] **BACK-05**: System computes bootstrap confidence intervals

#### Paper Trading
- [ ] **PAPER-01**: System runs daily automated trading loop
- [ ] **PAPER-02**: System submits paper orders to Alpaca sandbox
- [ ] **PAPER-03**: System logs all inputs, predictions, and fills
- [ ] **PAPER-04**: System implements kill switch on drawdown breach

#### Monitoring
- [ ] **MON-01**: System exposes Prometheus metrics
- [ ] **MON-02**: System displays Grafana dashboards
- [ ] **MON-03**: System alerts on data outage and drift

#### Reproducibility
- [ ] **REPR-01**: Single command rebuilds from raw data to backtest
- [ ] **REPR-02**: All predictions traceable to input data hashes
- [ ] **REPR-03**: Causality test suite passes (no lookahead)

### Out of Scope

- Real money trading — paper trading only for this milestone
- HFT/minute-bar execution — daily bars for simplicity
- Options/derivatives — equities and crypto spot only
- Fundamental data integration — social sentiment focus
- Mobile/web UI — CLI and API only

## Context

**Research Foundation**: The paper demonstrates that GNN-derived influence indices combined with sentiment analysis can predict stock returns. Key innovations include monthly influence graph construction and incremental threshold learning.

**Known Gaps in Paper**:
1. Cashtag-only entity linking misses mentions like "Apple" or "NVIDIA"
2. TextBlob sentiment is outdated; finance-tuned models perform better
3. Fixed threshold method has lookahead bias (uses future data to optimize)
4. No credibility filtering makes system vulnerable to manipulation
5. Limited evaluation without walk-forward and proper purging

**Technical Environment**:
- Python 3.11+ ecosystem
- PyTorch + PyTorch Geometric for GNN
- Hugging Face Transformers for NLP
- Alpaca for paper trading
- PostgreSQL + Parquet for storage

## Constraints

- **API Access**: Reddit via PRAW (official API); no X/Twitter access assumed initially
- **Compute**: Must run on single GPU (RTX 3080 or equivalent) for baseline
- **Latency**: Daily predictions acceptable; sub-hour not required for v1
- **Cost**: Minimize API costs; use free tiers where possible
- **Compliance**: Respect Reddit ToS; hash author IDs; no PII storage

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Reddit as primary source | Official API available; active finance communities | — Pending |
| TextBlob for baseline sentiment | Exact paper replication required | — Pending |
| GCNConv for influence GNN | Paper uses GCN architecture | — Pending |
| Daily retraining schedule | Paper specifies 90-day rolling window | — Pending |
| Alpaca for paper trading | Free paper trading API; good documentation | — Pending |
| Polars over pandas | Performance for large datasets | — Pending |
| Walk-forward with purging | Industry standard for avoiding lookahead | — Pending |

---
*Last updated: 2026-02-17 after initialization*
