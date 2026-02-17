# Requirements: IMST-Quant

**Defined:** 2026-02-17
**Core Value:** Faithful paper replication with production-grade upgrades for GNN-based sentiment trading

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Ingestion (ING)

- [ ] **ING-01**: System ingests Reddit posts/comments from configured subreddits via PRAW
- [ ] **ING-02**: System stores raw JSON with retrieval timestamp for causality
- [ ] **ING-03**: System handles rate limits and implements incremental crawling
- [ ] **ING-04**: System fetches daily OHLCV data for configured equity universe
- [ ] **ING-05**: System fetches daily OHLCV data for configured crypto universe
- [ ] **ING-06**: System normalizes raw data to bronze/silver/gold parquet layers
- [ ] **ING-07**: System deduplicates posts using MinHash similarity
- [ ] **ING-08**: System detects and filters non-English posts

### Entity Linking (ENT)

- [ ] **ENT-01**: System extracts cashtags via regex pattern matching
- [ ] **ENT-02**: System maps company names/aliases to tickers via dictionary
- [ ] **ENT-03**: System disambiguates entities using contextual embeddings
- [ ] **ENT-04**: System achieves >90% precision on labeled test set (200 samples)

### Baseline Sentiment (SENT)

- [ ] **SENT-01**: System computes TextBlob polarity scores (paper replication)
- [ ] **SENT-02**: System aggregates sentiment into 3-hour buckets then daily
- [ ] **SENT-03**: System weights sentiment by author influence scores
- [ ] **SENT-04**: System computes sentiment index per asset per day

### Upgraded Sentiment (SENT-U)

- [ ] **SENT-U01**: System runs FinBERT inference for sentiment classification
- [ ] **SENT-U02**: System classifies stance (bullish/bearish/neutral)
- [ ] **SENT-U03**: System tags event types (earnings, M&A, regulation, etc.)
- [ ] **SENT-U04**: System computes credibility-weighted sentiment aggregates
- [ ] **SENT-U05**: System computes manipulation-adjusted sentiment

### Influence Modeling (INF)

- [ ] **INF-01**: System builds monthly interaction graphs from Reddit data
- [ ] **INF-02**: System trains 2-layer GCN on interaction graphs
- [ ] **INF-03**: System outputs per-user monthly influence scores
- [ ] **INF-04**: System filters users by minimum interaction threshold (default 50)
- [ ] **INF-05**: System carries forward influence scores for daily inference

### Credibility/Bot Detection (CRED)

- [ ] **CRED-01**: System computes author credibility features (posting cadence, diversity)
- [ ] **CRED-02**: System estimates bot probability per author
- [ ] **CRED-03**: System detects coordinated brigade patterns
- [ ] **CRED-04**: System alerts on high manipulation risk

### Feature Engineering (FEAT)

- [ ] **FEAT-01**: System builds daily feature vectors with strict as-of semantics
- [ ] **FEAT-02**: System computes price features (returns, volatility)
- [ ] **FEAT-03**: System computes related stock features
- [ ] **FEAT-04**: System computes sentiment aggregate features
- [ ] **FEAT-05**: System enforces causality (no lookahead) via automated tests
- [ ] **FEAT-06**: System stores features in gold layer parquet files

### Forecasting - Baseline (FORE)

- [ ] **FORE-01**: System trains LSTM model with configurable window sizes
- [ ] **FORE-02**: System trains CNN model with configurable kernels
- [ ] **FORE-03**: System trains Transformer model for direction prediction
- [ ] **FORE-04**: System retrains daily on rolling 90-day window
- [ ] **FORE-05**: System predicts next-day return direction

### Forecasting - Upgrade (FORE-U)

- [ ] **FORE-U01**: System supports LightGBM cross-sectional ranking
- [ ] **FORE-U02**: System outputs calibrated prediction probabilities
- [ ] **FORE-U03**: System logs model performance metrics to MLflow

### Trading - Baseline (TRAD)

- [ ] **TRAD-01**: System implements fixed threshold (labeled as leakage demo)
- [ ] **TRAD-02**: System implements dynamic threshold with incremental updates
- [ ] **TRAD-03**: System generates trading signals from predictions

### Trading - Upgrade (TRAD-U)

- [ ] **TRAD-U01**: System implements portfolio policy with constraints
- [ ] **TRAD-U02**: System applies transaction cost model
- [ ] **TRAD-U03**: System enforces risk limits (max drawdown, exposure caps)
- [ ] **TRAD-U04**: System implements kill switch on limit breach

### Backtesting (BACK)

- [ ] **BACK-01**: System runs event-driven backtest with daily bars
- [ ] **BACK-02**: System applies configurable transaction costs
- [ ] **BACK-03**: System computes standard performance metrics (Sharpe, CAGR, etc.)
- [ ] **BACK-04**: System runs walk-forward validation with purging/embargo
- [ ] **BACK-05**: System computes bootstrap confidence intervals
- [ ] **BACK-06**: System runs ablation studies
- [ ] **BACK-07**: System generates backtest reports

### Paper Trading (PAPER)

- [ ] **PAPER-01**: System runs daily automated trading loop
- [ ] **PAPER-02**: System submits paper orders to Alpaca sandbox
- [ ] **PAPER-03**: System logs all inputs, predictions, and fills
- [ ] **PAPER-04**: System implements kill switch on drawdown breach
- [ ] **PAPER-05**: System tracks daily P&L and cumulative performance

### Monitoring (MON)

- [ ] **MON-01**: System exposes Prometheus metrics
- [ ] **MON-02**: System displays Grafana dashboards
- [ ] **MON-03**: System alerts on data outage
- [ ] **MON-04**: System detects feature drift
- [ ] **MON-05**: System alerts on manipulation risk spikes

### Reproducibility (REPR)

- [ ] **REPR-01**: Single command rebuilds from raw data to backtest
- [ ] **REPR-02**: All predictions traceable to input data hashes
- [ ] **REPR-03**: Causality test suite passes (no lookahead)
- [ ] **REPR-04**: Model versions tracked in MLflow
- [ ] **REPR-05**: Docker deployment works end-to-end

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Multi-Source Integration

- **MS-01**: System ingests X/Twitter data via API
- **MS-02**: System ingests StockTwits data
- **MS-03**: System ingests Threads data
- **MS-04**: System normalizes across sources

### Advanced Models

- **ADV-01**: System supports Temporal Fusion Transformer
- **ADV-02**: System supports multi-asset graph-time models
- **ADV-03**: System supports ensemble methods

### Real Trading

- **REAL-01**: System supports live trading with real broker
- **REAL-02**: System implements position management
- **REAL-03**: System handles corporate actions automatically

### Scale

- **SCALE-01**: System handles 500+ asset universe
- **SCALE-02**: System processes 1M+ posts daily
- **SCALE-03**: System supports distributed training

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real money trading | Paper trading only for v1 - risk management |
| HFT/minute-bar execution | Daily bars sufficient for thesis validation |
| Options/derivatives | Complexity; equities and crypto spot only |
| Mobile/web UI | CLI + API sufficient for v1 |
| Fundamental data integration | Focus on sentiment signal |
| Multi-language NLP | English only for v1 |
| News sentiment | RSS only for v1, full news in v2 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ING-01 | Phase 1 | Pending |
| ING-02 | Phase 1 | Pending |
| ING-03 | Phase 1 | Pending |
| ING-04 | Phase 1 | Pending |
| ING-05 | Phase 1 | Pending |
| ING-06 | Phase 1 | Pending |
| ING-07 | Phase 2 | Pending |
| ING-08 | Phase 2 | Pending |
| ENT-01 | Phase 2 | Pending |
| ENT-02 | Phase 2 | Pending |
| ENT-03 | Phase 2 | Pending |
| ENT-04 | Phase 2 | Pending |
| SENT-01 | Phase 3 | Pending |
| SENT-02 | Phase 3 | Pending |
| SENT-03 | Phase 3 | Pending |
| SENT-04 | Phase 3 | Pending |
| INF-01 | Phase 4 | Pending |
| INF-02 | Phase 4 | Pending |
| INF-03 | Phase 4 | Pending |
| INF-04 | Phase 4 | Pending |
| INF-05 | Phase 4 | Pending |
| CRED-01 | Phase 5 | Pending |
| CRED-02 | Phase 5 | Pending |
| CRED-03 | Phase 5 | Pending |
| CRED-04 | Phase 5 | Pending |
| FEAT-01 | Phase 6 | Pending |
| FEAT-02 | Phase 6 | Pending |
| FEAT-03 | Phase 6 | Pending |
| FEAT-04 | Phase 6 | Pending |
| FEAT-05 | Phase 6 | Pending |
| FEAT-06 | Phase 6 | Pending |
| FORE-01 | Phase 7 | Pending |
| FORE-02 | Phase 7 | Pending |
| FORE-03 | Phase 7 | Pending |
| FORE-04 | Phase 7 | Pending |
| FORE-05 | Phase 7 | Pending |
| TRAD-01 | Phase 8 | Pending |
| TRAD-02 | Phase 8 | Pending |
| TRAD-03 | Phase 8 | Pending |
| BACK-01 | Phase 9 | Pending |
| BACK-02 | Phase 9 | Pending |
| BACK-03 | Phase 9 | Pending |
| BACK-04 | Phase 9 | Pending |
| BACK-05 | Phase 9 | Pending |
| BACK-06 | Phase 9 | Pending |
| BACK-07 | Phase 9 | Pending |
| SENT-U01 | Phase 10 | Pending |
| SENT-U02 | Phase 10 | Pending |
| SENT-U03 | Phase 10 | Pending |
| SENT-U04 | Phase 10 | Pending |
| SENT-U05 | Phase 10 | Pending |
| FORE-U01 | Phase 11 | Pending |
| FORE-U02 | Phase 11 | Pending |
| FORE-U03 | Phase 11 | Pending |
| TRAD-U01 | Phase 12 | Pending |
| TRAD-U02 | Phase 12 | Pending |
| TRAD-U03 | Phase 12 | Pending |
| TRAD-U04 | Phase 12 | Pending |
| PAPER-01 | Phase 13 | Pending |
| PAPER-02 | Phase 13 | Pending |
| PAPER-03 | Phase 13 | Pending |
| PAPER-04 | Phase 13 | Pending |
| PAPER-05 | Phase 13 | Pending |
| MON-01 | Phase 14 | Pending |
| MON-02 | Phase 14 | Pending |
| MON-03 | Phase 14 | Pending |
| MON-04 | Phase 14 | Pending |
| MON-05 | Phase 14 | Pending |
| REPR-01 | Phase 15 | Pending |
| REPR-02 | Phase 15 | Pending |
| REPR-03 | Phase 15 | Pending |
| REPR-04 | Phase 15 | Pending |
| REPR-05 | Phase 15 | Pending |

**Coverage:**
- v1 requirements: 62 total
- Mapped to phases: 62
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-17*
*Last updated: 2026-02-17 after initial definition*
