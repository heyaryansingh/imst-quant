# Roadmap: IMST-Quant

**Created:** 2026-02-17
**Core Value:** Faithful paper replication with production-grade upgrades

## Overview

| Milestone | Phases | Focus |
|-----------|--------|-------|
| M1: Paper Replication | 1-9 | Reproduce paper methodology exactly |
| M2: Production Upgrade | 10-15 | Modern NLP, credibility, paper trading |

---

## Phase 1: Project Setup & Data Ingestion Infrastructure

**Goal:** Establish project foundation with data ingestion pipelines

**Requirements:** ING-01, ING-02, ING-03, ING-04, ING-05, ING-06

**Success Criteria:**
1. Repository structure matches specification with all directories
2. Reddit ingestion fetches posts from configured subreddits
3. Market data ingestion fetches OHLCV for 4 paper stocks
4. Raw data stored in `/data/raw/` with correct timestamps
5. Bronze layer parquet files generated successfully

**Dependencies:** None
**Status:** ● Complete (2026-02-18)

---

## Phase 2: Text Processing & Entity Linking

**Goal:** Process raw text and link mentions to assets

**Requirements:** ING-07, ING-08, ENT-01, ENT-02, ENT-03, ENT-04

**Success Criteria:**
1. Language detection filters non-English posts (>95% accuracy)
2. Deduplication removes near-duplicates (MinHash working)
3. Cashtag extraction identifies $TICKER patterns
4. Entity linker achieves >90% precision on test set
5. Silver layer parquet files contain entity links

**Dependencies:** Phase 1

---

## Phase 3: Baseline Sentiment Analysis (Paper Replication)

**Goal:** Implement exact paper sentiment methodology

**Requirements:** SENT-01, SENT-02, SENT-03, SENT-04

**Success Criteria:**
1. TextBlob polarity scores computed for all posts
2. 3-hour bucketing implemented correctly
3. Daily aggregation produces sentiment index
4. Influence weighting formula matches paper
5. Unit tests validate sentiment computation

**Dependencies:** Phase 2

---

## Phase 4: Influence Graph & GNN (Paper Replication)

**Goal:** Build monthly influence graphs and train GCN

**Requirements:** INF-01, INF-02, INF-03, INF-04, INF-05

**Success Criteria:**
1. Monthly interaction graphs constructed from Reddit data
2. Graphs filtered by min interaction threshold (50)
3. 2-layer GCN trains and converges
4. Influence scores generated per user per month
5. Scores carried forward for daily inference

**Dependencies:** Phase 2

---

## Phase 5: Credibility & Bot Detection

**Goal:** Implement author credibility scoring

**Requirements:** CRED-01, CRED-02, CRED-03, CRED-04

**Success Criteria:**
1. Author profiles computed with posting metrics
2. Bot probability estimated per author
3. Brigade detection identifies coordinated posts
4. Manipulation risk alerts trigger correctly
5. Credibility scores available for sentiment weighting

**Dependencies:** Phase 2

---

## Phase 6: Feature Engineering with Causality

**Goal:** Build as-of correct feature store

**Requirements:** FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06

**Success Criteria:**
1. Daily feature vectors computed with strict cutoffs
2. Price features use T-1 data only
3. Sentiment features use posts before cutoff
4. All 47 causality tests pass
5. Gold layer parquet files generated

**Dependencies:** Phases 3, 4, 5

---

## Phase 7: Forecasting Models (Paper Replication)

**Goal:** Implement LSTM/CNN/Transformer classifiers

**Requirements:** FORE-01, FORE-02, FORE-03, FORE-04, FORE-05

**Success Criteria:**
1. LSTM classifier trains and predicts
2. CNN classifier trains and predicts
3. Transformer classifier trains and predicts
4. Daily retraining pipeline works
5. Predictions match paper accuracy within 5%

**Dependencies:** Phase 6

---

## Phase 8: Trading Policies (Paper Replication)

**Goal:** Implement threshold-based trading

**Requirements:** TRAD-01, TRAD-02, TRAD-03

**Success Criteria:**
1. Fixed threshold implemented (labeled as leakage demo)
2. Dynamic threshold updates incrementally
3. Trading signals generated from predictions
4. Documentation clearly marks leakage method
5. Unit tests validate signal generation

**Dependencies:** Phase 7

---

## Phase 9: Backtesting & Validation

**Goal:** Run walk-forward backtest with full evaluation

**Requirements:** BACK-01, BACK-02, BACK-03, BACK-04, BACK-05, BACK-06, BACK-07

**Success Criteria:**
1. Event-driven backtester runs successfully
2. Transaction costs applied correctly
3. Walk-forward with purging/embargo implemented
4. Bootstrap CIs computed for key metrics
5. Ablation studies run and documented
6. Backtest report generated with all metrics
7. Results match paper within tolerance

**Dependencies:** Phase 8

---

## Phase 10: Upgraded Sentiment (Production)

**Goal:** Add modern NLP models

**Requirements:** SENT-U01, SENT-U02, SENT-U03, SENT-U04, SENT-U05

**Success Criteria:**
1. FinBERT inference runs on posts
2. Stance classification (bullish/bearish/neutral) works
3. Event tagging identifies key events
4. Credibility-weighted sentiment computed
5. Manipulation-adjusted sentiment computed
6. Ablation shows improvement over baseline

**Dependencies:** Phase 9

---

## Phase 11: Upgraded Forecasting (Production)

**Goal:** Add LightGBM and calibration

**Requirements:** FORE-U01, FORE-U02, FORE-U03

**Success Criteria:**
1. LightGBM cross-sectional model trains
2. Prediction probabilities calibrated
3. MLflow tracking operational
4. Model comparison shows best performer
5. Ensemble option available

**Dependencies:** Phase 10

---

## Phase 12: Upgraded Trading (Production)

**Goal:** Add portfolio policy with risk management

**Requirements:** TRAD-U01, TRAD-U02, TRAD-U03, TRAD-U04

**Success Criteria:**
1. Portfolio policy respects constraints
2. Transaction costs modeled realistically
3. Risk limits enforced (drawdown, exposure)
4. Kill switch triggers on limit breach
5. Backtest shows improved risk-adjusted returns

**Dependencies:** Phase 11

---

## Phase 13: Paper Trading System

**Goal:** Deploy automated daily trading

**Requirements:** PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05

**Success Criteria:**
1. Daily scheduler runs at correct times
2. Alpaca paper trading integration works
3. All inputs/predictions/fills logged
4. Kill switch halts trading on breach
5. P&L tracking accurate and auditable
6. 7 consecutive days run without errors

**Dependencies:** Phase 12

---

## Phase 14: Monitoring & Observability

**Goal:** Production monitoring infrastructure

**Requirements:** MON-01, MON-02, MON-03, MON-04, MON-05

**Success Criteria:**
1. Prometheus metrics exposed and scraped
2. Grafana dashboards display key metrics
3. Data outage alerts fire correctly
4. Feature drift detection operational
5. Manipulation risk alerts trigger

**Dependencies:** Phase 13

---

## Phase 15: Reproducibility & Documentation

**Goal:** Complete audit pack and reproducibility

**Requirements:** REPR-01, REPR-02, REPR-03, REPR-04, REPR-05

**Success Criteria:**
1. `make reproduce` rebuilds everything from scratch
2. Prediction → input tracing works via hashes
3. All causality tests pass in CI
4. MLflow has all model versions
5. Docker deployment works end-to-end
6. All documentation complete
7. Model cards and datasheets finalized

**Dependencies:** Phase 14

---

## Phase Summary

| Phase | Name | Requirements | Est. Effort |
|-------|------|--------------|-------------|
| 1 | Project Setup & Ingestion | 6 | Medium |
| 2 | Text Processing & Entity Linking | 6 | Medium |
| 3 | Baseline Sentiment | 4 | Low |
| 4 | Influence Graph & GNN | 5 | High |
| 5 | Credibility & Bot Detection | 4 | Medium |
| 6 | Feature Engineering | 6 | High |
| 7 | Forecasting Models | 5 | High |
| 8 | Trading Policies | 3 | Low |
| 9 | Backtesting & Validation | 7 | High |
| 10 | Upgraded Sentiment | 5 | Medium |
| 11 | Upgraded Forecasting | 3 | Medium |
| 12 | Upgraded Trading | 4 | Medium |
| 13 | Paper Trading | 5 | Medium |
| 14 | Monitoring | 5 | Medium |
| 15 | Reproducibility | 5 | Low |

**Total Requirements:** 62
**Total Phases:** 15

---

## Milestones

### Milestone 1: Paper Replication (Phases 1-9)
**Goal:** Reproduce paper results with same methodology
**Deliverables:**
- Working ingestion pipeline
- GCN influence model
- LSTM/CNN/Transformer forecasters
- Dynamic threshold trading
- Walk-forward backtest results
- Comparison to paper metrics

### Milestone 2: Production Upgrade (Phases 10-15)
**Goal:** Production-ready system with modern NLP
**Deliverables:**
- FinBERT sentiment
- Credibility filtering
- Portfolio optimization
- Automated paper trading
- Full monitoring stack
- Complete documentation

---
*Roadmap created: 2026-02-17*
*Last updated: 2026-02-17 after initial creation*
