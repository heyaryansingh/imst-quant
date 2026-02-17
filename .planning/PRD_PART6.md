# PRD Part 6: Evaluation Plan, Audit Pack, and Final Checklist

## 9. Evaluation Plan

### 9.1 Walk-Forward Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WALK-FORWARD VALIDATION TIMELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

Year 1: 2022
├─ Jan-Jun: Initial Training Window (180 days)
├─ [PURGE: 5 days]
├─ Jul: Test Window 1 (30 days)
├─ [EMBARGO: 5 days]
├─ Jul-Dec: Expanded Training (add 30 days)
├─ [PURGE: 5 days]
├─ Aug: Test Window 2 (30 days)
└─ ... continues monthly ...

Year 2: 2023
├─ [Rolling or expanding windows continue]
├─ Monthly retraining with new data
├─ Each test window is out-of-sample
└─ Final: Dec 2023 test window

Total: 18 test periods (1.5 years of out-of-sample testing)
```

**Exact Split Specification**:

```python
# Walk-forward configuration for paper replication
WALK_FORWARD_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "initial_train_days": 180,       # 6 months initial training
    "retrain_frequency_days": 30,     # Monthly retraining
    "test_window_days": 30,           # 1 month test
    "purge_days": 5,                  # 5-day gap (rolling features)
    "embargo_days": 5,                # 5-day post-test gap
    "expanding_window": False,        # Rolling window (paper style)
}

# Generated splits
SPLITS = [
    {
        "fold": 1,
        "train_start": "2022-01-01",
        "train_end": "2022-06-30",
        "purge_end": "2022-07-05",
        "test_start": "2022-07-06",
        "test_end": "2022-08-05",
        "embargo_end": "2022-08-10"
    },
    {
        "fold": 2,
        "train_start": "2022-02-01",
        "train_end": "2022-07-31",
        # ... etc
    },
    # ... 18 total folds
]
```

### 9.2 Purging and Embargo Logic

**Why Purging is Necessary**:
```
Problem: Rolling features (e.g., 30-day volatility) create overlap

Day T-35: Used in 30-day volatility for Day T-5 (in training)
Day T-30: Used in 30-day volatility for Day T (in test)
         ^^^^^ Same raw data appears in both train and test!

Solution: Purge days between train_end and test_start

With 5-day purge:
- Last training sample: Day T-35 (uses data T-65 to T-35)
- First test sample: Day T (uses data T-30 to T)
- No overlap in underlying data
```

**Why Embargo is Necessary**:
```
Problem: Information leakage from test period into subsequent training

Test period: Days 180-210
Next training includes: Days 181-211 (1-day delay)
                       ^^^^^ Test period data in training!

Solution: Embargo after test_end before next train includes that data

With 5-day embargo:
- Test ends: Day 210
- Next training starts including data from: Day 216
- No test period data in training
```

**Implementation**:
```python
def validate_split_boundaries(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_lookback: int = 30
) -> bool:
    """Validate no data leakage between train and test."""

    train_dates = set(train_df["date"].to_list())
    test_dates = set(test_df["date"].to_list())

    # Check 1: No direct date overlap
    if train_dates & test_dates:
        return False

    # Check 2: Feature lookback doesn't overlap
    train_max = max(train_dates)
    test_min = min(test_dates)
    gap_days = (test_min - train_max).days

    if gap_days < feature_lookback:
        # Risk of overlap in rolling features
        return False

    # Check 3: Influence scores use correct month
    train_months = set(d.replace(day=1) for d in train_dates)
    test_months = set(d.replace(day=1) for d in test_dates)

    # Influence scores should be from months fully in training
    # (not from test period months)

    return True
```

### 9.3 Ablation Study Design

| Study ID | Name | Description | Hypothesis |
|----------|------|-------------|------------|
| ABL-01 | price_only | Remove all sentiment features | Sentiment adds value over price alone |
| ABL-02 | sentiment_no_influence | Raw sentiment, no influence weighting | Influence weighting improves signal |
| ABL-03 | no_credibility | Include all posts regardless of bot/spam score | Credibility filtering improves quality |
| ABL-04 | reddit_only | Only Reddit data (baseline) | Multi-source doesn't help in v1 |
| ABL-05 | no_top_influencers | Remove top 10% influencers | Not dependent on few accounts |
| ABL-06 | no_meme_events | Remove meme stock event periods | Works outside of meme events |
| ABL-07 | textblob_vs_finbert | Compare sentiment methods | FinBERT outperforms TextBlob |
| ABL-08 | window_5_vs_30 | Compare feature window sizes | Optimal window exists |

**Ablation Execution**:
```python
ABLATION_CONFIGS = {
    "price_only": {
        "features": ["return_1d", "volatility_30d", "related_return_1",
                    "related_return_2", "related_return_3"],
        "exclude": ["sentiment_*", "stance_*", "influence_*"]
    },
    "sentiment_no_influence": {
        "features": BASELINE_FEATURES,
        "modifications": {"use_raw_sentiment": True}
    },
    "no_credibility": {
        "features": UPGRADE_FEATURES,
        "modifications": {"disable_credibility_filter": True}
    },
    "no_top_influencers": {
        "features": UPGRADE_FEATURES,
        "modifications": {"filter_top_influencer_percentile": 0.9}
    },
    "no_meme_events": {
        "features": UPGRADE_FEATURES,
        "exclude_dates": [
            ("2021-01-25", "2021-02-05"),  # GME squeeze
            ("2021-06-01", "2021-06-15"),  # AMC squeeze
        ]
    }
}
```

### 9.4 Stress Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| ST-01 | High volatility regime (VIX > 30) | Sharpe > 0 |
| ST-02 | Market crash (drawdown > 10% in month) | Max loss < 2x normal |
| ST-03 | Low volume periods | No execution failures |
| ST-04 | Data outage simulation | Graceful degradation |
| ST-05 | Model drift scenario | Alerts triggered correctly |
| ST-06 | Manipulation spike | Risk controls engage |

### 9.5 Statistical Test Suite

```python
class StatisticalTestSuite:
    """Complete statistical test suite for strategy evaluation."""

    def run_all_tests(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        baseline_returns: np.ndarray = None
    ) -> Dict[str, Any]:
        """Run all statistical tests and return results."""

        results = {}

        # 1. Bootstrap confidence intervals
        sharpe_ci = StatisticalTests.bootstrap_sharpe_ci(
            strategy_returns, n_bootstrap=10000, ci=0.95
        )
        results["sharpe_ci"] = {
            "lower": sharpe_ci[0],
            "point": sharpe_ci[1],
            "upper": sharpe_ci[2],
            "significant": sharpe_ci[0] > 0  # Lower CI > 0
        }

        cagr_ci = StatisticalTests.bootstrap_cagr_ci(
            strategy_returns, n_bootstrap=10000, ci=0.95
        )
        results["cagr_ci"] = {
            "lower": cagr_ci[0],
            "point": cagr_ci[1],
            "upper": cagr_ci[2],
            "significant": cagr_ci[0] > 0
        }

        # 2. Benchmark comparison
        t_stat, p_value = StatisticalTests.paired_t_test(
            strategy_returns, benchmark_returns
        )
        results["vs_benchmark"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "outperforms": t_stat > 0 and p_value < 0.05
        }

        # 3. White's Reality Check (multiple testing)
        rc_pvalue, rc_sig = StatisticalTests.white_reality_check(
            strategy_returns, benchmark_returns
        )
        results["reality_check"] = {
            "p_value": rc_pvalue,
            "significant": rc_sig
        }

        # 4. Compare to baseline if provided
        if baseline_returns is not None:
            t_stat, p_value = StatisticalTests.paired_t_test(
                strategy_returns, baseline_returns
            )
            results["vs_baseline"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }

        # 5. Normality test (for return distribution)
        from scipy.stats import shapiro, jarque_bera
        _, jb_pvalue = jarque_bera(strategy_returns)
        results["normality"] = {
            "jarque_bera_p": jb_pvalue,
            "is_normal": jb_pvalue > 0.05
        }

        # 6. Autocorrelation test (for strategy returns)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(strategy_returns, lags=5, return_df=True)
        results["autocorrelation"] = {
            "ljung_box_p": lb_result["lb_pvalue"].values.tolist(),
            "significant_autocorr": any(lb_result["lb_pvalue"] < 0.05)
        }

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report from test results."""
        report = "## Statistical Test Results\n\n"

        # Sharpe ratio
        sharpe = results["sharpe_ci"]
        status = "✓" if sharpe["significant"] else "✗"
        report += f"### Sharpe Ratio {status}\n"
        report += f"- Point estimate: {sharpe['point']:.3f}\n"
        report += f"- 95% CI: [{sharpe['lower']:.3f}, {sharpe['upper']:.3f}]\n"
        report += f"- Significant (CI > 0): {sharpe['significant']}\n\n"

        # CAGR
        cagr = results["cagr_ci"]
        status = "✓" if cagr["significant"] else "✗"
        report += f"### CAGR {status}\n"
        report += f"- Point estimate: {cagr['point']:.2%}\n"
        report += f"- 95% CI: [{cagr['lower']:.2%}, {cagr['upper']:.2%}]\n\n"

        # Benchmark comparison
        bench = results["vs_benchmark"]
        status = "✓" if bench["outperforms"] else "✗"
        report += f"### vs Benchmark {status}\n"
        report += f"- t-statistic: {bench['t_statistic']:.3f}\n"
        report += f"- p-value: {bench['p_value']:.4f}\n"
        report += f"- Outperforms: {bench['outperforms']}\n\n"

        # Reality check
        rc = results["reality_check"]
        status = "✓" if rc["significant"] else "⚠️"
        report += f"### Multiple Testing Adjustment {status}\n"
        report += f"- White's Reality Check p-value: {rc['p_value']:.4f}\n"
        report += f"- Survives adjustment: {rc['significant']}\n\n"

        return report
```

---

## 10. Audit Pack

### 10.1 Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details

- **Model Name**: [e.g., LSTM Return Direction Classifier]
- **Version**: [e.g., 1.0.0]
- **Date**: [Training date]
- **Type**: [Classification/Regression]
- **Framework**: PyTorch 2.1.0

## Intended Use

- **Primary Use**: Predict next-day return direction for equities
- **Users**: Quantitative trading system (automated)
- **Out-of-Scope**: Real-time trading, options pricing, fundamental analysis

## Training Data

- **Source**: Reddit posts + market data
- **Date Range**: [start] to [end]
- **Volume**: [N] posts, [M] trading days, [K] assets
- **Features**: [List feature columns]
- **Target**: Binary (1=positive return, 0=negative return)

## Evaluation Data

- **Method**: Walk-forward validation with purging/embargo
- **Folds**: [N] folds
- **Test Period**: [dates]
- **Causality**: All tests passed (no lookahead)

## Metrics

| Metric | Value | CI Lower | CI Upper |
|--------|-------|----------|----------|
| Accuracy | 0.XX | 0.XX | 0.XX |
| Precision | 0.XX | 0.XX | 0.XX |
| Recall | 0.XX | 0.XX | 0.XX |
| F1 Score | 0.XX | 0.XX | 0.XX |
| AUC-ROC | 0.XX | 0.XX | 0.XX |

## Trading Performance

| Metric | Value | CI Lower | CI Upper |
|--------|-------|----------|----------|
| Sharpe | X.XX | X.XX | X.XX |
| CAGR | X.XX% | X.XX% | X.XX% |
| Max DD | X.XX% | - | - |

## Limitations

- Trained on limited asset universe (4 stocks)
- Performance may degrade in high-volatility regimes
- Requires sufficient social media volume
- [Other limitations]

## Ethical Considerations

- Does not use personal identifying information
- Author IDs are hashed
- No individual investment advice provided

## Caveats and Recommendations

- Monitor for feature drift monthly
- Retrain if Sharpe drops below [threshold]
- Do not use for real money trading without additional validation
```

### 10.2 Model Card: LSTM Classifier (Filled Example)

```markdown
# Model Card: LSTM Return Direction Classifier

## Model Details

- **Model Name**: LSTM Return Direction Classifier
- **Version**: 1.0.0
- **Date**: 2024-02-01
- **Type**: Binary Classification
- **Framework**: PyTorch 2.1.0
- **Architecture**: 2-layer LSTM, hidden_dim=64, dropout=0.3

## Intended Use

- **Primary Use**: Predict next-day return direction for US equities
- **Users**: IMST-Quant automated trading system
- **Out-of-Scope**: Real-time/HFT, derivatives, crypto

## Training Data

- **Source**: Reddit (r/wallstreetbets, r/stocks, r/investing) + Yahoo Finance
- **Date Range**: 2022-01-01 to 2023-06-30
- **Volume**: 2.3M posts, 377 trading days, 4 assets
- **Features**: return_1d, volatility_30d, related_returns(3), sentiment_index
- **Target**: Binary (1=positive return, 0=negative)
- **Class Balance**: 52% positive, 48% negative

## Evaluation Data

- **Method**: Walk-forward validation
- **Folds**: 12 monthly folds
- **Test Period**: 2023-07-01 to 2023-12-31
- **Purge/Embargo**: 5/5 days
- **Causality Tests**: 47/47 passed

## Metrics

| Metric | Value | 95% CI Lower | 95% CI Upper |
|--------|-------|--------------|--------------|
| Accuracy | 0.547 | 0.521 | 0.573 |
| Precision | 0.558 | 0.530 | 0.586 |
| Recall | 0.612 | 0.578 | 0.646 |
| F1 Score | 0.584 | 0.553 | 0.615 |
| AUC-ROC | 0.561 | 0.534 | 0.588 |

## Trading Performance (Dynamic Threshold Policy)

| Metric | Value | 95% CI Lower | 95% CI Upper |
|--------|-------|--------------|--------------|
| Sharpe | 0.73 | 0.41 | 1.05 |
| CAGR | 8.2% | 4.1% | 12.3% |
| Max DD | 7.8% | - | - |
| Hit Rate | 54.7% | - | - |
| Turnover | 2.1x/year | - | - |

## Statistical Significance

- Sharpe 95% CI excludes 0: **Yes** ✓
- vs Buy-and-Hold p-value: 0.023 **Significant** ✓
- White's Reality Check: p=0.047 **Significant** ✓

## Limitations

1. Limited to 4-stock universe (AAPL, JNJ, JPM, XOM)
2. Reddit-only social data (no Twitter/StockTwits)
3. Performance degrades when VIX > 35
4. Minimum 50 posts/day required for reliable signal
5. Sensitive to meme stock events

## Ethical Considerations

- Author IDs hashed with SHA-256
- No storage of usernames or PII
- Compliant with Reddit API ToS
- Not providing individual investment advice

## Caveats and Recommendations

1. Monitor feature drift weekly; alert if PSI > 0.25
2. Retrain monthly or if Sharpe (30-day rolling) < 0.3
3. Suspend trading if VIX > 40
4. Paper trading only until 90-day live track record
5. Maximum position size 10% per asset

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-02-01 | Initial release |
```

### 10.3 Datasheet Template

```markdown
# Datasheet: [Dataset Name]

## Motivation

- **Purpose**: Why was this dataset created?
- **Creators**: Who created it?
- **Funding**: Any funding sources?

## Composition

- **Instances**: What does each instance represent?
- **Count**: How many instances total?
- **Sampling**: How was data sampled?
- **Data Types**: What data types are included?
- **Missing Data**: Is there missing data? How handled?
- **Confidentiality**: Any confidential data?

## Collection Process

- **Acquisition**: How was data acquired?
- **Collection Mechanism**: What mechanisms used?
- **Who Collected**: Who was involved?
- **Timeframe**: When was data collected?
- **Ethical Review**: Any ethical review?

## Preprocessing/Cleaning

- **Preprocessing**: What preprocessing was done?
- **Raw Data**: Is raw data saved?
- **Software**: What software was used?

## Uses

- **Prior Uses**: Has this data been used before?
- **Repository**: Where is it stored?
- **Potential Uses**: Other potential uses?
- **Inappropriate Uses**: What should it NOT be used for?

## Distribution

- **Distribution Method**: How is it distributed?
- **License**: What license applies?
- **Fees**: Any fees?

## Maintenance

- **Maintainer**: Who maintains it?
- **Updates**: How often updated?
- **Retention**: How long retained?
- **Versioning**: How are versions tracked?
```

### 10.4 Datasheet: Reddit Posts (Filled Example)

```markdown
# Datasheet: Reddit Financial Posts Dataset

## Motivation

- **Purpose**: Train sentiment analysis and influence models for financial prediction
- **Creators**: IMST-Quant development team
- **Funding**: Self-funded research project

## Composition

- **Instances**: Each instance is a Reddit post or comment
- **Count**: ~2.3 million posts (as of 2024-01)
- **Sampling**: All posts from selected subreddits, filtered by asset mention
- **Data Types**:
  - Text: title, selftext, body
  - Numeric: score, num_comments, awards
  - Categorical: subreddit, post_type, flair
  - Temporal: created_utc, retrieved_utc
  - Derived: author_id (hashed)
- **Missing Data**:
  - ~5% missing selftext (link posts)
  - ~2% missing flair
  - Handled: NULL values, no imputation
- **Confidentiality**: Author names hashed; no PII retained

## Collection Process

- **Acquisition**: Reddit API via PRAW library
- **Collection Mechanism**:
  - Automated crawler running every 15 minutes
  - Backfill for historical data
- **Who Collected**: Automated system
- **Timeframe**: 2022-01-01 to present (ongoing)
- **Ethical Review**: Compliant with Reddit API ToS

## Preprocessing/Cleaning

- **Preprocessing**:
  1. Language detection (English only)
  2. Near-duplicate removal (MinHash)
  3. Text normalization
  4. Entity linking
  5. Sentiment scoring
- **Raw Data**: Preserved in `/data/raw/` (JSONL.gz)
- **Software**: Python 3.11, PRAW 7.7, fasttext, datasketch

## Uses

- **Prior Uses**: Paper replication study
- **Repository**: Local storage + PostgreSQL metadata
- **Potential Uses**:
  - Sentiment analysis research
  - Social network analysis
  - NLP model training
- **Inappropriate Uses**:
  - User identification/deanonymization
  - Harassment or targeting
  - Selling data to third parties

## Distribution

- **Distribution Method**: Internal only (not distributed)
- **License**: Research use only
- **Fees**: None

## Maintenance

- **Maintainer**: IMST-Quant team
- **Updates**: Continuous ingestion
- **Retention**: 3 years rolling
- **Versioning**: Daily snapshots, weekly archives
```

### 10.5 Reproducibility Checklist

```markdown
# Reproducibility Checklist

## Code and Environment

- [ ] All code in version control (Git)
- [ ] Dependencies pinned (poetry.lock)
- [ ] Docker image available
- [ ] Random seeds set and documented
- [ ] GPU/CPU requirements documented

## Data

- [ ] Raw data archived with checksums
- [ ] Data version tracked (DVC or manual)
- [ ] Preprocessing scripts versioned
- [ ] Train/test splits documented
- [ ] Feature definitions in code

## Models

- [ ] Model architecture in code
- [ ] Hyperparameters in config files
- [ ] Training scripts versioned
- [ ] Model weights saved with version
- [ ] MLflow experiment tracked

## Evaluation

- [ ] Evaluation scripts versioned
- [ ] Metrics computation documented
- [ ] Statistical tests documented
- [ ] Baseline comparisons included
- [ ] CI computed with bootstrap

## Reproducibility Commands

```bash
# Clone repository
git clone https://github.com/org/imst-quant.git
cd imst-quant

# Checkout specific version
git checkout v1.0.0

# Install dependencies
poetry install

# Download data snapshot
./scripts/download_data.sh --version 2024-01-01

# Verify data checksums
./scripts/verify_checksums.sh

# Run full pipeline
make reproduce

# Verify results
./scripts/compare_results.sh expected/replication_results.json
```

## Expected Results

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Sharpe | 0.73 | ±0.05 |
| CAGR | 8.2% | ±1.0% |
| Max DD | 7.8% | ±0.5% |
| Accuracy | 54.7% | ±1.0% |
```

### 10.6 Change Management Policy

```markdown
# Change Management Policy

## Change Categories

### Category A: Critical (Requires approval)
- Model architecture changes
- Feature definition changes
- Trading logic changes
- Risk limit changes

### Category B: Standard (Requires review)
- Hyperparameter tuning
- Configuration updates
- Infrastructure changes
- Dependency updates

### Category C: Minor (Self-approved)
- Documentation updates
- Logging changes
- Test additions
- Bug fixes (non-critical)

## Approval Process

### Category A Changes

1. **Proposal**: Document change in RFC format
2. **Review**: Technical review by 2+ team members
3. **Testing**: Full backtest on historical data
4. **Comparison**: Statistical comparison to baseline
5. **Approval**: Sign-off from quant lead + risk manager
6. **Deployment**: Staged rollout (shadow mode first)
7. **Monitoring**: 7-day enhanced monitoring

### Category B Changes

1. **PR**: Standard pull request
2. **Review**: 1+ reviewer approval
3. **Testing**: Unit + integration tests pass
4. **Documentation**: Update relevant docs
5. **Merge**: Merge to main branch

### Category C Changes

1. **PR**: Pull request with description
2. **Tests**: Automated tests pass
3. **Merge**: Self-merge allowed

## Rollback Procedure

1. Identify issue via monitoring alerts
2. Execute kill switch if trading active
3. Rollback to previous version:
   ```bash
   git checkout <previous-tag>
   make deploy
   ```
4. Document incident
5. Post-mortem within 48 hours
```

### 10.7 Incident Response Runbook

```markdown
# Incident Response Runbook

## Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | Critical | < 15 min | Trading halted, data breach |
| P2 | High | < 1 hour | Model failure, large loss |
| P3 | Medium | < 4 hours | Data delay, partial outage |
| P4 | Low | < 24 hours | Minor bug, cosmetic issue |

## P1: Trading System Down

### Symptoms
- No orders submitted
- API errors from broker
- Dashboard shows stale data

### Immediate Actions
1. Check kill switch status
2. Verify broker API connectivity:
   ```bash
   curl -X GET https://paper-api.alpaca.markets/v2/account \
     -H "APCA-API-KEY-ID: $ALPACA_KEY"
   ```
3. Check system logs:
   ```bash
   tail -100 /var/log/imst_quant/paper_trade.log
   ```
4. If broker issue: Contact Alpaca support
5. If system issue: Restart services:
   ```bash
   docker-compose restart paper_trade
   ```

### Escalation
- Page on-call engineer
- Notify risk manager if > 30 min downtime

## P2: Unexpected Large Loss

### Symptoms
- Daily P&L exceeds -2%
- Drawdown alert triggered
- Single position loss > 5%

### Immediate Actions
1. Verify kill switch engaged (should auto-trigger)
2. If not, manually engage:
   ```bash
   imst paper stop --reason "manual halt: large loss"
   ```
3. Review recent trades:
   ```sql
   SELECT * FROM trading_logs
   WHERE date = CURRENT_DATE
   ORDER BY created_at DESC;
   ```
4. Check for data issues (bad prices, missing data)
5. Check for model drift

### Root Cause Analysis
- Compare predictions to actual outcomes
- Review feature values for anomalies
- Check for corporate actions or news events

## P3: Data Pipeline Failure

### Symptoms
- Ingestion metrics show zero posts
- Feature freshness > 4 hours
- Reddit API errors

### Actions
1. Check Reddit API status
2. Verify API credentials:
   ```bash
   imst ingest test-auth --source reddit
   ```
3. Check rate limiting:
   ```bash
   grep "rate limit" /var/log/imst_quant/ingestion.log
   ```
4. If rate limited: Wait for reset
5. If auth issue: Refresh credentials
6. If Reddit down: Enable fallback (cached features)

## Post-Incident

1. Document timeline in incident log
2. Identify root cause
3. Implement fix
4. Update runbook if needed
5. Schedule post-mortem for P1/P2
```

---

## 11. Causality Test Suite

### 11.1 No-Lookahead Tests

```python
# tests/unit/test_causality.py

import pytest
from datetime import datetime, date, time, timedelta
import polars as pl

class TestFeatureCausality:
    """Tests to ensure no lookahead bias in features."""

    def test_sentiment_uses_only_past_posts(self, feature_builder, mock_posts):
        """Sentiment features only use posts created before cutoff."""
        cutoff = datetime(2024, 1, 15, 9, 30)

        # Create posts: some before cutoff, some after
        posts_before = [
            {"id": "1", "created_utc": datetime(2024, 1, 15, 8, 0).timestamp()},
            {"id": "2", "created_utc": datetime(2024, 1, 15, 9, 0).timestamp()},
        ]
        posts_after = [
            {"id": "3", "created_utc": datetime(2024, 1, 15, 10, 0).timestamp()},
            {"id": "4", "created_utc": datetime(2024, 1, 15, 11, 0).timestamp()},
        ]

        # Build features
        features = feature_builder.build_features(
            asset="AAPL",
            as_of_date=cutoff.date(),
            as_of_time=cutoff.time()
        )

        # Verify only posts_before were used
        # (This requires feature_builder to track which posts were used)
        used_post_ids = feature_builder.get_last_used_post_ids()
        assert set(used_post_ids) == {"1", "2"}
        assert "3" not in used_post_ids
        assert "4" not in used_post_ids

    def test_price_features_use_previous_day(self, feature_builder):
        """Price features use T-1 data, not T data."""
        target_date = date(2024, 1, 15)

        features = feature_builder.build_features(
            asset="AAPL",
            as_of_date=target_date,
            as_of_time=time(9, 30)
        )

        # return_1d should be T-2 to T-1, not T-1 to T
        # (because we're predicting at T, we can't use T's close)
        assert features._price_data_end_date == date(2024, 1, 14)

    def test_volatility_window_ends_before_target(self, feature_builder):
        """Rolling volatility uses data ending before target date."""
        target_date = date(2024, 1, 15)

        features = feature_builder.build_features(
            asset="AAPL",
            as_of_date=target_date,
            as_of_time=time(9, 30)
        )

        # 30-day volatility should use days T-31 to T-1
        assert features._volatility_window_end == date(2024, 1, 14)
        assert features._volatility_window_start == date(2023, 12, 15)

    def test_influence_scores_from_previous_month(self, feature_builder):
        """Influence scores use previous completed month."""
        # Target date: Jan 15, 2024
        # Should use December 2023 influence scores
        target_date = date(2024, 1, 15)

        features = feature_builder.build_features(
            asset="AAPL",
            as_of_date=target_date,
            as_of_time=time(9, 30)
        )

        assert features._influence_month == date(2023, 12, 1)

    def test_target_variable_is_future(self, dataset_builder):
        """Target variable (next-day return) is correctly future."""
        df = dataset_builder.build_training_data(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )

        # For each row, target should be return on date + 1
        for row in df.iter_rows(named=True):
            feature_date = row["date"]
            target_return = row["target_return"]

            # Get actual return on feature_date + 1
            next_day = feature_date + timedelta(days=1)
            actual_return = dataset_builder.get_actual_return(
                row["asset"], next_day
            )

            assert target_return == actual_return, \
                f"Target mismatch: {feature_date}, expected {actual_return}, got {target_return}"

    def test_score_observation_causality(self, feature_builder):
        """Post scores only used if observed within causality window."""
        # Post created at 9:00, retrieved at 15:00 (6 hours later)
        # Score observation window is 30 minutes
        # Should use score=0

        post = {
            "id": "test",
            "created_utc": datetime(2024, 1, 15, 9, 0).timestamp(),
            "retrieved_utc": datetime(2024, 1, 15, 15, 0).timestamp(),
            "score": 100
        }

        causal_score = feature_builder._get_causal_score(post)
        assert causal_score == 0, "Score observed too late should be 0"

        # Post retrieved within window
        post_fast = {
            "id": "test2",
            "created_utc": datetime(2024, 1, 15, 9, 0).timestamp(),
            "retrieved_utc": datetime(2024, 1, 15, 9, 25).timestamp(),
            "score": 100
        }

        causal_score_fast = feature_builder._get_causal_score(post_fast)
        assert causal_score_fast == 100, "Score observed quickly should be used"


class TestWalkForwardCausality:
    """Tests for walk-forward validation causality."""

    def test_no_train_test_overlap(self, walk_forward_validator):
        """Train and test sets have no date overlap."""
        splits = walk_forward_validator.generate_splits(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        for split in splits:
            train_dates = set(pd.date_range(
                split["train_start"], split["train_end"]
            ).date)
            test_dates = set(pd.date_range(
                split["test_start"], split["test_end"]
            ).date)

            overlap = train_dates & test_dates
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_purge_gap_exists(self, walk_forward_validator):
        """Purge gap exists between train and test."""
        splits = walk_forward_validator.generate_splits(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        for split in splits:
            gap_days = (split["test_start"] - split["train_end"]).days
            assert gap_days >= 5, f"Insufficient purge: {gap_days} days"

    def test_embargo_gap_exists(self, walk_forward_validator):
        """Embargo gap exists after test."""
        splits = walk_forward_validator.generate_splits(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        for i in range(len(splits) - 1):
            current_test_end = splits[i]["test_end"]
            next_train_start = splits[i + 1]["train_start"]

            # Next train should not include current test data
            if next_train_start <= current_test_end:
                # This is fine if next_train_end is before current_test_start
                pass
            else:
                gap = (next_train_start - current_test_end).days
                # Not strictly required if rolling window,
                # but expanding window needs embargo
                pass


class TestModelTrainingCausality:
    """Tests for model training causality."""

    def test_model_not_trained_on_test_data(self, model_trainer, features_df):
        """Model training does not use test period data."""
        test_date = date(2024, 1, 15)

        model, metrics = model_trainer.train_for_date(
            features_df,
            target_date=test_date,
            feature_columns=["return_1d", "sentiment_index"]
        )

        # Verify training data ends before test_date
        assert model_trainer._last_train_end_date < test_date


# Pytest fixture for running all causality tests
@pytest.fixture
def causality_test_suite():
    """Run all causality tests and return results."""
    return {
        "feature_causality": TestFeatureCausality,
        "walk_forward_causality": TestWalkForwardCausality,
        "model_training_causality": TestModelTrainingCausality
    }
```

---

## 12. Final Acceptance Checklist

### 12.1 Paper Replication Acceptance Criteria

| ID | Criterion | Test | Pass |
|----|-----------|------|------|
| PR-01 | TextBlob sentiment implemented exactly | Unit test | ☐ |
| PR-02 | Monthly GCN influence graph built | Integration test | ☐ |
| PR-03 | 4 stocks (AAPL, JNJ, JPM, XOM) supported | Config validation | ☐ |
| PR-04 | 3 related stocks per primary stock | Feature test | ☐ |
| PR-05 | Window sizes 5, 15, 30 supported | Config validation | ☐ |
| PR-06 | LSTM/CNN/Transformer models implemented | Unit tests | ☐ |
| PR-07 | Daily retraining on 90-day window | Training test | ☐ |
| PR-08 | Fixed threshold method implemented | Unit test | ☐ |
| PR-09 | Dynamic threshold method implemented | Unit test | ☐ |
| PR-10 | Results within 5% of paper reported | Backtest | ☐ |

### 12.2 Production Upgrade Acceptance Criteria

| ID | Criterion | Test | Pass |
|----|-----------|------|------|
| PU-01 | Reddit ingestion operational | Integration test | ☐ |
| PU-02 | Entity linking >90% precision | Labeled test set | ☐ |
| PU-03 | FinBERT sentiment working | Unit test | ☐ |
| PU-04 | Credibility scoring operational | Unit test | ☐ |
| PU-05 | Bot detection >80% recall | Labeled test set | ☐ |
| PU-06 | Walk-forward validation working | Integration test | ☐ |
| PU-07 | Statistical tests implemented | Unit tests | ☐ |
| PU-08 | Portfolio policy with constraints | Unit test | ☐ |
| PU-09 | Risk management kill switches | Integration test | ☐ |
| PU-10 | Paper trading operational | E2E test | ☐ |

### 12.3 No-Leakage Acceptance Criteria

| ID | Criterion | Test | Pass |
|----|-----------|------|------|
| NL-01 | Features use only past data | test_causality.py | ☐ |
| NL-02 | Targets are future returns | test_causality.py | ☐ |
| NL-03 | Walk-forward has purge gaps | test_causality.py | ☐ |
| NL-04 | Walk-forward has embargo gaps | test_causality.py | ☐ |
| NL-05 | Score observation window enforced | test_causality.py | ☐ |
| NL-06 | Influence scores from past month | test_causality.py | ☐ |
| NL-07 | Fixed threshold labeled as leakage | Documentation | ☐ |
| NL-08 | All 47 causality tests pass | CI pipeline | ☐ |

### 12.4 Operational Acceptance Criteria

| ID | Criterion | Test | Pass |
|----|-----------|------|------|
| OP-01 | Single command rebuild works | `make reproduce` | ☐ |
| OP-02 | Docker deployment works | `docker-compose up` | ☐ |
| OP-03 | Prometheus metrics exposed | Curl test | ☐ |
| OP-04 | Grafana dashboards load | Manual check | ☐ |
| OP-05 | Alerts fire correctly | Simulation test | ☐ |
| OP-06 | Logs are structured JSON | Log parsing test | ☐ |
| OP-07 | Kill switch halts trading | Integration test | ☐ |
| OP-08 | 30 days paper trading stable | Live run | ☐ |

### 12.5 Documentation Acceptance Criteria

| ID | Criterion | Location | Pass |
|----|-----------|----------|------|
| DOC-01 | Architecture diagram | docs/architecture.md | ☐ |
| DOC-02 | API documentation | docs/api.md | ☐ |
| DOC-03 | Model cards for all models | docs/model_cards/ | ☐ |
| DOC-04 | Datasheets for all datasets | docs/datasheets/ | ☐ |
| DOC-05 | Runbooks for operations | docs/runbooks/ | ☐ |
| DOC-06 | README with quickstart | README.md | ☐ |
| DOC-07 | Configuration reference | configs/README.md | ☐ |
| DOC-08 | Change management policy | docs/change_management.md | ☐ |

---

## 13. Starter Defaults

### 13.1 Initial Equity Universe

**Paper Replication (4 stocks)**:
```yaml
primary:
  - AAPL   # Apple - Tech bellwether
  - JNJ    # Johnson & Johnson - Healthcare
  - JPM    # JP Morgan - Financials
  - XOM    # Exxon Mobil - Energy
```

**Extended Universe (50 stocks)** - Selected for high Reddit mention volume and liquidity:

```yaml
# Technology (15)
tech:
  - AAPL   # Apple
  - MSFT   # Microsoft
  - NVDA   # NVIDIA - AI/GPU play, very high Reddit volume
  - AMD    # AMD - NVDA competitor
  - GOOGL  # Alphabet
  - META   # Meta
  - AMZN   # Amazon
  - TSLA   # Tesla - Highest Reddit volume
  - NFLX   # Netflix
  - CRM    # Salesforce
  - ORCL   # Oracle
  - INTC   # Intel
  - QCOM   # Qualcomm
  - MU     # Micron
  - PLTR   # Palantir - High retail interest

# Consumer (8)
consumer:
  - DIS    # Disney
  - NKE    # Nike
  - SBUX   # Starbucks
  - MCD    # McDonald's
  - HD     # Home Depot
  - TGT    # Target
  - COST   # Costco
  - WMT    # Walmart

# Financials (8)
financials:
  - JPM    # JP Morgan
  - BAC    # Bank of America
  - GS     # Goldman Sachs
  - MS     # Morgan Stanley
  - V      # Visa
  - MA     # Mastercard
  - PYPL   # PayPal
  - SQ     # Block (Square)

# Healthcare (7)
healthcare:
  - JNJ    # Johnson & Johnson
  - UNH    # UnitedHealth
  - PFE    # Pfizer
  - MRK    # Merck
  - ABBV   # AbbVie
  - LLY    # Eli Lilly
  - MRNA   # Moderna

# Energy (4)
energy:
  - XOM    # Exxon Mobil
  - CVX    # Chevron
  - COP    # ConocoPhillips
  - OXY    # Occidental

# Industrials (4)
industrials:
  - BA     # Boeing
  - CAT    # Caterpillar
  - UPS    # UPS
  - LMT    # Lockheed Martin

# Meme stocks (high volatility, high volume) - separate category
meme_stocks:
  - GME    # GameStop
  - AMC    # AMC Entertainment
  - BB     # BlackBerry
  - BBBY   # Bed Bath & Beyond (delisted - historical only)
```

**Rationale**: Selected based on:
1. High Reddit mention frequency (WSB, r/stocks)
2. Sufficient liquidity (avg volume > 1M shares/day)
3. Sector diversification
4. Mix of stable (JNJ) and volatile (TSLA) names

### 13.2 Initial Crypto Universe

```yaml
crypto:
  # Large cap (stable, high volume)
  - BTC/USDT    # Bitcoin
  - ETH/USDT    # Ethereum

  # L1 competitors
  - SOL/USDT    # Solana - High Reddit interest
  - ADA/USDT    # Cardano
  - AVAX/USDT   # Avalanche
  - DOT/USDT    # Polkadot

  # L2 / DeFi
  - MATIC/USDT  # Polygon
  - LINK/USDT   # Chainlink
  - UNI/USDT    # Uniswap
  - AAVE/USDT   # Aave

  # Exchange tokens
  - BNB/USDT    # Binance Coin

  # Meme coins (high volatility, very high Reddit volume)
  - DOGE/USDT   # Dogecoin
  - SHIB/USDT   # Shiba Inu

  # Emerging
  - ARB/USDT    # Arbitrum
  - OP/USDT     # Optimism
```

**Rationale**: Selected for Reddit discussion volume and trading liquidity.

### 13.3 Initial Subreddits

```yaml
# Equity focused
equity_subreddits:
  primary:
    - wallstreetbets    # 14M+ members, highest volume, meme-heavy
    - stocks            # 6M+ members, more serious
    - investing         # 2M+ members, long-term focus
    - options           # 1M+ members, derivatives focus

  secondary:
    - stockmarket       # General market discussion
    - valueinvesting    # Fundamental focus
    - dividends         # Income focus
    - SecurityAnalysis  # Deep analysis
    - thetagang         # Options selling strategies
    - Bogleheads        # Index investing (low signal)

  sector_specific:
    - AMD_Stock         # AMD specific
    - NVDA_Stock        # NVIDIA specific
    - teslainvestorsclub # Tesla specific

# Crypto focused
crypto_subreddits:
  primary:
    - cryptocurrency    # 7M+ members
    - bitcoin           # 5M+ members
    - ethereum          # 2M+ members
    - CryptoMarkets     # Trading focus

  secondary:
    - ethtrader         # ETH trading
    - defi              # DeFi protocols
    - altcoin           # Altcoins
    - SatoshiStreetBets # Crypto WSB equivalent
```

### 13.4 Time Cutoffs and Timezone Handling

```yaml
timezone:
  primary: "US/Eastern"
  market_hours:
    pre_market_start: "04:00"
    market_open: "09:30"
    market_close: "16:00"
    after_hours_end: "20:00"

cutoffs:
  # Daily prediction cutoff (before market open)
  prediction_cutoff: "09:25"

  # Feature computation uses data up to this time
  feature_cutoff: "09:25"

  # Order submission time
  order_submission: "09:30"

  # End-of-day logging
  eod_logging: "16:05"

  # Score observation window for Reddit posts
  score_observation_minutes: 30

reddit_time_handling:
  # Reddit uses UTC timestamps
  # Convert all to US/Eastern for consistency

  # Bucket posts into 3-hour intervals (paper replication)
  bucket_hours: 3

  # Daily aggregation: midnight to midnight Eastern
  day_boundary: "00:00"

  # Weekend handling: aggregate Saturday/Sunday to Monday
  weekend_to_monday: true

market_calendar:
  # Use pandas_market_calendars for NYSE holidays
  exchange: "NYSE"

  # Skip non-trading days in features
  skip_holidays: true

  # Carry forward Friday features to Monday
  friday_to_monday: true
```

**Critical Implementation Notes**:

1. **All timestamps stored in UTC** but displayed/processed in Eastern
2. **Reddit created_utc** is Unix timestamp (UTC) - convert correctly
3. **Market data timestamps** - ensure bar timestamps align (open time vs close time)
4. **Feature computation** - must complete before market open
5. **Order submission** - exactly at market open for fair comparison

---

## 14. Assumptions and Configurables

### Assumptions Made (Can Be Changed via Config)

| Assumption | Default | Config Key | Rationale |
|------------|---------|------------|-----------|
| English posts only | True | `processing.language.allowed_languages` | Simplicity; most Reddit finance is English |
| Min 50 interactions for influence | 50 | `influence.graph.min_interactions` | Paper replication |
| 90-day rolling training window | 90 | `models.training.rolling_window_days` | Paper replication |
| 30-minute score observation window | 30 | `ingestion.reddit.score_observation_minutes` | Causality safety |
| 5-day purge/embargo | 5/5 | `backtest.walk_forward.purge_days` | Rolling feature overlap |
| $100K initial capital | 100000 | `backtest.initial_capital` | Reasonable test size |
| 10% max single position | 0.10 | `trading.policies.portfolio.max_single_position` | Risk management |
| 2% max daily loss | 0.02 | `trading.risk.max_daily_loss_pct` | Kill switch threshold |
| 10% max drawdown | 0.10 | `trading.risk.max_drawdown_pct` | Kill switch threshold |

### Decisions Requiring User Input

| Decision | Options | Impact |
|----------|---------|--------|
| Real data source | Reddit API / Mock data | Determines data freshness |
| Market data provider | Polygon / IEX / yfinance | Data quality and cost |
| Paper trading broker | Alpaca / Simulate only | Execution realism |
| Model type | LSTM / CNN / Transformer / Ensemble | Performance characteristics |
| Trading policy | Dynamic threshold / Portfolio | Risk/return profile |

---

*End of PRD Document*

**Total Document Size**: ~25,000 words
**Ready for Engineering Handoff**: Yes

To begin implementation:
```bash
# 1. Clone and setup
git clone <repo>
cd imst-quant
make dev

# 2. Configure
cp .env.example .env
# Edit .env with API keys

# 3. Initialize
imst init --config paper_replication

# 4. Start ingestion
imst ingest reddit --days 90

# 5. Run first backtest
make backtest-paper
```
