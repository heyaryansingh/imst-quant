# IMST-Quant: Product Requirements Document

**Version**: 1.0.0
**Date**: 2026-02-17
**Status**: Draft

---

## 1. Executive Summary

IMST-Quant (Influence-aware Multi-Source Sentiment Trading) is a production-grade algorithmic trading system that combines social media sentiment analysis with graph neural networks to forecast asset returns. The project has two modes:

**Mode A (Paper Replication)**: Faithfully reproduce the methodology from "GNN-based social media sentiment analysis for stock market forecasting" — monthly GCN influence graphs, TextBlob sentiment, LSTM/CNN/Transformer forecasting, and threshold-based trading on 4 stocks (AAPL, JNJ, JPM, XOM).

**Mode B (Production Upgrade)**: Extend to multi-source ingestion (Reddit primary), modern NLP (FinBERT), credibility filtering, walk-forward evaluation, and live paper trading with institutional-grade audit trails.

**Key Differentiators**:
- Strict temporal causality enforcement with automated testing
- Influence-weighted sentiment capturing social network dynamics
- Credibility filtering against bot/manipulation attacks
- Complete reproducibility from raw data to backtest results
- Production paper trading with kill switches and monitoring

**Success Criteria**:
1. Replicate paper's reported accuracy within 5% margin
2. Pass all causality/no-lookahead tests
3. Achieve positive risk-adjusted returns in walk-forward backtest
4. Run 30 consecutive days of paper trading without incidents

**Timeline**: 12 weeks to full paper trading capability

---

## 2. PRD: Goals and Non-Goals

### 2.1 Goals

| ID | Goal | Success Metric |
|----|------|----------------|
| G1 | Replicate paper methodology exactly | Match reported metrics ±5% |
| G2 | Zero lookahead bias in all features | 100% causality tests pass |
| G3 | Production-ready paper trading | 30 days stable operation |
| G4 | Full audit trail | Any prediction traceable to inputs |
| G5 | Realistic backtesting | Costs, slippage, survivorship handled |
| G6 | Multi-source architecture | Add new sources without core changes |
| G7 | Manipulation resistance | Detect and downweight bot activity |

### 2.2 Non-Goals

| ID | Non-Goal | Reason |
|----|----------|--------|
| NG1 | Real money trading | Risk management; paper trading only |
| NG2 | High-frequency trading | Daily predictions sufficient for thesis |
| NG3 | Options/derivatives | Complexity; equities + crypto spot only |
| NG4 | Mobile/web UI | CLI + API sufficient for v1 |
| NG5 | Fundamental data | Focus on sentiment signal |
| NG6 | Multi-language support | English only for v1 |

---

## 3. Personas and User Stories

### 3.1 Personas

**P1: Quant Researcher**
- Wants to validate paper findings and run experiments
- Needs: reproducibility, ablation framework, statistical tests

**P2: ML Engineer**
- Wants to train and deploy models reliably
- Needs: clear interfaces, monitoring, versioning

**P3: Risk Manager**
- Wants to understand and control trading risk
- Needs: audit trails, kill switches, exposure reports

**P4: Compliance Officer**
- Wants to ensure regulatory and ToS compliance
- Needs: data lineage, retention policies, access controls

### 3.2 User Stories

```
US-01: As a quant researcher, I want to run the paper's exact methodology
       so that I can validate their reported results.

US-02: As a quant researcher, I want to run ablation studies
       so that I can understand which components contribute to performance.

US-03: As an ML engineer, I want a single command to rebuild everything
       so that results are reproducible.

US-04: As an ML engineer, I want automated tests for temporal causality
       so that I catch lookahead bugs immediately.

US-05: As a risk manager, I want automatic position liquidation on drawdown
       so that losses are bounded.

US-06: As a risk manager, I want daily attribution reports
       so that I understand P&L drivers.

US-07: As a compliance officer, I want hashed author IDs only
       so that we don't store PII.

US-08: As a compliance officer, I want documented data retention policies
       so that we comply with platform ToS.
```

---

## 4. Functional Requirements

### 4.1 Data Ingestion

#### FR-ING-01: Reddit Ingestion

**Description**: Ingest submissions and comments from configured subreddits.

**Implementation**:
```python
# Required fields per post/comment
class RedditPost(BaseModel):
    id: str                      # Reddit's base36 ID
    author_id: str               # Hashed author name (SHA256)
    author_name_hash: str        # For dedup only, not stored long-term
    created_utc: int             # Unix timestamp of creation
    retrieved_utc: int           # Unix timestamp of retrieval (CRITICAL)
    subreddit: str               # Subreddit name
    post_type: Literal["submission", "comment"]
    title: Optional[str]         # Submissions only
    selftext: Optional[str]      # Submissions only
    body: Optional[str]          # Comments only
    score: int                   # Score AT RETRIEVAL TIME
    upvote_ratio: Optional[float]  # Submissions only
    num_comments: Optional[int]  # Submissions only
    permalink: str
    parent_id: Optional[str]     # Comments only
    link_id: Optional[str]       # Comments only
    distinguished: Optional[str]
    edited: bool
    awards: int                  # Total award count
    flair: Optional[str]
    url: Optional[str]           # External links
```

**Subreddits (Default)**:
```yaml
equity_subreddits:
  - wallstreetbets      # High volume, retail sentiment
  - stocks              # General stock discussion
  - investing           # Long-term focus
  - options             # Derivatives discussion
  - stockmarket         # General market
  - valueinvesting      # Fundamental focus
  - dividends           # Income focus
  - SecurityAnalysis    # Deep analysis

crypto_subreddits:
  - cryptocurrency      # General crypto
  - bitcoin             # BTC specific
  - ethereum            # ETH specific
  - CryptoMarkets       # Trading focus
  - ethtrader           # ETH trading
  - defi                # DeFi protocols
  - altcoin             # Alt discussion
```

**Crawl Schedule**:
- New posts: Every 15 minutes
- Backfill missing: Every 6 hours
- Full refresh: Weekly

**Rate Limit Handling**:
```python
REDDIT_RATE_LIMIT = 100  # requests per minute
BACKOFF_BASE = 60        # seconds
MAX_RETRIES = 5

async def fetch_with_backoff(endpoint, retries=0):
    try:
        return await reddit.request(endpoint)
    except RateLimitException as e:
        if retries >= MAX_RETRIES:
            raise
        wait = BACKOFF_BASE * (2 ** retries)
        await asyncio.sleep(wait)
        return await fetch_with_backoff(endpoint, retries + 1)
```

**CRITICAL - Score Causality**:
```python
# Score evolves over time. For features, we have two options:

# Option A (Conservative): Use score=0 for all posts
# Rationale: Any non-zero score observed at retrieval could reflect
#            future engagement that happened after the post was created

# Option B (Time-bounded): Only use score if retrieved within T minutes
SCORE_OBSERVATION_WINDOW_MINUTES = 30

def get_causal_score(post: RedditPost) -> int:
    """Return score only if observed within causality window."""
    observation_delay = post.retrieved_utc - post.created_utc
    if observation_delay <= SCORE_OBSERVATION_WINDOW_MINUTES * 60:
        return post.score
    return 0  # Default to zero if observed too late
```

#### FR-ING-02: Market Data Ingestion

**Equity Data**:
```python
class OHLCVBar(BaseModel):
    symbol: str
    timestamp: datetime        # Bar open time, timezone-aware
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal    # Split/dividend adjusted
    dividend: Decimal          # Dividend on this date
    split_factor: Decimal      # Split ratio (1.0 if none)
    source: str                # polygon, iex, yfinance

class CorporateAction(BaseModel):
    symbol: str
    ex_date: date
    action_type: Literal["split", "dividend", "spinoff", "merger"]
    factor: Decimal            # Split ratio or dividend amount
    description: str
```

**Data Sources Priority**:
1. Polygon.io (if API key exists) - Most reliable
2. IEX Cloud (if API key exists) - Good alternative
3. yfinance (fallback) - Free but less reliable

**Crypto Data**:
```python
class CryptoOHLCV(BaseModel):
    symbol: str                # e.g., BTC/USDT
    exchange: str              # e.g., binance
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal            # Base currency volume
    quote_volume: Decimal      # Quote currency volume
```

**CCXT Configuration**:
```python
CRYPTO_EXCHANGES = ["binance", "coinbase"]  # Primary exchanges
CRYPTO_TIMEFRAME = "1d"                      # Daily bars
```

#### FR-ING-03: Data Layer Architecture

**Storage Layers**:
```
/data/
├── raw/                    # Immutable raw JSON
│   ├── reddit/
│   │   └── YYYY/MM/DD/
│   │       └── {subreddit}_{hour}.jsonl.gz
│   ├── market/
│   │   └── YYYY/MM/DD/
│   │       └── {source}_{symbol}.json
│   └── crypto/
│       └── YYYY/MM/DD/
│           └── {exchange}_{symbol}.json
│
├── bronze/                 # Parsed, typed, deduplicated
│   ├── reddit/
│   │   └── posts.parquet   # Partitioned by date
│   ├── market/
│   │   └── ohlcv.parquet
│   └── crypto/
│       └── ohlcv.parquet
│
├── silver/                 # Enriched (entity links, sentiment)
│   ├── posts_enriched.parquet
│   ├── author_profiles.parquet
│   └── entity_links.parquet
│
└── gold/                   # Features (as-of correct)
    ├── daily_features.parquet
    ├── influence_scores.parquet
    └── sentiment_aggregates.parquet
```

### 4.2 Entity Linking

#### FR-ENT-01: Candidate Generation

**Cashtag Extraction**:
```python
import re

CASHTAG_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

def extract_cashtags(text: str) -> List[str]:
    """Extract potential ticker symbols from cashtags."""
    return CASHTAG_PATTERN.findall(text.upper())
```

**Ticker Dictionary**:
```python
# Load from NYSE/NASDAQ listings + manual additions
TICKER_DICT = {
    # Direct tickers
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology"},
    # ... loaded from CSV
}

# Company name aliases
COMPANY_ALIASES = {
    "apple": ["AAPL"],
    "iphone": ["AAPL"],
    "tim cook": ["AAPL"],
    "nvidia": ["NVDA"],
    "jensen": ["NVDA"],
    "hopper": ["NVDA"],
    "cuda": ["NVDA"],
    "microsoft": ["MSFT"],
    "azure": ["MSFT"],
    "satya": ["MSFT"],
    "tesla": ["TSLA"],
    "elon": ["TSLA"],  # Note: Also SpaceX context
    "the mouse": ["DIS"],
    "mouse house": ["DIS"],
    "mag 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "magnificent seven": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
}

# Crypto aliases
CRYPTO_ALIASES = {
    "bitcoin": ["BTC"],
    "btc": ["BTC"],
    "satoshi": ["BTC"],
    "ethereum": ["ETH"],
    "eth": ["ETH"],
    "vitalik": ["ETH"],
    "solana": ["SOL"],
    "sol": ["SOL"],
}
```

#### FR-ENT-02: Disambiguation

**Context-Based Disambiguation**:
```python
from sentence_transformers import SentenceTransformer

class EntityDisambiguator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.asset_embeddings = self._precompute_asset_embeddings()

    def _precompute_asset_embeddings(self) -> Dict[str, np.ndarray]:
        """Precompute embeddings for asset descriptions."""
        embeddings = {}
        for ticker, info in TICKER_DICT.items():
            desc = f"{ticker} {info['name']} {info['sector']}"
            embeddings[ticker] = self.model.encode(desc)
        return embeddings

    def disambiguate(
        self,
        text: str,
        candidates: List[str],
        subreddit: str,
        confidence_threshold: float = 0.7
    ) -> List[EntityLink]:
        """Disambiguate candidates using context."""
        text_embedding = self.model.encode(text)

        links = []
        for candidate in candidates:
            if candidate not in self.asset_embeddings:
                continue

            # Cosine similarity
            sim = np.dot(text_embedding, self.asset_embeddings[candidate])
            sim /= (np.linalg.norm(text_embedding) *
                    np.linalg.norm(self.asset_embeddings[candidate]))

            # Subreddit prior boost
            if self._subreddit_matches(subreddit, candidate):
                sim += 0.1

            if sim >= confidence_threshold:
                links.append(EntityLink(
                    asset_id=candidate,
                    confidence=float(sim),
                    method="embedding_similarity",
                    matched_span=candidate
                ))

        return links

    def _subreddit_matches(self, subreddit: str, ticker: str) -> bool:
        """Check if subreddit context boosts this ticker."""
        crypto_subs = {"cryptocurrency", "bitcoin", "ethereum", "cryptomarkets"}
        if subreddit.lower() in crypto_subs:
            return ticker in CRYPTO_ALIASES.values()
        return True

class EntityLink(BaseModel):
    asset_id: str
    confidence: float
    method: str
    matched_span: str
```

### 4.3 Text Processing

#### FR-TXT-01: Language Detection

```python
import fasttext

class LanguageDetector:
    def __init__(self, model_path: str = "lid.176.ftz"):
        self.model = fasttext.load_model(model_path)
        self.threshold = 0.8

    def is_english(self, text: str) -> Tuple[bool, float]:
        """Check if text is English with confidence."""
        predictions = self.model.predict(text.replace('\n', ' '))
        lang = predictions[0][0].replace('__label__', '')
        conf = predictions[1][0]
        return (lang == 'en' and conf >= self.threshold), conf
```

#### FR-TXT-02: Deduplication

```python
from datasketch import MinHash, MinHashLSH

class Deduplicator:
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_hashes = {}

    def get_minhash(self, text: str) -> MinHash:
        """Compute MinHash for text."""
        m = MinHash(num_perm=self.num_perm)
        for word in text.lower().split():
            m.update(word.encode('utf8'))
        return m

    def is_duplicate(self, post_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text is near-duplicate of existing post."""
        mh = self.get_minhash(text)

        # Query LSH for similar documents
        result = self.lsh.query(mh)
        if result:
            return True, result[0]  # Return first duplicate ID

        # Insert new document
        self.lsh.insert(post_id, mh)
        return False, None
```

#### FR-TXT-03: Text Normalization

```python
import re
from typing import NamedTuple

class NormalizedText(NamedTuple):
    raw: str              # Original text
    cleaned: str          # Normalized version
    urls_removed: int     # Count of URLs removed
    emojis: List[str]     # Extracted emojis

class TextNormalizer:
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    CASHTAG_PATTERN = re.compile(r'\$([a-zA-Z]{1,5})')

    def normalize(self, text: str) -> NormalizedText:
        """Normalize text while preserving raw version."""
        raw = text

        # Extract emojis before removal
        emojis = self._extract_emojis(text)

        # Count and remove URLs
        urls = self.URL_PATTERN.findall(text)
        cleaned = self.URL_PATTERN.sub(' [URL] ', text)

        # Standardize cashtags to uppercase
        cleaned = self.CASHTAG_PATTERN.sub(
            lambda m: f"${m.group(1).upper()}", cleaned
        )

        # Collapse whitespace
        cleaned = self.WHITESPACE_PATTERN.sub(' ', cleaned).strip()

        return NormalizedText(
            raw=raw,
            cleaned=cleaned,
            urls_removed=len(urls),
            emojis=emojis
        )

    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emoji characters from text."""
        import emoji
        return [c for c in text if c in emoji.EMOJI_DATA]
```

---

*Continued in next section...*
