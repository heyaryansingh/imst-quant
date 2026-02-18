# Phase 1: Project Setup & Data Ingestion Infrastructure - Research

**Researched:** 2026-02-17
**Domain:** Data ingestion pipelines (Reddit, Market Data, Parquet storage)
**Confidence:** HIGH

## Summary

Phase 1 establishes the project foundation and data ingestion infrastructure for a GNN-based sentiment analysis trading system. The research covers four critical areas: Reddit data collection via PRAW, equity market data via yfinance, cryptocurrency data via CCXT, and bronze/silver/gold medallion architecture with Parquet storage.

The standard approach uses PRAW 7.7+ for Reddit API access with automatic rate limiting, yfinance for equity OHLCV data, CCXT for cryptocurrency OHLCV data, and Polars with PyArrow for high-performance Parquet I/O. All raw data must include retrieval timestamps (not just content timestamps) to maintain causality guarantees required for the trading system.

**Primary recommendation:** Use the src/ layout project structure with Pydantic Settings for configuration, implement checkpoint-based incremental crawling for Reddit (since PRAW lacks native timestamp filtering), and store raw JSON with both `created_utc` (content time) and `retrieved_at` (ingestion time) fields.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PRAW | 7.7.1+ | Reddit API access | Official wrapper, automatic rate limiting, follows Reddit API rules |
| yfinance | 0.2.40+ | Equity OHLCV data | Most popular Yahoo Finance wrapper, 21k+ GitHub stars |
| CCXT | 4.4+ | Crypto OHLCV data | Unified API for 100+ exchanges, actively maintained |
| Polars | 1.0+ | DataFrame operations | 5-50x faster than pandas, native Parquet support |
| PyArrow | 17.0+ | Parquet I/O | Industry standard, Hive partitioning support |
| Pydantic Settings | 2.13+ | Configuration | Type-safe config from .env files, validation on load |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dotenv | 1.0+ | Environment loading | Backup for pydantic-settings |
| structlog | 24.0+ | Structured logging | JSON logs for observability |
| tenacity | 8.0+ | Retry logic | API call resilience with exponential backoff |
| orjson | 3.10+ | Fast JSON parsing | CCXT performance boost (optional) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Polars | Pandas | Pandas is more familiar but 10-50x slower for large datasets |
| yfinance | Alpha Vantage | Alpha Vantage has official API but restrictive free tier |
| CCXT | Direct exchange APIs | Direct APIs are faster but require per-exchange implementation |
| Pydantic Settings | python-decouple | Decouple is simpler but lacks type validation |

**Installation:**
```bash
pip install praw yfinance ccxt polars pyarrow pydantic-settings structlog tenacity orjson
```

## Architecture Patterns

### Recommended Project Structure
```
project/
├── src/
│   └── imst_quant/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py      # Pydantic Settings classes
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── reddit.py        # Reddit ingestion
│       │   ├── market.py        # yfinance equity ingestion
│       │   └── crypto.py        # CCXT crypto ingestion
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── raw.py           # Raw JSON storage
│       │   └── bronze.py        # Bronze parquet layer
│       └── utils/
│           ├── __init__.py
│           └── logging.py       # Structured logging setup
├── data/
│   ├── raw/                     # Raw JSON files
│   │   ├── reddit/
│   │   └── market/
│   └── bronze/                  # Bronze parquet files
│       ├── reddit/
│       └── market/
├── tests/
│   ├── unit/
│   └── integration/
├── config/
│   └── subreddits.yaml          # Subreddit configuration
├── .env                         # Local environment variables
├── .env.example                 # Template for .env
└── pyproject.toml
```

### Pattern 1: Causality-Preserving Raw Storage

**What:** Store both content timestamp and retrieval timestamp in every raw record.

**When to use:** Always for financial data to prevent lookahead bias.

**Example:**
```python
# Source: Derived from causality research best practices
import json
from datetime import datetime, timezone

def store_reddit_post(submission, output_dir: Path):
    """Store Reddit submission with causality timestamps."""
    record = {
        # Content timestamp (when post was created)
        "created_utc": submission.created_utc,
        # Retrieval timestamp (when we fetched it) - CRITICAL for causality
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        # Original data
        "id": submission.id,
        "title": submission.title,
        "selftext": submission.selftext,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "subreddit": submission.subreddit.display_name,
        "author": str(submission.author) if submission.author else "[deleted]",
        # Additional metadata
        "url": submission.url,
        "permalink": submission.permalink,
    }

    # Store with date-based path for organization
    date_str = datetime.fromtimestamp(submission.created_utc, timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / date_str / f"{submission.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(record, f)
```

### Pattern 2: Checkpoint-Based Incremental Crawling

**What:** Track last processed timestamp to enable resume after interruption.

**When to use:** Reddit crawling (PRAW lacks native timestamp filtering).

**Example:**
```python
# Source: Derived from PRAW documentation and incremental crawling research
import json
from pathlib import Path

class CheckpointManager:
    """Manage crawling checkpoints for incremental ingestion."""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self._checkpoints = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}

    def save(self):
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self._checkpoints, f, indent=2)

    def get_last_timestamp(self, subreddit: str) -> float:
        """Get last processed timestamp for subreddit."""
        return self._checkpoints.get(subreddit, {}).get("last_created_utc", 0)

    def update(self, subreddit: str, created_utc: float):
        """Update checkpoint for subreddit."""
        if subreddit not in self._checkpoints:
            self._checkpoints[subreddit] = {}
        self._checkpoints[subreddit]["last_created_utc"] = created_utc
        self._checkpoints[subreddit]["updated_at"] = datetime.now(timezone.utc).isoformat()
```

### Pattern 3: Medallion Architecture (Bronze Layer)

**What:** Raw data normalized to Parquet with minimal transformation.

**When to use:** After raw JSON ingestion, before any processing.

**Example:**
```python
# Source: Medallion architecture best practices
import polars as pl
from pathlib import Path

def raw_to_bronze_reddit(raw_dir: Path, bronze_dir: Path, date: str):
    """Convert raw Reddit JSON to bronze Parquet."""
    json_files = list((raw_dir / date).glob("*.json"))

    if not json_files:
        return

    # Read all JSON files for the date
    records = []
    for f in json_files:
        with open(f) as fp:
            records.append(json.load(fp))

    # Convert to Polars DataFrame
    df = pl.DataFrame(records)

    # Ensure consistent schema (bronze = raw + schema)
    df = df.with_columns([
        pl.col("created_utc").cast(pl.Float64),
        pl.col("score").cast(pl.Int64),
        pl.col("num_comments").cast(pl.Int64),
    ])

    # Write partitioned by date
    output_path = bronze_dir / f"date={date}" / "data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(
        output_path,
        compression="zstd",  # Best compression/speed tradeoff
    )
```

### Pattern 4: Configuration with Pydantic Settings

**What:** Type-safe configuration loaded from environment and .env files.

**When to use:** All configuration (API keys, paths, parameters).

**Example:**
```python
# Source: Pydantic Settings documentation
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from pathlib import Path

class RedditSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    client_id: str
    client_secret: SecretStr
    user_agent: str = "IMST-Quant/1.0"

class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=".env")

    raw_dir: Path = Path("data/raw")
    bronze_dir: Path = Path("data/bronze")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    reddit: RedditSettings = RedditSettings()
    data: DataSettings = DataSettings()

    # Asset universes
    equity_tickers: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    crypto_pairs: list[str] = ["BTC/USDT", "ETH/USDT"]
```

### Anti-Patterns to Avoid

- **Hardcoded paths/credentials:** Use Pydantic Settings with .env files, never hardcode API keys or paths
- **Missing retrieval timestamps:** Always store `retrieved_at` alongside content timestamps for causality
- **Eager loading large Parquet:** Use `pl.scan_parquet()` (lazy) instead of `pl.read_parquet()` (eager) for large files
- **Ignoring rate limits:** Always enable rate limiting in CCXT (`enableRateLimit=True`) and let PRAW handle Reddit limits
- **Over-partitioning Parquet:** Partition by date only; avoid high-cardinality columns that create many small files

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reddit rate limiting | Custom sleep/retry logic | PRAW's built-in rate limiter | PRAW handles X-Ratelimit-* headers automatically |
| Stock split adjustment | Manual price adjustment | yfinance `auto_adjust=True` | Yahoo handles corporate actions (mostly) |
| Exchange-specific crypto APIs | Per-exchange implementations | CCXT unified interface | CCXT abstracts 100+ exchanges |
| JSON parsing performance | Custom parsers | orjson | 10x faster than stdlib json |
| Config validation | Manual type checking | Pydantic Settings | Automatic validation on load |
| Retry with backoff | Custom retry loops | tenacity library | Battle-tested exponential backoff |

**Key insight:** Data ingestion is a solved problem. Every custom implementation introduces bugs that libraries have already fixed. Focus effort on causality guarantees and domain-specific logic, not reinventing HTTP clients.

## Common Pitfalls

### Pitfall 1: Lookahead Bias in Timestamps

**What goes wrong:** Using content `created_utc` without `retrieved_at` allows using data that wasn't actually available at prediction time.

**Why it happens:** Natural to think "post was created at time T, so it was available at time T." But crawling delays mean posts aren't available until retrieved.

**How to avoid:** Store both timestamps. In feature engineering, filter by `retrieved_at < prediction_time`, not `created_utc`.

**Warning signs:** Backtest performance much better than live performance; "magic" predictive ability.

### Pitfall 2: PRAW Timestamp Filtering Limitation

**What goes wrong:** Trying to use `subreddit.submissions(start, end)` which was deprecated in 2018.

**Why it happens:** Old tutorials and Stack Overflow answers reference this method.

**How to avoid:** Use checkpoint-based incremental crawling. For historical data, consider PushShift (limited post-2023) or accept PRAW's 1000-post limit per listing.

**Warning signs:** `AttributeError` or deprecation warnings about `submissions()` method.

### Pitfall 3: yfinance Split Adjustment Issues

**What goes wrong:** Yahoo Finance sometimes fails to apply stock splits correctly, especially around split dates.

**Why it happens:** Yahoo's data quality varies; free service with no SLA.

**How to avoid:** Always use `auto_adjust=True`. Cross-check data around known split dates. Consider caching validated data locally.

**Warning signs:** Price discontinuities of exactly 2x, 3x, or 10x; Adj Close not matching expected values.

### Pitfall 4: Small Parquet Files

**What goes wrong:** Creating many small Parquet files (< 100MB) that slow down reads.

**Why it happens:** Writing one file per record or over-partitioning by high-cardinality columns.

**How to avoid:** Batch records before writing. Target 100MB-1GB per file. Partition by date only, not by ticker or subreddit.

**Warning signs:** Thousands of files under 10MB; slow `scan_parquet()` performance.

### Pitfall 5: Missing Error Handling for API Failures

**What goes wrong:** Ingestion pipeline crashes on transient network errors, losing progress.

**Why it happens:** Not implementing retry logic; not saving checkpoints frequently.

**How to avoid:** Use tenacity for retries. Save checkpoints after each successful batch. Log all failures with context.

**Warning signs:** `ConnectionError` or `Timeout` exceptions crashing the pipeline.

### Pitfall 6: CCXT Exchange-Specific Quirks

**What goes wrong:** Assuming all exchanges behave identically through CCXT's unified API.

**Why it happens:** Different exchanges have different data formats, limits, and behaviors.

**How to avoid:** Test with each exchange separately. Check `exchange.has['fetchOHLCV']` before calling. Handle exchange-specific errors.

**Warning signs:** `ExchangeNotAvailable`, `BadSymbol`, or inconsistent data between exchanges.

## Code Examples

### Reddit Ingestion with PRAW

```python
# Source: PRAW documentation + best practices research
import praw
from datetime import datetime, timezone
from pathlib import Path
import json

def create_reddit_client(settings: RedditSettings) -> praw.Reddit:
    """Create authenticated Reddit client."""
    return praw.Reddit(
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        user_agent=settings.user_agent,
        ratelimit_seconds=300,  # Wait up to 5 min for rate limits
    )

def ingest_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    output_dir: Path,
    checkpoint_mgr: CheckpointManager,
    limit: int = 1000,
) -> int:
    """Ingest posts from subreddit with checkpointing."""
    subreddit = reddit.subreddit(subreddit_name)
    last_ts = checkpoint_mgr.get_last_timestamp(subreddit_name)

    count = 0
    newest_ts = last_ts

    # PRAW handles rate limiting automatically
    for submission in subreddit.new(limit=limit):
        # Skip already-processed posts
        if submission.created_utc <= last_ts:
            continue

        # Store with causality timestamps
        store_reddit_post(submission, output_dir / subreddit_name)

        newest_ts = max(newest_ts, submission.created_utc)
        count += 1

        # Checkpoint every 100 posts
        if count % 100 == 0:
            checkpoint_mgr.update(subreddit_name, newest_ts)
            checkpoint_mgr.save()

    # Final checkpoint
    if count > 0:
        checkpoint_mgr.update(subreddit_name, newest_ts)
        checkpoint_mgr.save()

    return count
```

### Market Data Ingestion with yfinance

```python
# Source: yfinance documentation + best practices research
import yfinance as yf
import polars as pl
from datetime import datetime, timezone
from pathlib import Path

def ingest_equity_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> pl.DataFrame:
    """Fetch equity OHLCV data with retrieval timestamp."""
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # Batch download for efficiency
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,  # Handle splits/dividends
        threads=True,      # Parallel download
    )

    # Convert to long format with Polars
    records = []
    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue

        ticker_data = data[ticker].reset_index()
        for _, row in ticker_data.iterrows():
            records.append({
                "ticker": ticker,
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                "high": float(row["High"]) if pd.notna(row["High"]) else None,
                "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
                "retrieved_at": retrieved_at,
            })

    df = pl.DataFrame(records)

    # Save raw JSON for audit trail
    raw_path = output_dir / "raw" / f"equity_{start_date}_{end_date}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_json(raw_path)

    return df
```

### Crypto Data Ingestion with CCXT

```python
# Source: CCXT documentation + best practices research
import ccxt
import polars as pl
from datetime import datetime, timezone
from pathlib import Path

def ingest_crypto_ohlcv(
    exchange_id: str,
    symbols: list[str],
    timeframe: str = "1d",
    since_days: int = 365,
    output_dir: Path = None,
) -> pl.DataFrame:
    """Fetch crypto OHLCV data from exchange."""
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # Initialize exchange with rate limiting
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,  # CRITICAL: respect rate limits
    })

    # Calculate since timestamp
    since = exchange.milliseconds() - (since_days * 24 * 60 * 60 * 1000)

    all_records = []

    for symbol in symbols:
        # Check if exchange supports OHLCV
        if not exchange.has.get("fetchOHLCV"):
            continue

        try:
            # Fetch with pagination for large date ranges
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,  # Most exchanges cap at 1000
            )

            for candle in ohlcv:
                all_records.append({
                    "exchange": exchange_id,
                    "symbol": symbol,
                    "timestamp": candle[0],
                    "date": datetime.fromtimestamp(candle[0] / 1000, timezone.utc).strftime("%Y-%m-%d"),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "retrieved_at": retrieved_at,
                })
        except ccxt.BaseError as e:
            # Log but don't crash on individual symbol failures
            print(f"Error fetching {symbol} from {exchange_id}: {e}")

    df = pl.DataFrame(all_records)

    if output_dir:
        raw_path = output_dir / "raw" / f"crypto_{exchange_id}_{since_days}d.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_json(raw_path)

    return df
```

### Bronze Layer Parquet Writing

```python
# Source: Polars + Parquet best practices
import polars as pl
from pathlib import Path

def write_bronze_parquet(
    df: pl.DataFrame,
    bronze_dir: Path,
    partition_col: str = "date",
    source: str = "reddit",
) -> None:
    """Write DataFrame to bronze Parquet with Hive partitioning."""
    output_base = bronze_dir / source
    output_base.mkdir(parents=True, exist_ok=True)

    # Partition by date for optimal file sizes
    for partition_value in df[partition_col].unique().to_list():
        partition_df = df.filter(pl.col(partition_col) == partition_value)

        partition_path = output_base / f"{partition_col}={partition_value}"
        partition_path.mkdir(parents=True, exist_ok=True)

        output_file = partition_path / "data.parquet"

        partition_df.write_parquet(
            output_file,
            compression="zstd",       # Good compression ratio
            row_group_size=100_000,   # ~128MB row groups
            use_pyarrow=True,         # Better compatibility
        )

def read_bronze_parquet(
    bronze_dir: Path,
    source: str,
    date_filter: str = None,
) -> pl.LazyFrame:
    """Read bronze Parquet with lazy evaluation."""
    source_dir = bronze_dir / source

    # Use scan (lazy) for large datasets
    lf = pl.scan_parquet(
        source_dir / "**/*.parquet",
        hive_partitioning=True,
    )

    if date_filter:
        lf = lf.filter(pl.col("date") == date_filter)

    return lf
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pandas for DataFrames | Polars | 2023-2024 | 5-50x faster, lower memory usage |
| Direct Reddit API | PRAW with OAuth | 2023 (API changes) | Required for any significant usage |
| PushShift for historical | Limited/unavailable | 2023 | Must use PRAW or alternative sources |
| JSON log files | Structured JSON logging | 2024+ | Better observability, easier debugging |
| pandas `.to_parquet()` | Polars `.write_parquet()` | 2024+ | Native support, better performance |

**Deprecated/outdated:**
- **PRAW `submissions()` method:** Removed in 2018 when Reddit deprecated the underlying API
- **PushShift unlimited access:** Severely limited after Reddit API changes in 2023
- **Pandas for large datasets:** Still works but significantly slower than Polars

## Open Questions

1. **Historical Reddit data availability**
   - What we know: PRAW limited to ~1000 posts per listing; PushShift restricted post-2023
   - What's unclear: Best source for historical finance subreddit data if needed
   - Recommendation: Start fresh with incremental crawling; if historical needed, explore academic data sources or paid alternatives

2. **yfinance data reliability**
   - What we know: Free service with no SLA; occasional data quality issues
   - What's unclear: Frequency and severity of split adjustment errors
   - Recommendation: Implement validation checks; cross-reference critical dates with official sources

3. **Optimal Parquet file sizes for this dataset**
   - What we know: Target 100MB-1GB generally
   - What's unclear: Daily Reddit volume for finance subreddits
   - Recommendation: Start with daily partitioning; monitor file sizes and adjust

## Sources

### Primary (HIGH confidence)
- [PRAW Rate Limits Documentation](https://praw.readthedocs.io/en/stable/getting_started/ratelimits.html) - Rate limiting behavior
- [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance) - Core functionality and limitations
- [CCXT GitHub Repository](https://github.com/ccxt/ccxt) - Exchange support and features
- [Polars Parquet Documentation](https://docs.pola.rs/user-guide/io/parquet/) - Parquet I/O patterns
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - Configuration management

### Secondary (MEDIUM confidence)
- [Medallion Architecture Guide](https://www.chaosgenius.io/blog/medallion-architecture/) - Bronze/silver/gold patterns
- [Python Project Structure Best Practices](https://realpython.com/ref/best-practices/project-layout/) - src/ layout
- [yfinance Complete Guide](https://algotrading101.com/learn/yfinance-guide/) - Usage patterns
- [Python Logging Best Practices 2026](https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/) - Structured logging

### Tertiary (LOW confidence)
- Various Medium articles on incremental crawling patterns
- Stack Overflow discussions on PRAW timestamp limitations
- Community discussions on yfinance data quality issues

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via official documentation
- Architecture: HIGH - Medallion architecture well-documented, src/ layout is Python standard
- Pitfalls: MEDIUM - Based on GitHub issues and community reports; some may be fixed

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (30 days - stable libraries)
