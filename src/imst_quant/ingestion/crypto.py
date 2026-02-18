"""Crypto OHLCV ingestion via CCXT."""

from datetime import datetime, timezone
from pathlib import Path

import ccxt
import polars as pl
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def ingest_crypto_ohlcv(
    exchange_id: str,
    symbols: list[str],
    timeframe: str = "1d",
    since_days: int = 365,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    """Fetch crypto OHLCV data from exchange."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    since = exchange.milliseconds() - (since_days * 24 * 60 * 60 * 1000)
    all_records = []

    for symbol in symbols:
        if not exchange.has.get("fetchOHLCV", True):
            continue
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,
            )
            for candle in ohlcv:
                all_records.append({
                    "exchange": exchange_id,
                    "symbol": symbol,
                    "timestamp": candle[0],
                    "date": datetime.fromtimestamp(
                        candle[0] / 1000, timezone.utc
                    ).strftime("%Y-%m-%d"),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "retrieved_at": retrieved_at,
                })
        except ccxt.BaseError as e:
            logger.warning("crypto_fetch_failed", symbol=symbol, exchange=exchange_id, error=str(e))

    df = pl.DataFrame(all_records) if all_records else pl.DataFrame(
        schema={
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "timestamp": pl.Int64,
            "date": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "retrieved_at": pl.Utf8,
        }
    )

    if len(df) > 0 and output_dir:
        output_dir = Path(output_dir)
        crypto_dir = output_dir / "crypto"
        crypto_dir.mkdir(parents=True, exist_ok=True)
        raw_path = crypto_dir / f"crypto_{exchange_id}_{since_days}d.json"
        df.write_json(raw_path)

    return df
