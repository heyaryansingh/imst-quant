"""Equity OHLCV ingestion via yfinance."""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yfinance as yf


def _scalar(val):
    """Extract scalar from pandas/numpy types."""
    if val is None:
        return None
    if hasattr(val, "iloc"):
        val = val.iloc[0]
    if hasattr(val, "item"):
        return val.item()
    return val


def _safe_float(val):
    v = _scalar(val)
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    v = _scalar(val)
    if v is None:
        return 0
    try:
        f = float(v)
        return int(f) if f == f else 0
    except (TypeError, ValueError):
        return 0


def ingest_equity_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> pl.DataFrame:
    """Fetch equity OHLCV data with retrieval timestamp."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(output_dir)
    records = []

    for ticker in tickers:
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            continue
        for idx, row in data.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            records.append({
                "ticker": ticker,
                "date": date_str,
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": _safe_int(row.get("Volume")),
                "retrieved_at": retrieved_at,
            })

    df = pl.DataFrame(records) if records else pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "date": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "retrieved_at": pl.Utf8,
        }
    )

    if len(df) > 0:
        market_dir = output_dir / "market"
        market_dir.mkdir(parents=True, exist_ok=True)
        raw_path = market_dir / f"equity_{start_date}_{end_date}.json"
        df.write_json(raw_path)

    return df
