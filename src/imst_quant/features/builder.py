"""Daily feature vectors with strict causality (FEAT-01 to FEAT-05)."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import polars as pl
import structlog

logger = structlog.get_logger()

RELATED_STOCKS = {
    "AAPL": ["MSFT", "GOOGL", "META"],
    "JNJ": ["PFE", "MRK", "UNH"],
    "JPM": ["BAC", "GS", "MS"],
    "XOM": ["CVX", "COP", "SLB"],
}


def _load_market(bronze_dir: Path, asset: str, start: str, end: str) -> pl.DataFrame:
    market_dir = bronze_dir / "market"
    if not market_dir.exists():
        return pl.DataFrame()
    dfs = []
    for p in market_dir.glob("date=*/data.parquet"):
        d = p.parent.name.replace("date=", "")
        if start <= d <= end:
            df = pl.read_parquet(p)
            if "ticker" in df.columns:
                df = df.filter(pl.col("ticker") == asset)
            elif "symbol" in df.columns:
                df = df.filter(pl.col("symbol").str.contains(asset))
            if len(df) > 0:
                dfs.append(df)
    return pl.concat(dfs) if dfs else pl.DataFrame()


def _returns_and_vol(df: pl.DataFrame) -> tuple:
    if len(df) < 2 or "close" not in df.columns:
        return 0.0, 0.0, 0.0
    df = df.sort("date")
    closes = df["close"].to_list()
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    if not rets:
        return 0.0, 0.0, 0.0
    ret_1d = rets[-1]
    ret_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else ret_1d
    vol = (sum((r - sum(rets) / len(rets)) ** 2 for r in rets) / len(rets)) ** 0.5
    return ret_1d, ret_5d, vol


def _load_sentiment(sentiment_path: Path, asset: str, as_of_date: str) -> Dict:
    if not sentiment_path.exists():
        return {"sentiment_index": 0.0, "post_count": 0}
    df = pl.read_parquet(sentiment_path)
    df = df.filter((pl.col("asset_id") == asset) & (pl.col("date") <= as_of_date))
    if len(df) == 0:
        return {"sentiment_index": 0.0, "post_count": 0}
    row = df.sort("date").tail(1).to_dicts()[0]
    return {
        "sentiment_index": row.get("sentiment_index", 0.0),
        "post_count": row.get("post_count", 0),
    }


def build_daily_features(
    bronze_dir: Path,
    sentiment_path: Path,
    output_path: Path,
    assets: List[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Build gold feature vectors with T-1 price, sentiment before date."""
    bronze_dir = Path(bronze_dir)
    sentiment_path = Path(sentiment_path)
    output_path = Path(output_path)
    assets = assets or ["AAPL", "JNJ", "JPM", "XOM"]

    rows = []
    end = end_date or "2099-12-31"
    start = start_date or "2020-01-01"

    for asset in assets:
        for d in _date_range(start, end):
            prev = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
            window_start = (date.fromisoformat(d) - timedelta(days=35)).isoformat()

            market = _load_market(bronze_dir, asset, window_start, prev)
            ret_1d, ret_5d, vol_30d = _returns_and_vol(market)

            related = RELATED_STOCKS.get(asset, [])[:3]
            rel_1, rel_2, rel_3 = 0.0, 0.0, 0.0
            for i, r in enumerate(related):
                rm = _load_market(bronze_dir, r, prev, prev)
                r1, _, _ = _returns_and_vol(rm)
                if i == 0:
                    rel_1 = r1
                elif i == 1:
                    rel_2 = r1
                else:
                    rel_3 = r1

            sent = _load_sentiment(sentiment_path, asset, d)

            rows.append({
                "date": d,
                "asset_id": asset,
                "return_1d": ret_1d,
                "return_5d": ret_5d,
                "volatility_30d": vol_30d,
                "related_return_1": rel_1,
                "related_return_2": rel_2,
                "related_return_3": rel_3,
                "sentiment_index": sent["sentiment_index"],
                "post_count": sent["post_count"],
            })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info("features_written", path=str(output_path), rows=len(df))
    return df


def _date_range(start: str, end: str, limit: int = 1000) -> List[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    d = s
    while d <= e and len(out) < limit:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class FeatureBuilder:
    """Build feature vectors (wrapper)."""

    def build_features(
        self,
        bronze_dir: Path,
        sentiment_path: Path,
        output_path: Path,
        assets: List[str] | None = None,
    ) -> pl.DataFrame:
        return build_daily_features(bronze_dir, sentiment_path, output_path, assets)
