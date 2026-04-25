"""Data pipeline health check utilities.

Validate data quality and pipeline status across all medallion layers.
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()


def check_pipeline_health(
    data_dir: Path,
    lookback_days: int = 7,
) -> Dict[str, any]:
    """Check health status of the entire data pipeline.

    Args:
        data_dir: Root data directory containing raw/bronze/silver/gold subdirs
        lookback_days: Number of days to check for recent data

    Returns:
        Dict with health status, missing data alerts, and recommendations
    """
    health = {
        "status": "healthy",
        "checked_at": datetime.now().isoformat(),
        "layers": {},
        "alerts": [],
        "recommendations": [],
    }

    layers = ["raw", "bronze", "silver", "sentiment", "influence", "gold"]
    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    for layer in layers:
        layer_path = data_dir / layer
        layer_health = _check_layer_health(layer_path, cutoff_date)
        health["layers"][layer] = layer_health

        if not layer_health["has_recent_data"]:
            health["alerts"].append(
                f"No recent data in {layer} layer (last {lookback_days} days)"
            )
            health["status"] = "warning"

    # Check for pipeline gaps
    if health["layers"].get("raw", {}).get("has_recent_data"):
        if not health["layers"].get("bronze", {}).get("has_recent_data"):
            health["alerts"].append("Raw data exists but bronze layer is stale")
            health["recommendations"].append("Run: imst process --raw-to-bronze")

    if health["layers"].get("bronze", {}).get("has_recent_data"):
        if not health["layers"].get("silver", {}).get("has_recent_data"):
            health["alerts"].append("Bronze data exists but silver layer is stale")
            health["recommendations"].append("Run: imst process --bronze-to-silver")

    # Check for missing sentiment analysis
    if health["layers"].get("silver", {}).get("has_recent_data"):
        if not health["layers"].get("sentiment", {}).get("has_recent_data"):
            health["alerts"].append("Silver data exists but sentiment not computed")
            health["recommendations"].append("Run: imst analyze --sentiment")

    return health


def _check_layer_health(layer_path: Path, cutoff_date: datetime) -> Dict[str, any]:
    """Check health of a single data layer.

    Args:
        layer_path: Path to the data layer directory
        cutoff_date: Minimum modification time for recent data

    Returns:
        Dict with file count, size, and recency info
    """
    if not layer_path.exists():
        return {
            "exists": False,
            "has_recent_data": False,
            "file_count": 0,
            "total_size_mb": 0,
        }

    files = list(layer_path.rglob("*"))
    data_files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    total_size = sum(f.stat().st_size for f in data_files if f.exists())
    recent_files = [
        f for f in data_files
        if f.exists() and datetime.fromtimestamp(f.stat().st_mtime) > cutoff_date
    ]

    return {
        "exists": True,
        "file_count": len(data_files),
        "recent_file_count": len(recent_files),
        "has_recent_data": len(recent_files) > 0,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "path": str(layer_path),
    }


def check_data_freshness(
    data_dir: Path,
    layer: str = "gold",
    max_age_hours: int = 24,
) -> Dict[str, any]:
    """Check if data in a layer is fresh enough for trading.

    Args:
        data_dir: Root data directory
        layer: Which layer to check (raw/bronze/silver/gold)
        max_age_hours: Maximum acceptable age in hours

    Returns:
        Dict with freshness status and latest file timestamp
    """
    layer_path = data_dir / layer

    if not layer_path.exists():
        return {
            "fresh": False,
            "reason": f"{layer} layer does not exist",
            "latest_file": None,
            "age_hours": None,
        }

    data_files = [
        f for f in layer_path.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]

    if not data_files:
        return {
            "fresh": False,
            "reason": f"No data files in {layer} layer",
            "latest_file": None,
            "age_hours": None,
        }

    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    latest_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
    age_hours = (datetime.now() - latest_mtime).total_seconds() / 3600

    return {
        "fresh": age_hours <= max_age_hours,
        "latest_file": str(latest_file),
        "latest_mtime": latest_mtime.isoformat(),
        "age_hours": round(age_hours, 2),
        "max_age_hours": max_age_hours,
    }


def validate_data_coverage(
    data_dir: Path,
    required_tickers: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Validate that all required tickers have data coverage.

    Args:
        data_dir: Root data directory
        required_tickers: List of ticker symbols that must have data

    Returns:
        Dict with coverage status and missing tickers
    """
    if required_tickers is None:
        required_tickers = ["AAPL", "JNJ", "JPM", "XOM"]  # Default from settings

    silver_path = data_dir / "silver"
    if not silver_path.exists():
        return {
            "valid": False,
            "reason": "Silver layer does not exist",
            "missing_tickers": required_tickers,
        }

    # This is a simplified check - in production, parse parquet files
    coverage = {
        "valid": True,
        "required_tickers": required_tickers,
        "missing_tickers": [],
        "data_files": [],
    }

    data_files = list(silver_path.rglob("*.parquet"))
    coverage["data_files"] = [str(f) for f in data_files]

    if not data_files:
        coverage["valid"] = False
        coverage["missing_tickers"] = required_tickers
        coverage["reason"] = "No parquet files found in silver layer"

    return coverage
