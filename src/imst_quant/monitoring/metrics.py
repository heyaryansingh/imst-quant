"""Trading system monitoring and performance metrics.

This module provides utilities for monitoring trading system health,
tracking performance metrics, and generating alerts. Designed for
integration with Prometheus/Grafana or standalone usage.

Features:
    - Pipeline latency tracking
    - Data quality metrics
    - Trading performance metrics
    - System health checks
    - Alert generation

Example:
    >>> from imst_quant.monitoring.metrics import MetricsCollector
    >>> metrics = MetricsCollector()
    >>> metrics.record_pipeline_latency("ingestion", 2.5)
    >>> metrics.record_data_quality("reddit", rows=1000, nulls=5)
    >>> print(metrics.get_summary())
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics tracked."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    LATENCY = "latency"


@dataclass
class Alert:
    """Represents a monitoring alert.

    Attributes:
        name: Alert identifier.
        message: Human-readable alert description.
        severity: Alert severity level.
        timestamp: When the alert was generated.
        labels: Additional context labels.
        value: Metric value that triggered the alert.
    """

    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None


@dataclass
class MetricPoint:
    """A single metric observation.

    Attributes:
        name: Metric name.
        value: Metric value.
        timestamp: Observation timestamp.
        labels: Dimensional labels.
        metric_type: Type of metric.
    """

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


class MetricsCollector:
    """Collects and aggregates trading system metrics.

    This class provides a centralized interface for recording various
    metrics from the trading pipeline, including latency, data quality,
    and trading performance metrics.

    Attributes:
        metrics: List of recorded metric points.
        alerts: List of generated alerts.
        thresholds: Alert threshold configurations.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_pipeline_latency("sentiment", 1.5)
        >>> collector.record_trading_metric("sharpe_ratio", 1.8)
        >>> summary = collector.get_summary()
        >>> print(f"Total metrics: {summary['total_metrics']}")
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            thresholds: Optional dict of metric name to alert threshold.
                Default thresholds are used for common metrics if not provided.
        """
        self.metrics: List[MetricPoint] = []
        self.alerts: List[Alert] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []

        # Default thresholds
        self.thresholds: Dict[str, Dict[str, Any]] = {
            "pipeline_latency_seconds": {
                "warning": 60.0,
                "critical": 300.0,
            },
            "data_quality_null_pct": {
                "warning": 5.0,
                "critical": 20.0,
            },
            "model_inference_latency_ms": {
                "warning": 100.0,
                "critical": 500.0,
            },
            "position_drawdown_pct": {
                "warning": 5.0,
                "critical": 10.0,
            },
        }
        if thresholds:
            for name, value in thresholds.items():
                if name in self.thresholds:
                    self.thresholds[name]["warning"] = value

        logger.info("metrics_collector_initialized")

    def _record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE,
    ) -> MetricPoint:
        """Record a metric point.

        Args:
            name: Metric name.
            value: Metric value.
            labels: Optional dimensional labels.
            metric_type: Type of metric.

        Returns:
            The recorded metric point.
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=metric_type,
        )
        self.metrics.append(point)

        # Check thresholds and generate alerts
        self._check_thresholds(point)

        return point

    def _check_thresholds(self, point: MetricPoint) -> None:
        """Check if metric exceeds thresholds and generate alerts.

        Args:
            point: The metric point to check.
        """
        if point.name not in self.thresholds:
            return

        threshold = self.thresholds[point.name]
        critical = threshold.get("critical")
        warning = threshold.get("warning")

        if critical is not None and point.value >= critical:
            alert = Alert(
                name=f"{point.name}_critical",
                message=f"{point.name} exceeded critical threshold: {point.value} >= {critical}",
                severity=AlertSeverity.CRITICAL,
                labels=point.labels,
                value=point.value,
            )
            self._fire_alert(alert)
        elif warning is not None and point.value >= warning:
            alert = Alert(
                name=f"{point.name}_warning",
                message=f"{point.name} exceeded warning threshold: {point.value} >= {warning}",
                severity=AlertSeverity.WARNING,
                labels=point.labels,
                value=point.value,
            )
            self._fire_alert(alert)

    def _fire_alert(self, alert: Alert) -> None:
        """Fire an alert and notify handlers.

        Args:
            alert: The alert to fire.
        """
        self.alerts.append(alert)
        logger.warning(
            "alert_fired",
            name=alert.name,
            severity=alert.severity.value,
            message=alert.message,
        )
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("alert_handler_failed", error=str(e))

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler callback.

        Args:
            handler: Function to call when alerts are fired.
        """
        self._alert_handlers.append(handler)

    def record_pipeline_latency(
        self,
        pipeline_name: str,
        latency_seconds: float,
        stage: Optional[str] = None,
    ) -> MetricPoint:
        """Record pipeline execution latency.

        Args:
            pipeline_name: Name of the pipeline (e.g., "ingestion", "sentiment").
            latency_seconds: Execution time in seconds.
            stage: Optional sub-stage within the pipeline.

        Returns:
            The recorded metric point.
        """
        labels = {"pipeline": pipeline_name}
        if stage:
            labels["stage"] = stage

        return self._record(
            name="pipeline_latency_seconds",
            value=latency_seconds,
            labels=labels,
            metric_type=MetricType.LATENCY,
        )

    def record_data_quality(
        self,
        source: str,
        rows: int,
        nulls: int = 0,
        duplicates: int = 0,
        outliers: int = 0,
    ) -> Dict[str, MetricPoint]:
        """Record data quality metrics.

        Args:
            source: Data source identifier (e.g., "reddit", "market").
            rows: Total number of rows processed.
            nulls: Number of null values encountered.
            duplicates: Number of duplicate rows detected.
            outliers: Number of outliers detected.

        Returns:
            Dictionary of recorded metric points.
        """
        labels = {"source": source}
        null_pct = (nulls / rows * 100) if rows > 0 else 0.0
        dup_pct = (duplicates / rows * 100) if rows > 0 else 0.0

        points = {}
        points["rows"] = self._record(
            "data_quality_rows",
            float(rows),
            labels,
            MetricType.COUNTER,
        )
        points["null_pct"] = self._record(
            "data_quality_null_pct",
            null_pct,
            labels,
        )
        points["duplicate_pct"] = self._record(
            "data_quality_duplicate_pct",
            dup_pct,
            labels,
        )
        points["outliers"] = self._record(
            "data_quality_outliers",
            float(outliers),
            labels,
            MetricType.COUNTER,
        )

        logger.debug(
            "data_quality_recorded",
            source=source,
            rows=rows,
            null_pct=round(null_pct, 2),
        )

        return points

    def record_trading_metric(
        self,
        metric_name: str,
        value: float,
        asset: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> MetricPoint:
        """Record a trading performance metric.

        Args:
            metric_name: Name of the metric (e.g., "sharpe_ratio", "win_rate").
            value: Metric value.
            asset: Optional asset identifier.
            strategy: Optional strategy name.

        Returns:
            The recorded metric point.
        """
        labels = {}
        if asset:
            labels["asset"] = asset
        if strategy:
            labels["strategy"] = strategy

        return self._record(
            name=f"trading_{metric_name}",
            value=value,
            labels=labels,
        )

    def record_model_inference(
        self,
        model_name: str,
        latency_ms: float,
        batch_size: int = 1,
    ) -> MetricPoint:
        """Record model inference metrics.

        Args:
            model_name: Name of the model.
            latency_ms: Inference latency in milliseconds.
            batch_size: Number of samples in the batch.

        Returns:
            The recorded metric point.
        """
        labels = {"model": model_name, "batch_size": str(batch_size)}
        return self._record(
            "model_inference_latency_ms",
            latency_ms,
            labels,
            MetricType.LATENCY,
        )

    def record_position_metric(
        self,
        symbol: str,
        metric_name: str,
        value: float,
    ) -> MetricPoint:
        """Record position-level metrics.

        Args:
            symbol: Asset symbol.
            metric_name: Name of the metric (e.g., "unrealized_pnl", "drawdown_pct").
            value: Metric value.

        Returns:
            The recorded metric point.
        """
        return self._record(
            name=f"position_{metric_name}",
            value=value,
            labels={"symbol": symbol},
        )

    def get_metrics_by_name(self, name: str) -> List[MetricPoint]:
        """Get all metric points with a specific name.

        Args:
            name: Metric name to filter by.

        Returns:
            List of matching metric points.
        """
        return [m for m in self.metrics if m.name == name]

    def get_latest_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get the most recent value for a metric.

        Args:
            name: Metric name.
            labels: Optional labels to filter by.

        Returns:
            Most recent value or None if not found.
        """
        matching = [m for m in self.metrics if m.name == name]
        if labels:
            matching = [
                m
                for m in matching
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]
        if not matching:
            return None
        return max(matching, key=lambda m: m.timestamp).value

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alerts with optional filtering.

        Args:
            severity: Filter by severity level.
            since: Filter to alerts after this time.

        Returns:
            List of matching alerts.
        """
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        return alerts

    def get_summary(self) -> Dict:
        """Get a summary of collected metrics and alerts.

        Returns:
            Dictionary containing metric counts, alert counts, and
            latest values for key metrics.
        """
        metric_counts: Dict[str, int] = {}
        for m in self.metrics:
            metric_counts[m.name] = metric_counts.get(m.name, 0) + 1

        alert_counts = {
            "total": len(self.alerts),
            "critical": len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            "warning": len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in self.alerts if a.severity == AlertSeverity.INFO]),
        }

        return {
            "total_metrics": len(self.metrics),
            "unique_metrics": len(metric_counts),
            "metric_counts": metric_counts,
            "alert_counts": alert_counts,
            "latest_pipeline_latency": self.get_latest_value("pipeline_latency_seconds"),
            "latest_data_quality_null_pct": self.get_latest_value("data_quality_null_pct"),
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format.
        """
        lines = []
        for m in self.metrics:
            label_str = ",".join(f'{k}="{v}"' for k, v in m.labels.items())
            if label_str:
                lines.append(f"{m.name}{{{label_str}}} {m.value}")
            else:
                lines.append(f"{m.name} {m.value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all recorded metrics and alerts."""
        self.metrics.clear()
        self.alerts.clear()
        logger.info("metrics_collector_cleared")


def check_system_health(data_dir: str) -> Dict[str, Any]:
    """Check overall system health and data availability.

    Args:
        data_dir: Root data directory to check.

    Returns:
        Dictionary with health check results:
            - healthy: Boolean overall health status
            - checks: Dictionary of individual check results
            - issues: List of identified issues
    """
    from pathlib import Path

    data_path = Path(data_dir)
    checks: Dict[str, bool] = {}
    issues: List[str] = []

    # Check directory structure
    expected_dirs = ["raw", "bronze", "silver", "gold", "sentiment"]
    for dir_name in expected_dirs:
        dir_path = data_path / dir_name
        checks[f"{dir_name}_dir_exists"] = dir_path.exists()
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_name}")

    # Check for recent data files
    gold_path = data_path / "gold"
    if gold_path.exists():
        parquet_files = list(gold_path.glob("*.parquet"))
        checks["gold_has_data"] = len(parquet_files) > 0
        if not parquet_files:
            issues.append("No parquet files in gold directory")
    else:
        checks["gold_has_data"] = False

    # Check sentiment data
    sentiment_path = data_path / "sentiment" / "sentiment_aggregates.parquet"
    checks["sentiment_exists"] = sentiment_path.exists()
    if not sentiment_path.exists():
        issues.append("Missing sentiment aggregates file")

    healthy = all(checks.values())

    return {
        "healthy": healthy,
        "checks": checks,
        "issues": issues,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
