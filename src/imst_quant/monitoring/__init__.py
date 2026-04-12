"""Trading system monitoring and observability.

This module provides utilities for monitoring trading system health,
tracking performance metrics, generating alerts, and exposing metrics
for external monitoring systems like Prometheus.

Features:
    - Pipeline latency tracking
    - Data quality metrics collection
    - Trading performance metric recording
    - Alert generation and threshold monitoring
    - Prometheus-compatible metric export
    - System health checks

Example:
    >>> from imst_quant.monitoring import MetricsCollector, check_system_health
    >>> collector = MetricsCollector()
    >>> collector.record_pipeline_latency("ingestion", 2.5)
    >>> collector.record_data_quality("reddit", rows=1000, nulls=5)
    >>> health = check_system_health("data/")
    >>> print(f"System healthy: {health['healthy']}")
"""

from imst_quant.monitoring.metrics import (
    Alert,
    AlertSeverity,
    MetricPoint,
    MetricsCollector,
    MetricType,
    check_system_health,
)

__all__ = [
    "MetricsCollector",
    "MetricPoint",
    "MetricType",
    "Alert",
    "AlertSeverity",
    "check_system_health",
]


def expose_metrics():
    """Placeholder for Prometheus metrics HTTP endpoint.

    This function will be implemented to start an HTTP server
    exposing metrics in Prometheus format on a configurable port.
    """
    pass
