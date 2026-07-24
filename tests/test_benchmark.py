"""Tests for the benchmark comparison module.

Tests cover:
- BenchmarkAnalyzer alpha/beta regression against a known linear relationship
- Tracking error, information ratio, correlation
- Up/down capture ratios
- create_benchmark_from_prices equal-weight aggregation
- compare_to_benchmark convenience wrapper
"""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.benchmark import (
    BenchmarkAnalyzer,
    compare_to_benchmark,
    create_benchmark_from_prices,
)


def test_beta_one_when_strategy_equals_benchmark():
    returns = [0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.01]
    analyzer = BenchmarkAnalyzer(returns, returns)

    assert analyzer.beta == pytest.approx(1.0, abs=1e-6)
    assert analyzer.alpha == pytest.approx(0.0, abs=1e-6)
    assert analyzer.r_squared == pytest.approx(1.0, abs=1e-6)


def test_beta_recovers_known_linear_relationship():
    benchmark = np.array([0.01, -0.02, 0.015, -0.005, 0.02, 0.008, -0.01])
    strategy = 2.0 * benchmark  # beta = 2, alpha = 0
    analyzer = BenchmarkAnalyzer(strategy, benchmark)

    assert analyzer.beta == pytest.approx(2.0, abs=1e-6)
    assert analyzer.alpha == pytest.approx(0.0, abs=1e-6)


def test_tracking_error_zero_when_returns_match():
    returns = [0.01, -0.02, 0.015, -0.005, 0.02]
    analyzer = BenchmarkAnalyzer(returns, returns)

    assert analyzer.tracking_error == pytest.approx(0.0, abs=1e-9)
    assert analyzer.information_ratio == 0.0


def test_correlation_perfect_positive():
    returns = [0.01, -0.02, 0.015, -0.005, 0.02]
    analyzer = BenchmarkAnalyzer(returns, returns)

    assert analyzer.correlation == pytest.approx(1.0, abs=1e-6)


def test_capture_ratios_outperform_on_up_days_only():
    benchmark = [0.02, -0.01, 0.03, -0.02]
    strategy = [0.04, -0.01, 0.06, -0.02]  # doubles benchmark on up days only
    analyzer = BenchmarkAnalyzer(strategy, benchmark)
    up_capture, down_capture, capture_ratio = analyzer.capture_ratios()

    assert up_capture > 100.0
    assert down_capture == pytest.approx(100.0, abs=1e-6)
    assert capture_ratio > 1.0


def test_excess_return_positive_when_strategy_outperforms():
    benchmark = [0.01, 0.01, 0.01]
    strategy = [0.02, 0.02, 0.02]
    analyzer = BenchmarkAnalyzer(strategy, benchmark)

    assert analyzer.excess_return() > 0


def test_short_series_falls_back_to_neutral_regression():
    analyzer = BenchmarkAnalyzer([0.01], [0.01])

    assert analyzer.alpha == 0.0
    assert analyzer.beta == 1.0
    assert analyzer.r_squared == 0.0


def test_calculate_all_metrics_returns_populated_dataclass():
    benchmark = [0.01, -0.02, 0.015, -0.005, 0.02]
    strategy = [0.012, -0.018, 0.02, -0.004, 0.025]
    metrics = BenchmarkAnalyzer(strategy, benchmark).calculate_all_metrics()

    assert metrics.tracking_error >= 0
    assert -1.0 <= metrics.correlation <= 1.0


def test_create_benchmark_from_prices_equal_weight():
    prices = pl.DataFrame({
        "date": [1, 1, 2, 2, 3, 3],
        "asset_id": ["A", "B", "A", "B", "A", "B"],
        "close": [100.0, 50.0, 110.0, 55.0, 99.0, 60.0],
    })
    benchmark = create_benchmark_from_prices(prices, "equal_weight")

    assert benchmark.columns == ["date", "benchmark_return"]
    # day 2: A +10%, B +10% -> equal-weight avg = 10%
    day2 = benchmark.filter(pl.col("date") == 2)["benchmark_return"][0]
    assert day2 == pytest.approx(0.10, abs=1e-9)


def test_compare_to_benchmark_returns_expected_keys():
    benchmark = pl.Series([0.01, -0.02, 0.015, -0.005, 0.02])
    strategy = pl.Series([0.012, -0.018, 0.02, -0.004, 0.025])
    result = compare_to_benchmark(strategy, benchmark)

    for key in ("alpha", "beta", "information_ratio", "tracking_error",
                "excess_return", "correlation", "r_squared",
                "up_capture", "down_capture", "capture_ratio"):
        assert key in result
