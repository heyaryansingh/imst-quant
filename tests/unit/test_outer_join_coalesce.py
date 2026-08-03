"""Tests for full-join key coalescing in drift and Brinson attribution."""

import polars as pl
import pytest

from imst_quant.utils.attribution import PerformanceAttributor
from imst_quant.utils.rebalancing import calculate_drift, needs_rebalancing


def test_drift_keeps_symbols_for_target_only_positions():
    """A target-only symbol used to come back as a null key."""
    current = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.6, 0.4]})
    target = pl.DataFrame({"symbol": ["AAPL", "NVDA"], "weight": [0.5, 0.5]})

    drift = calculate_drift(current, target).sort("symbol")

    assert drift["symbol"].to_list() == ["AAPL", "MSFT", "NVDA"]
    assert drift["symbol"].null_count() == 0


def test_drift_returns_the_documented_columns():
    current = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.55]})
    target = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.50]})

    drift = calculate_drift(current, target)

    assert drift.columns == [
        "symbol",
        "current_weight",
        "target_weight",
        "absolute_drift",
        "relative_drift",
    ]
    assert drift["absolute_drift"][0] == pytest.approx(0.05)


def test_drift_weights_for_one_sided_positions():
    current = pl.DataFrame({"symbol": ["MSFT"], "weight": [0.4]})
    target = pl.DataFrame({"symbol": ["NVDA"], "weight": [0.5]})

    rows = {r["symbol"]: r for r in calculate_drift(current, target).to_dicts()}

    assert rows["MSFT"]["target_weight"] == 0.0
    assert rows["NVDA"]["current_weight"] == 0.0
    assert rows["NVDA"]["absolute_drift"] == pytest.approx(-0.5)


def test_needs_rebalancing_still_detects_drift():
    current = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.56]})
    target = pl.DataFrame({"symbol": ["AAPL"], "weight": [0.50]})

    assert needs_rebalancing(current, target, threshold=0.05) is True
    assert needs_rebalancing(target, target, threshold=0.05) is False


def test_brinson_keeps_every_benchmark_only_sector():
    """Two benchmark-only sectors used to collapse onto a single null key."""
    portfolio = pl.DataFrame(
        {
            "asset_id": ["A", "B", "C"],
            "sector": ["tech", "tech", "energy"],
            "weight": [0.4, 0.2, 0.4],
            "return": [0.10, 0.05, 0.02],
            "date": ["2024-01-01"] * 3,
        }
    )
    benchmark = pl.DataFrame(
        {
            "asset_id": ["A", "D", "E"],
            "sector": ["tech", "health", "utils"],
            "weight": [0.3, 0.4, 0.3],
            "return": [0.08, 0.03, 0.01],
            "date": ["2024-01-01"] * 3,
        }
    )

    result = PerformanceAttributor(portfolio, benchmark).brinson_attribution()

    assert set(result.sector_details) == {"tech", "energy", "health", "utils"}
    assert None not in result.sector_details
    assert result.total_active_return == pytest.approx(
        result.allocation_effect + result.selection_effect + result.interaction_effect
    )
