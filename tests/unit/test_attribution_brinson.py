"""Tests for Brinson sector attribution in imst_quant.utils.attribution."""

import math

import polars as pl
import pytest

from imst_quant.utils.attribution import PerformanceAttributor


def _attributor(portfolio: pl.DataFrame, benchmark: pl.DataFrame) -> PerformanceAttributor:
    return PerformanceAttributor(portfolio, benchmark)


def test_matched_sectors_decompose_active_return():
    """Effects sum to the active return when both sides hold every sector."""
    portfolio = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Energy"],
        "weight": [0.6, 0.4],
        "return": [0.10, 0.02],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Energy"],
        "weight": [0.5, 0.5],
        "return": [0.08, 0.03],
    })

    result = _attributor(portfolio, benchmark).brinson_attribution()

    portfolio_return = 0.6 * 0.10 + 0.4 * 0.02
    benchmark_return = 0.5 * 0.08 + 0.5 * 0.03
    assert result.total_active_return == pytest.approx(portfolio_return - benchmark_return)
    assert result.total_active_return == pytest.approx(
        result.allocation_effect + result.selection_effect + result.interaction_effect
    )


def test_benchmark_only_sector_is_pure_allocation():
    """A sector the portfolio does not hold is an allocation call, not selection."""
    portfolio = pl.DataFrame({
        "asset_id": ["A"],
        "sector": ["Tech"],
        "weight": [1.0],
        "return": [0.10],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A", "E"],
        "sector": ["Tech", "Energy"],
        "weight": [0.8, 0.2],
        "return": [0.10, 0.05],
    })

    result = _attributor(portfolio, benchmark).brinson_attribution()
    energy = result.sector_details["Energy"]

    assert energy["selection"] == pytest.approx(0.0)
    assert energy["interaction"] == pytest.approx(0.0)
    assert energy["allocation"] == pytest.approx(-0.2 * 0.05)
    assert energy["total"] == pytest.approx(-0.2 * 0.05)


def test_portfolio_only_sector_is_pure_allocation():
    """The mirror case: a sector absent from the benchmark also lands in allocation."""
    portfolio = pl.DataFrame({
        "asset_id": ["A", "C"],
        "sector": ["Tech", "Crypto"],
        "weight": [0.8, 0.2],
        "return": [0.10, 0.30],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A"],
        "sector": ["Tech"],
        "weight": [1.0],
        "return": [0.10],
    })

    crypto = _attributor(portfolio, benchmark).brinson_attribution().sector_details["Crypto"]

    assert crypto["selection"] == pytest.approx(0.0)
    assert crypto["interaction"] == pytest.approx(0.0)
    assert crypto["allocation"] == pytest.approx(0.2 * 0.30)


def test_net_zero_sector_weight_does_not_produce_nan():
    """A long/short pair netting to zero weight used to divide by zero."""
    portfolio = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Tech"],
        "weight": [1.0, -1.0],
        "return": [0.10, 0.02],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A"],
        "sector": ["Tech"],
        "weight": [1.0],
        "return": [0.05],
    })

    result = _attributor(portfolio, benchmark).brinson_attribution()
    tech = result.sector_details["Tech"]

    assert not math.isnan(result.total_active_return)
    assert all(math.isfinite(v) for v in tech.values())
    assert tech["allocation"] == pytest.approx(-0.05)
    assert result.total_active_return == pytest.approx(-0.05)


def test_sector_held_by_neither_side_contributes_nothing():
    """Rows carrying zero weight on both sides leave every effect at zero."""
    portfolio = pl.DataFrame({
        "asset_id": ["A", "Z"],
        "sector": ["Tech", "Cash"],
        "weight": [1.0, 0.0],
        "return": [0.10, 0.00],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A", "Z"],
        "sector": ["Tech", "Cash"],
        "weight": [1.0, 0.0],
        "return": [0.10, 0.00],
    })

    cash = _attributor(portfolio, benchmark).brinson_attribution().sector_details["Cash"]

    assert cash["allocation"] == pytest.approx(0.0)
    assert cash["selection"] == pytest.approx(0.0)
    assert cash["interaction"] == pytest.approx(0.0)


def test_identical_portfolio_and_benchmark_has_no_active_return():
    frame = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Energy"],
        "weight": [0.5, 0.5],
        "return": [0.10, 0.02],
    })

    result = _attributor(frame, frame.clone()).brinson_attribution()

    assert result.allocation_effect == pytest.approx(0.0)
    assert result.selection_effect == pytest.approx(0.0)
    assert result.interaction_effect == pytest.approx(0.0)
    assert result.total_active_return == pytest.approx(0.0)


def test_selection_effect_isolates_within_sector_picking():
    """Same weights, better picks: everything is selection."""
    portfolio = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Energy"],
        "weight": [0.5, 0.5],
        "return": [0.12, 0.04],
    })
    benchmark = pl.DataFrame({
        "asset_id": ["A", "B"],
        "sector": ["Tech", "Energy"],
        "weight": [0.5, 0.5],
        "return": [0.10, 0.02],
    })

    result = _attributor(portfolio, benchmark).brinson_attribution()

    assert result.allocation_effect == pytest.approx(0.0)
    assert result.interaction_effect == pytest.approx(0.0)
    assert result.selection_effect == pytest.approx(0.5 * 0.02 + 0.5 * 0.02)


def test_benchmark_required():
    frame = pl.DataFrame({
        "asset_id": ["A"],
        "sector": ["Tech"],
        "weight": [1.0],
        "return": [0.10],
    })

    with pytest.raises(ValueError, match="Benchmark"):
        PerformanceAttributor(frame).brinson_attribution()
