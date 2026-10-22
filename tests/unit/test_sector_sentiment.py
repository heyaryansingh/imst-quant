"""Tests for imst_quant.utils.sector_sentiment."""

import math

import polars as pl
import pytest

from imst_quant.utils.sector_sentiment import (
    aggregate_sector_sentiment,
    cross_sector_dispersion,
)


def _df():
    return pl.DataFrame({
        "ticker": ["AAPL", "MSFT", "JPM", "BAC", "XOM", "ZZZZ"],
        "sentiment": [0.8, 0.6, -0.2, float("nan"), 0.1, 0.5],
    })


def test_aggregate_skips_non_finite_sentiment():
    sectors = {s.sector: s for s in aggregate_sector_sentiment(_df())}
    fin = sectors["Financials"]
    assert fin.avg_sentiment == pytest.approx(-0.2)
    assert fin.num_assets == 1
    assert fin.bearish_pct == 100.0
    assert sectors["Technology"].avg_sentiment == pytest.approx(0.7)
    assert "ZZZZ" not in {s.strongest_ticker for s in sectors.values()}


def test_dispersion_is_finite_with_nan_input():
    result = cross_sector_dispersion(_df())
    assert math.isfinite(result["dispersion"])
    assert result["top_sector"] == "Technology"
    assert result["bottom_sector"] == "Financials"
