"""Tests for turnover_summary's annual cost estimate.

The estimate multiplied an already-unitless turnover multiple by an extra 100,
reporting 1000 bps of cost drag for a portfolio that actually loses 10.
"""

import pytest

from imst_quant.utils.portfolio_turnover import (
    COST_PER_UNIT_TURNOVER_BPS,
    turnover_summary,
)


def _history_with_monthly_turnover(turnover, periods=4):
    """Alternate two weights so each step has a known one-way turnover."""
    a = 0.5
    b = 0.5 + turnover
    return [{"X": a, "Y": 1 - a} if i % 2 == 0 else {"X": b, "Y": 1 - b}
            for i in range(periods)]


def test_cost_estimate_is_bps_not_percent():
    """10% monthly one-way turnover annualizes to 1.2x, costing 6 bps."""
    summary = turnover_summary(_history_with_monthly_turnover(0.10))

    assert summary.avg_monthly_turnover == pytest.approx(0.10)
    assert summary.annualized_turnover == pytest.approx(1.2)
    assert summary.estimated_annual_cost_bps == pytest.approx(6.0)


def test_cost_estimate_tracks_the_named_constant():
    summary = turnover_summary(_history_with_monthly_turnover(0.05))

    assert summary.estimated_annual_cost_bps == pytest.approx(
        summary.annualized_turnover * COST_PER_UNIT_TURNOVER_BPS
    )


def test_cost_estimate_stays_under_a_full_percent_for_normal_turnover():
    """A 200%-a-year book should not be told it loses 10% to trading."""
    summary = turnover_summary(_history_with_monthly_turnover(0.1667))

    assert summary.annualized_turnover == pytest.approx(2.0, abs=0.01)
    assert summary.estimated_annual_cost_bps < 100


def test_insufficient_history_reports_no_cost():
    summary = turnover_summary([{"X": 1.0}])

    assert summary.estimated_annual_cost_bps == 0.0
    assert summary.turnover_trend == "insufficient_data"
