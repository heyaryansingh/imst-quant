"""Tests for position accounting in the paper trading simulator.

Focus on the cases where a fill changes the sign of a position or closes a
short, which are the paths where realized P&L is easiest to get wrong.
"""

import pytest

from imst_quant.paper_trading.simulator import (
    PaperTradingSimulator,
    SimulatorConfig,
)


@pytest.fixture
def sim() -> PaperTradingSimulator:
    """Simulator with no slippage or commission for exact arithmetic."""
    config = SimulatorConfig(
        slippage_bps=0.0,
        commission_per_share=0.0,
        min_commission=0.0,
        allow_short_selling=True,
    )
    return PaperTradingSimulator(initial_cash=100_000.0, config=config)


def test_long_partial_sell_realizes_pnl_and_keeps_cost_basis(sim):
    sim.submit_order("AAPL", 100, "buy", current_price=100.0)
    sim.submit_order("AAPL", 40, "sell", current_price=110.0)

    position = sim.positions["AAPL"]
    assert position.quantity == 60
    assert position.avg_cost == pytest.approx(100.0)
    assert position.realized_pnl == pytest.approx(400.0)


def test_closing_a_long_zeroes_the_cost_basis(sim):
    sim.submit_order("AAPL", 50, "buy", current_price=20.0)
    sim.submit_order("AAPL", 50, "sell", current_price=25.0)

    position = sim.positions["AAPL"]
    assert position.quantity == 0
    assert position.avg_cost == pytest.approx(0.0)
    assert position.realized_pnl == pytest.approx(250.0)


def test_covering_a_short_at_a_lower_price_realizes_a_profit(sim):
    sim.submit_order("AAPL", 100, "sell", current_price=100.0)
    assert sim.positions["AAPL"].avg_cost == pytest.approx(100.0)

    sim.submit_order("AAPL", 100, "buy", current_price=90.0)

    position = sim.positions["AAPL"]
    assert position.quantity == 0
    assert position.realized_pnl == pytest.approx(1000.0)


def test_covering_a_short_at_a_higher_price_realizes_a_loss(sim):
    sim.submit_order("AAPL", 10, "sell", current_price=50.0)
    sim.submit_order("AAPL", 10, "buy", current_price=55.0)

    assert sim.positions["AAPL"].realized_pnl == pytest.approx(-50.0)


def test_adding_to_a_short_rolls_the_average_cost(sim):
    sim.submit_order("AAPL", 100, "sell", current_price=100.0)
    sim.submit_order("AAPL", 100, "sell", current_price=120.0)

    position = sim.positions["AAPL"]
    assert position.quantity == -200
    assert position.avg_cost == pytest.approx(110.0)
    assert position.realized_pnl == pytest.approx(0.0)


def test_flipping_long_to_short_resets_basis_to_the_fill_price(sim):
    sim.submit_order("AAPL", 100, "buy", current_price=100.0)
    sim.submit_order("AAPL", 150, "sell", current_price=110.0)

    position = sim.positions["AAPL"]
    assert position.quantity == -50
    # Only the 100 long shares are closed; the residual short opens at 110.
    assert position.realized_pnl == pytest.approx(1000.0)
    assert position.avg_cost == pytest.approx(110.0)


def test_flipping_short_to_long_resets_basis_to_the_fill_price(sim):
    sim.submit_order("AAPL", 100, "sell", current_price=100.0)
    sim.submit_order("AAPL", 150, "buy", current_price=90.0)

    position = sim.positions["AAPL"]
    assert position.quantity == 50
    assert position.realized_pnl == pytest.approx(1000.0)
    assert position.avg_cost == pytest.approx(90.0)


def test_short_unrealized_pnl_is_positive_when_price_falls(sim):
    sim.submit_order("AAPL", 100, "sell", current_price=100.0)

    summary = sim.get_account_summary({"AAPL": 90.0})
    assert summary["total_unrealized_pnl"] == pytest.approx(1000.0)


def test_round_trip_realized_pnl_matches_the_cash_change(sim):
    starting_cash = sim.cash
    sim.submit_order("AAPL", 25, "buy", current_price=40.0)
    sim.submit_order("AAPL", 25, "sell", current_price=44.0)

    position = sim.positions["AAPL"]
    assert sim.cash - starting_cash == pytest.approx(position.realized_pnl)
