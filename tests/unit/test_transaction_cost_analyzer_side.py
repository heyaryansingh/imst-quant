import pytest

from imst_quant.utils.transaction_cost_analyzer import TransactionCostAnalyzer


def _timing(side, exec_px):
    a = TransactionCostAnalyzer(commission_rate=0.0, impact_model="none")
    return a.analyze_trade(100.0, exec_px, volume=10, adv=1000, spread_bps=0, side=side).timing_cost


def test_timing_cost_signed_by_side():
    assert _timing("buy", 101.0) == pytest.approx(10.0)   # paid up
    assert _timing("buy", 99.0) == pytest.approx(-10.0)   # improvement
    assert _timing("sell", 99.0) == pytest.approx(10.0)   # sold lower
    assert _timing("sell", 101.0) == pytest.approx(-10.0)  # improvement


def test_unknown_side_rejected():
    with pytest.raises(ValueError):
        _timing("short", 100.0)
