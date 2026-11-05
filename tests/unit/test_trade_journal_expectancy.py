import pytest

from imst_quant.utils.trade_journal import TradeEntry, TradeJournal


def _closed(pnl):
    t = TradeEntry(symbol="X", direction="long", entry_price=100.0, quantity=1.0)
    t.exit_price = 100.0 + pnl
    t.pnl = pnl
    t.status = "closed"
    return t


def test_expectancy_negative_when_all_trades_lose():
    stats = TradeJournal().get_statistics([_closed(-10.0), _closed(-20.0)])
    assert stats.expectancy == pytest.approx(-15.0)


def test_expectancy_ignores_breakeven_as_loss():
    stats = TradeJournal().get_statistics([_closed(30.0), _closed(-10.0), _closed(0.0)])
    assert stats.expectancy == pytest.approx(20.0 / 3)
