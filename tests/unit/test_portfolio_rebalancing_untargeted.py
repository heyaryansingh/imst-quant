from datetime import datetime

import pytest

from imst_quant.utils.portfolio_rebalancing import PortfolioRebalancer


def test_untargeted_holding_counts_and_is_sold():
    r = PortfolioRebalancer({"A": 0.5, "B": 0.5}, min_trade_value=1.0)
    positions = {"A": 10.0, "B": 10.0, "OLD": 20.0}
    prices = {"A": 10.0, "B": 10.0, "OLD": 10.0}

    weights = r.calculate_current_weights(positions, prices)
    assert weights["OLD"] == pytest.approx(0.5)
    assert r.needs_rebalance_threshold(weights)

    trades, _ = r.execute_rebalance(positions, prices, datetime(2026, 1, 1))
    by_sym = {t.symbol: t for t in trades}
    assert by_sym["OLD"].trade_shares == pytest.approx(-20.0)
    assert by_sym["A"].target_value == pytest.approx(200.0)
