"""Tests for tax lot accounting."""

from datetime import date

import pytest

from imst_quant.utils.tax_lots import (
    InsufficientLotsError,
    LotSale,
    TaxLotTracker,
    compare_methods,
    find_harvestable_lots,
    sales_to_polars,
    summarize_sales,
)


@pytest.fixture
def two_lots() -> TaxLotTracker:
    """Tracker holding a cheap old lot and an expensive newer lot."""
    tracker = TaxLotTracker(method="fifo")
    tracker.buy("AAPL", 100, 150.0, "2024-01-10")
    tracker.buy("AAPL", 100, 170.0, "2024-06-10")
    return tracker


def test_rejects_unknown_matching_method():
    with pytest.raises(ValueError, match="Unknown matching method"):
        TaxLotTracker(method="random")


@pytest.mark.parametrize("method", ["fifo", "lifo", "hifo", "lofo"])
def test_accepts_every_documented_method(method):
    assert TaxLotTracker(method=method).method == method


def test_method_is_case_insensitive():
    assert TaxLotTracker(method="FIFO").method == "fifo"


@pytest.mark.parametrize(
    "quantity,price",
    [(0, 10.0), (-5, 10.0), (10, 0.0), (10, -10.0)],
)
def test_buy_rejects_non_positive_inputs(quantity, price):
    tracker = TaxLotTracker()
    with pytest.raises(ValueError):
        tracker.buy("AAPL", quantity, price, "2024-01-01")


def test_buy_rejects_negative_commission():
    tracker = TaxLotTracker()
    with pytest.raises(ValueError, match="Commission"):
        tracker.buy("AAPL", 10, 100.0, "2024-01-01", commission=-1.0)


def test_commission_is_amortized_into_the_cost_basis():
    tracker = TaxLotTracker()
    lot = tracker.buy("AAPL", 100, 150.0, "2024-01-01", commission=50.0)
    assert lot.cost_per_share == pytest.approx(150.5)
    assert tracker.cost_basis("AAPL") == pytest.approx(15_050.0)


def test_fifo_consumes_the_oldest_lot_first(two_lots):
    sales = two_lots.sell("AAPL", 150, 180.0, "2024-08-10")

    assert [sale.quantity for sale in sales] == [100, 50]
    assert sales[0].cost_per_share == pytest.approx(150.0)
    assert sales[1].cost_per_share == pytest.approx(170.0)
    assert summarize_sales(sales).realized_pnl == pytest.approx(3500.0)


def test_lifo_consumes_the_newest_lot_first():
    tracker = TaxLotTracker(method="lifo")
    tracker.buy("AAPL", 100, 150.0, "2024-01-10")
    tracker.buy("AAPL", 100, 170.0, "2024-06-10")

    sales = tracker.sell("AAPL", 150, 180.0, "2024-08-10")

    assert sales[0].cost_per_share == pytest.approx(170.0)
    assert sales[1].cost_per_share == pytest.approx(150.0)
    assert summarize_sales(sales).realized_pnl == pytest.approx(2500.0)


def test_hifo_defers_more_gain_than_lofo():
    trades = [
        {"symbol": "AAPL", "quantity": 100, "price": 150.0, "date": "2024-01-10", "side": "buy"},
        {"symbol": "AAPL", "quantity": 100, "price": 170.0, "date": "2024-06-10", "side": "buy"},
        {"symbol": "AAPL", "quantity": 100, "price": 180.0, "date": "2024-08-10", "side": "sell"},
    ]
    results = compare_methods(trades)

    assert results["hifo"].realized_pnl == pytest.approx(1000.0)
    assert results["lofo"].realized_pnl == pytest.approx(3000.0)
    assert results["hifo"].realized_pnl < results["lofo"].realized_pnl


def test_compare_methods_rejects_an_unknown_side():
    trades = [
        {"symbol": "AAPL", "quantity": 10, "price": 10.0, "date": "2024-01-01", "side": "hold"}
    ]
    with pytest.raises(ValueError, match="Unknown trade side"):
        compare_methods(trades, methods=["fifo"])


def test_selling_more_than_held_is_rejected(two_lots):
    with pytest.raises(InsufficientLotsError, match="long only"):
        two_lots.sell("AAPL", 500, 180.0, "2024-08-10")


def test_selling_an_unheld_symbol_is_rejected():
    tracker = TaxLotTracker()
    with pytest.raises(InsufficientLotsError):
        tracker.sell("MSFT", 1, 100.0, "2024-08-10")


def test_a_rejected_sell_leaves_the_lots_untouched(two_lots):
    with pytest.raises(InsufficientLotsError):
        two_lots.sell("AAPL", 500, 180.0, "2024-08-10")

    assert two_lots.shares_held("AAPL") == 200
    assert two_lots.sales() == []


def test_partial_sells_leave_the_remainder_open(two_lots):
    two_lots.sell("AAPL", 150, 180.0, "2024-08-10")

    assert two_lots.shares_held("AAPL") == 50
    assert two_lots.cost_basis("AAPL") == pytest.approx(8500.0)
    assert two_lots.average_cost("AAPL") == pytest.approx(170.0)
    assert len(two_lots.open_lots("AAPL")) == 1


def test_repeated_fifo_sells_stay_in_purchase_order():
    tracker = TaxLotTracker(method="fifo")
    tracker.buy("AAPL", 10, 100.0, "2024-01-01")
    tracker.buy("AAPL", 10, 110.0, "2024-02-01")
    tracker.buy("AAPL", 10, 120.0, "2024-03-01")

    first = tracker.sell("AAPL", 15, 130.0, "2024-04-01")
    second = tracker.sell("AAPL", 10, 130.0, "2024-05-01")

    assert [sale.cost_per_share for sale in first] == [100.0, 110.0]
    assert [sale.cost_per_share for sale in second] == [110.0, 120.0]


def test_average_cost_is_zero_when_flat():
    tracker = TaxLotTracker()
    assert tracker.average_cost("AAPL") == 0.0
    assert tracker.shares_held("AAPL") == 0


def test_sale_commission_reduces_net_proceeds():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 100, 100.0, "2024-01-01")

    sale = tracker.sell("AAPL", 100, 110.0, "2024-02-01", commission=100.0)[0]

    assert sale.proceeds == pytest.approx(10_900.0)
    assert sale.realized_pnl == pytest.approx(900.0)


def test_holding_period_splits_short_and_long_term():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 10, 100.0, "2022-01-01")
    tracker.buy("MSFT", 10, 100.0, "2024-01-01")
    tracker.sell("AAPL", 10, 120.0, "2024-06-01")
    tracker.sell("MSFT", 10, 90.0, "2024-06-01")

    summary = tracker.summarize()
    assert summary.long_term_pnl == pytest.approx(200.0)
    assert summary.short_term_pnl == pytest.approx(-100.0)
    assert summary.realized_pnl == pytest.approx(100.0)
    assert summary.realized_gains == pytest.approx(200.0)
    assert summary.realized_losses == pytest.approx(-100.0)


def test_exactly_one_year_counts_as_long_term():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 1, 100.0, "2023-01-01")
    sale = tracker.sell("AAPL", 1, 110.0, "2024-01-01")[0]

    assert sale.holding_days == 365
    assert sale.is_long_term


def test_one_day_short_of_a_year_is_short_term():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 1, 100.0, "2023-01-02")
    sale = tracker.sell("AAPL", 1, 110.0, "2024-01-01")[0]

    assert sale.holding_days == 364
    assert not sale.is_long_term


def test_summary_of_a_symbol_excludes_other_symbols():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 10, 100.0, "2024-01-01")
    tracker.buy("MSFT", 10, 100.0, "2024-01-01")
    tracker.sell("AAPL", 10, 110.0, "2024-02-01")
    tracker.sell("MSFT", 10, 200.0, "2024-02-01")

    assert tracker.summarize("AAPL").realized_pnl == pytest.approx(100.0)
    assert tracker.summarize().realized_pnl == pytest.approx(1100.0)


def test_summary_of_no_sales_is_all_zero():
    summary = summarize_sales([])
    assert summary.realized_pnl == 0.0
    assert summary.sale_count == 0
    assert summary.avg_holding_days == 0.0


def test_avg_holding_days_is_quantity_weighted():
    tracker = TaxLotTracker(method="fifo")
    tracker.buy("AAPL", 90, 100.0, "2024-01-01")  # held 100 days
    tracker.buy("AAPL", 10, 100.0, "2024-04-01")  # held 9 days
    tracker.sell("AAPL", 100, 110.0, "2024-04-10")

    summary = tracker.summarize()
    assert summary.quantity_sold == 100
    assert summary.avg_holding_days == pytest.approx((90 * 100 + 10 * 9) / 100)


def test_unrealized_pnl_skips_symbols_without_a_price(two_lots):
    two_lots.buy("MSFT", 10, 100.0, "2024-01-01")

    # AAPL only: 100 @ 150 and 100 @ 170 marked at 160.
    assert two_lots.unrealized_pnl({"AAPL": 160.0}) == pytest.approx(0.0)
    assert two_lots.unrealized_pnl({"AAPL": 160.0, "MSFT": 120.0}) == pytest.approx(200.0)


def test_find_harvestable_lots_orders_by_largest_loss():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 10, 200.0, "2024-01-01")
    tracker.buy("MSFT", 10, 120.0, "2024-01-01")
    tracker.buy("NVDA", 10, 50.0, "2024-01-01")

    harvestable = find_harvestable_lots(
        tracker, {"AAPL": 100.0, "MSFT": 110.0, "NVDA": 90.0}
    )

    assert [lot.symbol for lot in harvestable] == ["AAPL", "MSFT"]


def test_find_harvestable_lots_respects_the_minimum_loss():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 10, 200.0, "2024-01-01")
    tracker.buy("MSFT", 10, 120.0, "2024-01-01")

    harvestable = find_harvestable_lots(
        tracker, {"AAPL": 100.0, "MSFT": 110.0}, min_loss=500.0
    )

    assert [lot.symbol for lot in harvestable] == ["AAPL"]


def test_find_harvestable_lots_rejects_a_negative_threshold():
    with pytest.raises(ValueError, match="min_loss"):
        find_harvestable_lots(TaxLotTracker(), {}, min_loss=-1.0)


def test_dates_accept_strings_and_date_objects():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 1, 100.0, date(2024, 1, 1))
    sale = tracker.sell("AAPL", 1, 110.0, "2024-03-01T15:30:00")[0]

    assert sale.acquired == date(2024, 1, 1)
    assert sale.disposed == date(2024, 3, 1)


def test_unsupported_date_type_is_rejected():
    tracker = TaxLotTracker()
    with pytest.raises(TypeError):
        tracker.buy("AAPL", 1, 100.0, 20240101)


def test_to_polars_has_one_row_per_disposal(two_lots):
    two_lots.sell("AAPL", 150, 180.0, "2024-08-10")

    frame = two_lots.to_polars()
    assert frame.height == 2
    assert frame["realized_pnl"].sum() == pytest.approx(3500.0)


def test_to_polars_on_no_sales_keeps_the_schema():
    frame = sales_to_polars([])
    assert frame.height == 0
    assert "realized_pnl" in frame.columns


def test_lot_sale_properties_are_self_consistent():
    sale = LotSale(
        symbol="AAPL",
        quantity=10,
        cost_per_share=100.0,
        proceeds_per_share=115.0,
        acquired=date(2024, 1, 1),
        disposed=date(2024, 3, 1),
        holding_days=60,
        lot_id=1,
    )
    assert sale.cost_basis == pytest.approx(1000.0)
    assert sale.proceeds == pytest.approx(1150.0)
    assert sale.realized_pnl == pytest.approx(150.0)
    assert not sale.is_long_term
