"""Tests for portfolio rebalancing signal generation."""

from datetime import datetime, timedelta

import pytest

from imst_quant.utils.rebalance_signals import (
    calculate_drift,
    calendar_rebalance_signal,
    generate_rebalance_orders,
    rebalance_summary,
    threshold_rebalance_signal,
    volatility_adjusted_threshold,
)


class TestCalculateDrift:
    def test_no_drift_when_weights_match(self):
        drift = calculate_drift({"AAPL": 0.5}, {"AAPL": 0.5})
        assert drift[0].absolute_drift == 0.0
        assert drift[0].relative_drift == 0.0

    def test_positive_drift_when_overweight(self):
        drift = calculate_drift({"AAPL": 0.4}, {"AAPL": 0.3})
        assert drift[0].absolute_drift == pytest.approx(0.1)
        assert drift[0].relative_drift == pytest.approx(1 / 3)

    def test_asset_missing_from_target_uses_zero(self):
        drift = calculate_drift({"AAPL": 0.2}, {})
        assert drift[0].target_weight == 0.0
        assert drift[0].relative_drift == float("inf")

    def test_covers_union_of_assets(self):
        drift = calculate_drift({"AAPL": 0.5}, {"AAPL": 0.5, "GOOG": 0.5})
        assets = {d.asset for d in drift}
        assert assets == {"AAPL", "GOOG"}


class TestThresholdRebalanceSignal:
    def test_within_thresholds_no_trigger(self):
        trigger, reason = threshold_rebalance_signal(
            {"AAPL": 0.51}, {"AAPL": 0.50}
        )
        assert trigger is False
        assert "within thresholds" in reason

    def test_absolute_threshold_exceeded_triggers(self):
        trigger, reason = threshold_rebalance_signal(
            {"AAPL": 0.60}, {"AAPL": 0.50}, abs_threshold=0.05
        )
        assert trigger is True
        assert "AAPL" in reason

    def test_relative_threshold_exceeded_triggers(self):
        trigger, reason = threshold_rebalance_signal(
            {"AAPL": 0.02}, {"AAPL": 0.01}, abs_threshold=0.5, rel_threshold=0.25
        )
        assert trigger is True


class TestCalendarRebalanceSignal:
    def test_due_when_interval_elapsed(self):
        last = datetime(2024, 1, 1)
        now = datetime(2024, 2, 15)
        trigger, reason = calendar_rebalance_signal(last, now, frequency="monthly")
        assert trigger is True
        assert "due" in reason

    def test_not_due_within_interval(self):
        last = datetime(2024, 1, 1)
        now = datetime(2024, 1, 5)
        trigger, reason = calendar_rebalance_signal(last, now, frequency="monthly")
        assert trigger is False
        assert "Next" in reason

    def test_weekly_frequency(self):
        last = datetime(2024, 1, 1)
        now = datetime(2024, 1, 10)
        trigger, _ = calendar_rebalance_signal(last, now, frequency="weekly")
        assert trigger is True


class TestVolatilityAdjustedThreshold:
    def test_widens_in_high_volatility(self):
        adjusted = volatility_adjusted_threshold(
            base_threshold=0.05, recent_volatility=0.40, long_term_volatility=0.20
        )
        assert adjusted > 0.05

    def test_tightens_in_low_volatility(self):
        adjusted = volatility_adjusted_threshold(
            base_threshold=0.05, recent_volatility=0.10, long_term_volatility=0.20
        )
        assert adjusted < 0.05

    def test_zero_long_term_vol_returns_base(self):
        adjusted = volatility_adjusted_threshold(
            base_threshold=0.05, recent_volatility=0.10, long_term_volatility=0.0
        )
        assert adjusted == 0.05

    def test_adjustment_bounded(self):
        adjusted = volatility_adjusted_threshold(
            base_threshold=0.05,
            recent_volatility=10.0,
            long_term_volatility=0.01,
            sensitivity=1.0,
        )
        assert adjusted <= 0.10  # capped at 2x base


class TestGenerateRebalanceOrders:
    def test_generates_buy_and_sell_orders(self):
        orders = generate_rebalance_orders(
            {"AAPL": 0.60, "GOOG": 0.40},
            {"AAPL": 0.50, "GOOG": 0.50},
            portfolio_value=100_000,
        )
        sides = {o.asset: o.side for o in orders}
        assert sides["AAPL"] == "SELL"
        assert sides["GOOG"] == "BUY"

    def test_filters_below_min_trade_size(self):
        orders = generate_rebalance_orders(
            {"AAPL": 0.501}, {"AAPL": 0.500}, portfolio_value=100_000, min_trade_size=500
        )
        assert orders == []

    def test_sorted_by_dollar_amount_descending(self):
        orders = generate_rebalance_orders(
            {"AAPL": 0.70, "GOOG": 0.10, "MSFT": 0.20},
            {"AAPL": 0.33, "GOOG": 0.33, "MSFT": 0.34},
            portfolio_value=100_000,
        )
        amounts = [o.dollar_amount for o in orders]
        assert amounts == sorted(amounts, reverse=True)


class TestRebalanceSummary:
    def test_full_summary_structure(self):
        summary = rebalance_summary(
            {"AAPL": 0.60, "GOOG": 0.40},
            {"AAPL": 0.50, "GOOG": 0.50},
            portfolio_value=100_000,
        )
        assert summary.trigger is True
        assert summary.max_absolute_drift == pytest.approx(0.10)
        assert len(summary.orders) == 2
        assert summary.estimated_cost > 0

    def test_no_trigger_when_balanced(self):
        summary = rebalance_summary({"AAPL": 0.50}, {"AAPL": 0.50})
        assert summary.trigger is False
        assert summary.orders == []
        assert summary.estimated_cost == 0
