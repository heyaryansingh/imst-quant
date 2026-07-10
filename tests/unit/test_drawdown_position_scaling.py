"""Tests for drawdown-based position scaling."""

import pytest

from imst_quant.utils.drawdown_position_scaling import (
    ScalingConfig,
    apply_drawdown_scaling,
    calculate_current_drawdown,
    concave_scale_factor,
    convex_scale_factor,
    drawdown_scaling_report,
    get_scaling_state,
    linear_scale_factor,
    step_scale_factor,
)


class TestCalculateCurrentDrawdown:
    def test_no_drawdown_at_peak(self):
        assert calculate_current_drawdown(100_000, 100_000) == 0.0

    def test_drawdown_below_peak(self):
        dd = calculate_current_drawdown(90_000, 100_000)
        assert dd == pytest.approx(0.10)

    def test_zero_high_water_mark_returns_zero(self):
        assert calculate_current_drawdown(50_000, 0) == 0.0

    def test_equity_above_hwm_clamped_to_zero(self):
        assert calculate_current_drawdown(110_000, 100_000) == 0.0


class TestLinearScaleFactor:
    def test_zero_drawdown_full_scale(self):
        assert linear_scale_factor(0.0) == 1.0

    def test_max_drawdown_returns_floor(self):
        assert linear_scale_factor(0.20, max_drawdown=0.20, floor=0.25) == 0.25

    def test_beyond_max_drawdown_clamped_to_floor(self):
        assert linear_scale_factor(0.50, max_drawdown=0.20, floor=0.25) == 0.25

    def test_midpoint_is_between_floor_and_one(self):
        scale = linear_scale_factor(0.10, max_drawdown=0.20, floor=0.25)
        assert 0.25 < scale < 1.0


class TestConvexConcaveScaleFactor:
    def test_convex_cuts_faster_than_linear_early(self):
        dd = 0.05
        convex = convex_scale_factor(dd, max_drawdown=0.20, floor=0.25)
        linear = linear_scale_factor(dd, max_drawdown=0.20, floor=0.25)
        assert convex < linear

    def test_concave_holds_longer_than_linear_early(self):
        dd = 0.05
        concave = concave_scale_factor(dd, max_drawdown=0.20, floor=0.25)
        linear = linear_scale_factor(dd, max_drawdown=0.20, floor=0.25)
        assert concave > linear

    def test_convex_and_concave_bounded(self):
        for fn in (convex_scale_factor, concave_scale_factor):
            assert fn(0.0) == 1.0
            assert fn(0.20, max_drawdown=0.20, floor=0.25) == 0.25


class TestStepScaleFactor:
    def test_no_drawdown_full_scale(self):
        assert step_scale_factor(0.0) == 1.0

    def test_kill_switch_returns_zero(self):
        assert step_scale_factor(0.35, kill_switch=0.30) == 0.0

    def test_tier_selection(self):
        tiers = [(0.05, 0.80), (0.10, 0.50)]
        assert step_scale_factor(0.07, tiers=tiers, kill_switch=0.30) == 0.80
        assert step_scale_factor(0.12, tiers=tiers, kill_switch=0.30) == 0.50


class TestApplyDrawdownScaling:
    def test_no_drawdown_returns_full_size(self):
        scaled = apply_drawdown_scaling(10_000, equity=100_000, high_water_mark=100_000)
        assert scaled == 10_000

    def test_kill_switch_zeros_position(self):
        config = ScalingConfig(kill_switch=0.30)
        scaled = apply_drawdown_scaling(
            10_000, equity=65_000, high_water_mark=100_000, config=config
        )
        assert scaled == 0.0

    def test_partial_drawdown_reduces_size(self):
        config = ScalingConfig(method="linear", max_drawdown=0.20, floor=0.25, kill_switch=0.30)
        scaled = apply_drawdown_scaling(
            10_000, equity=90_000, high_water_mark=100_000, config=config
        )
        assert 0 < scaled < 10_000

    def test_step_method_dispatches_correctly(self):
        config = ScalingConfig(method="step", kill_switch=0.30)
        scaled = apply_drawdown_scaling(
            10_000, equity=94_000, high_water_mark=100_000, config=config
        )
        assert scaled == 8_000  # 6% dd -> 0.80 tier


class TestGetScalingState:
    def test_state_reflects_halted_flag(self):
        config = ScalingConfig(kill_switch=0.10)
        state = get_scaling_state(equity=85_000, high_water_mark=100_000, config=config)
        assert state.is_halted is True
        assert state.scale_factor == 0.0

    def test_state_not_halted_below_kill_switch(self):
        config = ScalingConfig(kill_switch=0.30)
        state = get_scaling_state(equity=95_000, high_water_mark=100_000, config=config)
        assert state.is_halted is False
        assert state.drawdown_pct == pytest.approx(0.05)


class TestDrawdownScalingReport:
    def test_report_structure_and_aggregates(self):
        report = drawdown_scaling_report(
            equity=90_000,
            high_water_mark=100_000,
            base_position_sizes={"AAPL": 5_000, "GOOG": 5_000},
        )
        assert "state" in report and "positions" in report and "aggregate" in report
        assert report["aggregate"]["total_base_exposure"] == 10_000
        assert report["aggregate"]["total_scaled_exposure"] < 10_000
        assert 0 <= report["aggregate"]["reduction_pct"] <= 1

    def test_report_no_positions_no_division_error(self):
        report = drawdown_scaling_report(
            equity=100_000, high_water_mark=100_000, base_position_sizes={}
        )
        assert report["aggregate"]["reduction_pct"] == 0.0
