"""Tests for covariance shrinkage estimators and drawdown circuit breaker.

Covers:
- Ledoit-Wolf, OAS, identity, and custom target shrinkage
- Shrinkage intensity bounds and condition number improvement
- Circuit breaker threshold escalation, cooldown, and reset
- Simulation over drawdown series
"""

import numpy as np
import pytest

from imst_quant.utils.covariance_shrinkage import (
    ShrinkageResult,
    compare_shrinkage_methods,
    custom_target_shrinkage,
    identity_shrinkage,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
)
from imst_quant.utils.drawdown_circuit_breaker import (
    CircuitAction,
    DrawdownCircuitBreaker,
    simulate_circuit_breaker,
)


# ---- Covariance Shrinkage Tests ----


class TestLedoitWolfShrinkage:
    """Tests for Ledoit-Wolf shrinkage estimator."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.returns = self.rng.normal(0, 0.02, (100, 5))

    def test_returns_shrinkage_result(self):
        result = ledoit_wolf_shrinkage(self.returns)
        assert isinstance(result, ShrinkageResult)

    def test_covariance_shape(self):
        result = ledoit_wolf_shrinkage(self.returns)
        assert result.covariance.shape == (5, 5)

    def test_covariance_symmetric(self):
        result = ledoit_wolf_shrinkage(self.returns)
        np.testing.assert_allclose(
            result.covariance, result.covariance.T, atol=1e-12
        )

    def test_covariance_positive_semidefinite(self):
        result = ledoit_wolf_shrinkage(self.returns)
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues >= -1e-10)

    def test_shrinkage_intensity_bounded(self):
        result = ledoit_wolf_shrinkage(self.returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_condition_number_improves(self):
        # Use a near-singular case: many assets, few observations
        returns = self.rng.normal(0, 0.02, (15, 10))
        result = ledoit_wolf_shrinkage(returns)
        assert result.condition_number_after <= result.condition_number_before + 1e-6

    def test_single_asset(self):
        returns = self.rng.normal(0, 0.02, (50, 1))
        result = ledoit_wolf_shrinkage(returns)
        assert result.covariance.shape == (1, 1)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            ledoit_wolf_shrinkage(np.array([[1.0]]))  # Only 1 observation

    def test_assume_centered(self):
        centered = self.returns - self.returns.mean(axis=0)
        r1 = ledoit_wolf_shrinkage(centered, assume_centered=True)
        r2 = ledoit_wolf_shrinkage(centered, assume_centered=False)
        # Both should be close since data is already centered
        np.testing.assert_allclose(
            r1.covariance, r2.covariance, atol=1e-6
        )


class TestOASShrinkage:
    """Tests for Oracle Approximating Shrinkage."""

    def setup_method(self):
        self.rng = np.random.default_rng(123)
        self.returns = self.rng.normal(0, 0.02, (100, 5))

    def test_returns_shrinkage_result(self):
        result = oas_shrinkage(self.returns)
        assert isinstance(result, ShrinkageResult)

    def test_shrinkage_intensity_bounded(self):
        result = oas_shrinkage(self.returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_target_is_scaled_identity(self):
        result = oas_shrinkage(self.returns)
        # Target should be diagonal
        off_diag = result.target - np.diag(np.diag(result.target))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)
        # Diagonal values should be equal
        diag_vals = np.diag(result.target)
        np.testing.assert_allclose(diag_vals, diag_vals[0], atol=1e-15)

    def test_condition_improvement_high_dim(self):
        # More assets than observations
        returns = self.rng.normal(0, 0.02, (10, 20))
        result = oas_shrinkage(returns)
        # Shrunk matrix should have finite condition number
        assert np.isfinite(result.condition_number_after)


class TestIdentityShrinkage:
    """Tests for identity shrinkage."""

    def setup_method(self):
        self.rng = np.random.default_rng(7)
        self.returns = self.rng.normal(0, 0.02, (100, 5))

    def test_fixed_intensity(self):
        result = identity_shrinkage(self.returns, shrinkage_intensity=0.3)
        assert result.shrinkage_intensity == 0.3

    def test_intensity_clamped(self):
        result = identity_shrinkage(self.returns, shrinkage_intensity=1.5)
        assert result.shrinkage_intensity == 1.0

    def test_zero_intensity_equals_sample(self):
        result = identity_shrinkage(self.returns, shrinkage_intensity=0.0)
        np.testing.assert_allclose(
            result.covariance, result.sample_covariance, atol=1e-12
        )

    def test_auto_intensity(self):
        result = identity_shrinkage(self.returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0


class TestCustomTargetShrinkage:
    """Tests for custom target shrinkage."""

    def setup_method(self):
        self.rng = np.random.default_rng(99)
        self.returns = self.rng.normal(0, 0.02, (100, 5))

    def test_with_identity_target(self):
        target = np.eye(5) * 0.0004  # ~2% daily vol squared
        result = custom_target_shrinkage(self.returns, target, 0.5)
        assert result.covariance.shape == (5, 5)

    def test_mismatched_target_raises(self):
        target = np.eye(3)
        with pytest.raises(ValueError, match="doesn't match"):
            custom_target_shrinkage(self.returns, target, 0.5)


class TestCompareShrinkageMethods:
    """Test comparison utility."""

    def test_returns_all_methods(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (100, 5))
        results = compare_shrinkage_methods(returns)
        assert set(results.keys()) == {"ledoit_wolf", "oas", "identity"}
        for name, res in results.items():
            assert isinstance(res, ShrinkageResult), f"{name} failed"


# ---- Drawdown Circuit Breaker Tests ----


class TestCircuitBreakerCheck:
    """Test stateless check method."""

    def setup_method(self):
        self.breaker = DrawdownCircuitBreaker(
            warning_threshold=0.05,
            reduce_threshold=0.10,
            hedge_threshold=0.15,
            halt_threshold=0.20,
        )

    def test_normal(self):
        assert self.breaker.check(0.02) == CircuitAction.NORMAL

    def test_warning(self):
        assert self.breaker.check(0.06) == CircuitAction.WARNING

    def test_reduce(self):
        assert self.breaker.check(0.12) == CircuitAction.REDUCE

    def test_hedge(self):
        assert self.breaker.check(0.16) == CircuitAction.HEDGE

    def test_halt(self):
        assert self.breaker.check(0.25) == CircuitAction.HALT

    def test_exact_threshold_warning(self):
        assert self.breaker.check(0.05) == CircuitAction.WARNING

    def test_exact_threshold_halt(self):
        assert self.breaker.check(0.20) == CircuitAction.HALT


class TestCircuitBreakerUpdate:
    """Test stateful update method."""

    def setup_method(self):
        self.breaker = DrawdownCircuitBreaker(
            warning_threshold=0.05,
            reduce_threshold=0.10,
            hedge_threshold=0.15,
            halt_threshold=0.20,
            cooldown_periods=3,
            recovery_threshold=0.04,
        )

    def test_normal_state(self):
        state = self.breaker.update(0.02)
        assert state.current_action == CircuitAction.NORMAL
        assert state.position_scale == 1.0
        assert not state.is_halted

    def test_reduce_scales_position(self):
        state = self.breaker.update(0.12)
        assert state.current_action == CircuitAction.REDUCE
        assert state.position_scale == 0.5

    def test_halt_triggers_cooldown(self):
        state = self.breaker.update(0.25)
        assert state.is_halted
        assert state.position_scale == 0.0
        assert state.cooldown_remaining == 3

    def test_cooldown_countdown(self):
        self.breaker.update(0.25)  # triggers halt
        s1 = self.breaker.update(0.15)  # cooldown period 1
        assert s1.cooldown_remaining == 2
        assert s1.is_halted

        s2 = self.breaker.update(0.10)  # cooldown period 2
        assert s2.cooldown_remaining == 1

        s3 = self.breaker.update(0.05)  # cooldown period 3
        assert s3.cooldown_remaining == 0

    def test_recovery_after_cooldown(self):
        self.breaker.update(0.25)  # halt
        self.breaker.update(0.10)  # cooldown 1
        self.breaker.update(0.08)  # cooldown 2
        self.breaker.update(0.05)  # cooldown 3 (remaining=0)
        # Now cooldown expired, but dd=0.05 > recovery=0.04, still halted
        s = self.breaker.update(0.05)
        assert s.is_halted

        # Recover below threshold
        s = self.breaker.update(0.03)
        assert not s.is_halted
        assert s.current_action == CircuitAction.NORMAL

    def test_events_recorded(self):
        self.breaker.update(0.02)  # normal (no event for initial normal)
        self.breaker.update(0.06)  # warning
        self.breaker.update(0.12)  # reduce
        events = self.breaker.events
        assert len(events) == 2
        assert events[0].action == CircuitAction.WARNING
        assert events[1].action == CircuitAction.REDUCE

    def test_peak_drawdown_tracked(self):
        self.breaker.update(0.05)
        self.breaker.update(0.15)
        state = self.breaker.update(0.08)
        assert state.peak_drawdown == 0.15

    def test_trigger_count(self):
        self.breaker.update(0.06)  # warning
        self.breaker.update(0.12)  # reduce
        self.breaker.update(0.06)  # back to warning
        counts = self.breaker.trigger_count
        assert counts.get("warning", 0) == 2
        assert counts.get("reduce", 0) == 1


class TestCircuitBreakerReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        breaker = DrawdownCircuitBreaker()
        breaker.update(0.25)
        breaker.reset()
        state = breaker.update(0.01)
        assert state.current_action == CircuitAction.NORMAL
        assert state.peak_drawdown == 0.01
        assert len(state.events) == 0


class TestCircuitBreakerValidation:
    """Test input validation."""

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            DrawdownCircuitBreaker(
                warning_threshold=0.20,
                reduce_threshold=0.10,  # warning > reduce: invalid
            )

    def test_hedge_out_of_range(self):
        with pytest.raises(ValueError):
            DrawdownCircuitBreaker(
                reduce_threshold=0.10,
                hedge_threshold=0.05,  # hedge < reduce: invalid
                halt_threshold=0.20,
            )


class TestSimulateCircuitBreaker:
    """Test simulation function."""

    def test_simulation_length(self):
        drawdowns = [0.01, 0.05, 0.12, 0.18, 0.22, 0.10, 0.03]
        states = simulate_circuit_breaker(drawdowns)
        assert len(states) == 7

    def test_escalation_sequence(self):
        drawdowns = [0.01, 0.06, 0.12, 0.16, 0.22]
        states = simulate_circuit_breaker(drawdowns)
        actions = [s.current_action for s in states]
        assert actions[0] == CircuitAction.NORMAL
        assert actions[1] == CircuitAction.WARNING
        assert actions[2] == CircuitAction.REDUCE
        assert actions[3] == CircuitAction.HEDGE
        assert actions[4] == CircuitAction.HALT

    def test_halt_persists_during_cooldown(self):
        drawdowns = [0.22, 0.15, 0.10, 0.05, 0.03, 0.01]
        states = simulate_circuit_breaker(drawdowns, cooldown_periods=3)
        # First 4 should be halted (1 trigger + 3 cooldown)
        for i in range(4):
            assert states[i].is_halted, f"Period {i} should be halted"
