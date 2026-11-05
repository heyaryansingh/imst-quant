import math

import pytest

from imst_quant.utils.drawdown_circuit_breaker import CircuitAction, DrawdownCircuitBreaker


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_drawdown_rejected(bad):
    breaker = DrawdownCircuitBreaker()
    with pytest.raises(ValueError):
        breaker.update(bad)
    with pytest.raises(ValueError):
        breaker.check(bad)


def test_recovery_from_halt_logs_single_normal_event():
    breaker = DrawdownCircuitBreaker(cooldown_periods=1)
    breaker.update(0.25)  # halt
    breaker.update(0.25)  # cooldown tick
    state = breaker.update(0.01)  # recovered
    assert state.current_action == CircuitAction.NORMAL
    assert [e.action for e in breaker.events] == [CircuitAction.HALT, CircuitAction.NORMAL]
