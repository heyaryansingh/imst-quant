import pytest

from imst_quant.utils.scenario_analysis import (
    apply_scenario,
    custom_stress_test,
    define_scenario,
    run_scenario_analysis,
)


def test_max_loss_is_zero_when_every_contribution_gains():
    result = apply_scenario({"SPY": 0.5, "TLT": 0.5}, define_scenario("rally", {"SPY": 0.1, "TLT": 0.02}))
    assert result.max_loss == 0.0
    assert result.portfolio_impact == pytest.approx(0.06)


def test_max_loss_reports_worst_negative_contribution():
    result = apply_scenario({"SPY": 0.6, "TLT": 0.4}, define_scenario("crash", {"SPY": -0.2, "TLT": 0.05}))
    assert result.max_loss == pytest.approx(-0.12)


def test_stress_shock_capped_at_total_loss():
    result = custom_stress_test({"EEM": 1.0}, {"EEM": -0.8}, correlation_multiplier=1.5)
    assert result.portfolio_impact == pytest.approx(-1.0)


@pytest.mark.parametrize("mult", [0.0, -1.0])
def test_stress_rejects_non_positive_multiplier(mult):
    with pytest.raises(ValueError):
        custom_stress_test({"SPY": 1.0}, {"SPY": 0.05}, correlation_multiplier=mult)


def test_empty_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="weights"):
        apply_scenario({}, define_scenario("x", {"SPY": -0.1}))
    with pytest.raises(ValueError, match="scenarios"):
        run_scenario_analysis({"SPY": 1.0}, [])
