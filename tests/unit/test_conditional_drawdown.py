import numpy as np

from imst_quant.utils.conditional_drawdown import conditional_drawdown_at_risk


def test_cdar_is_tail_mean_when_dar_is_zero():
    # 18 flat periods then two losses: DaR(0.8) == 0 but CDaR must be tail mean
    r = np.array([0.0] * 18 + [-0.10, -0.10])
    # drawdowns: 0 x18, 0.10, 0.19 -> tail beyond dar=0 is mean(0.10, 0.19)
    assert abs(conditional_drawdown_at_risk(r, alpha=0.8) - 0.145) < 1e-9


def test_first_period_loss_is_a_drawdown():
    # Peak floors at starting capital: a -10% opening bar is 10% underwater.
    import pytest
    from imst_quant.utils.conditional_drawdown import drawdown_series

    dd = drawdown_series([-0.10, 0.05])
    assert dd[0] == pytest.approx(0.10)
    assert dd[1] == pytest.approx(1 - 0.9 * 1.05)
