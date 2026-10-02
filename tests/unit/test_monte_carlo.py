import numpy as np

from imst_quant.utils.monte_carlo import MonteCarloSimulator


def test_gbm_var_relative_to_initial_price():
    sim = MonteCarloSimulator(np.array([0.001, -0.001] * 10), n_simulations=2000, seed=1)
    res = sim.run_gbm_simulation(horizon=5, initial_price=100.0)
    expected = -float(np.percentile(res.terminal_values / 100.0 - 1, 5))
    assert abs(res.var - expected) < 1e-12
