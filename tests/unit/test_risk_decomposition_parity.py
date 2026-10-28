"""Risk parity must equalize risk contributions; beta uses one aligned sample."""

import numpy as np
import pandas as pd

from imst_quant.utils.risk_decomposition import (
    calculate_component_risk,
    calculate_risk_parity_weights,
    decompose_systematic_idiosyncratic,
)


def test_risk_parity_diagonal_cov_is_inverse_vol():
    vols = np.array([0.1, 0.2, 0.3, 0.4])
    w = calculate_risk_parity_weights(np.diag(vols**2))
    expected = (1 / vols) / (1 / vols).sum()
    assert np.allclose(w, expected, atol=1e-6)


def test_risk_parity_correlated_equal_contributions():
    cov = np.array([[0.04, 0.018, 0.0], [0.018, 0.09, 0.01], [0.0, 0.01, 0.01]])
    _, pct = calculate_component_risk(calculate_risk_parity_weights(cov), cov)
    assert np.allclose(pct, 1 / 3, atol=1e-5)


def test_beta_with_missing_asset_data():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 500))
    a = 2 * m + rng.normal(0, 0.001, 500)
    a.iloc[:250] = np.nan
    # Make the dropped half of the market much more volatile
    m.iloc[:250] *= 5
    out = decompose_systematic_idiosyncratic(pd.DataFrame({"a": a}), m)
    assert abs(out["beta"][0] - 2) < 0.05
    assert out["idiosyncratic_var"][0] >= 0
