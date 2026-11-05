import pandas as pd

from imst_quant.utils.signal_validator import validate_signal_quality


def test_hit_rate_pairs_each_return_with_prior_bar_signal():
    # Long on bar 0 -> wins bar 1; flat bars 1-2; short on bar 3 -> wins bar 4.
    signal = pd.Series([1, 0, 0, -1, 0])
    returns = pd.Series([0.0, 0.02, 0.05, 0.0, -0.03])
    out = validate_signal_quality(signal, returns)
    assert out["hit_rate"] == 1.0
