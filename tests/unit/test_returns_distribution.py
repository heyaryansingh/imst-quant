import math

import numpy as np
import pandas as pd

from imst_quant.utils.returns_distribution import analyze_distribution, analyze_tails
from imst_quant.utils.returns_distribution import test_normality as run_normality_tests


def test_inf_and_nan_returns_are_ignored():
    rng = np.random.default_rng(0)
    clean = pd.Series(rng.normal(0, 0.01, 300))
    dirty = pd.concat([clean, pd.Series([np.inf, -np.inf, np.nan])], ignore_index=True)

    stats = analyze_distribution(dirty)
    ref = analyze_distribution(clean)
    assert stats.n_observations == 300
    assert math.isclose(stats.skewness, ref.skewness)
    assert math.isfinite(stats.jarque_bera_stat)
    assert stats.max_return == ref.max_return

    tails = analyze_tails(dirty)
    assert math.isfinite(tails.var_95) and math.isfinite(tails.expected_shortfall_95)
    assert math.isfinite(run_normality_tests(dirty).kolmogorov_smirnov[0])
