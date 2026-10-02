import numpy as np
from scipy import stats

from imst_quant.utils.higher_moments import coskewness, cokurtosis


def _b():
    return np.random.default_rng(0).standard_t(4, 25)


def test_coskewness_of_self_equals_skewness():
    b = _b()
    assert np.isclose(coskewness(b, b), stats.skew(b))


def test_cokurtosis_of_self_equals_pearson_kurtosis():
    b = _b()
    assert np.isclose(cokurtosis(b, b), stats.kurtosis(b, fisher=False))
