"""ICSS must not let an edge outlier hide an interior variance break."""

import numpy as np

from imst_quant.utils.change_point import icss_variance_breaks


def test_edge_outlier_does_not_mask_break():
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0, 0.01, 150), rng.normal(0, 0.03, 150)])
    r[3] = 0.6  # one fat-finger print inside the first min_segment bars
    res = icss_variance_breaks(r, min_segment=30)
    assert res["n_breaks"] >= 1
    assert any(130 <= b <= 170 for b in res["break_indices"])
    for seg in res["segments"]:
        assert seg["end"] - seg["start"] >= 30


def test_homoskedastic_series_has_no_break():
    r = np.random.default_rng(1).normal(0, 0.01, 400)
    assert icss_variance_breaks(r)["n_breaks"] == 0
