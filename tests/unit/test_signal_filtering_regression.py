import polars as pl
import pytest

from imst_quant.utils.signal_filtering import calculate_signal_decay


def test_constant_signal_decays_by_half_life():
    out = calculate_signal_decay(
        pl.DataFrame({"signal": [1.0] * 5}), half_life=2, min_threshold=0.0
    )["signal_decayed"].to_list()
    assert out[0] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)
    assert out[4] == pytest.approx(0.25)
