"""Regression tests for signal_quality win rate and weighted scoring."""

import polars as pl
import pytest

from imst_quant.utils.signal_quality import calculate_signal_win_rate, score_signal_quality


def test_last_rows_unknown_not_losses():
    df = pl.DataFrame({"signal_direction": [1, 1, 1], "returns": [0.0, 0.01, 0.02]})
    out = calculate_signal_win_rate(df)
    assert out["signal_outcome"].to_list() == [1, 1, None]


def test_rolling_win_rate_ignores_neutral_rows():
    # alternating long / flat, every long is right
    n = 50
    df = pl.DataFrame({
        "signal_direction": [1 if i % 2 == 0 else 0 for i in range(n)],
        "returns": [0.01] * n,
    })
    out = calculate_signal_win_rate(df)
    rates = out.filter(pl.col("rolling_win_rate").is_not_null())["rolling_win_rate"]
    assert len(rates) > 0
    assert rates.min() == pytest.approx(1.0)
    assert out.height == n


def test_missing_weight_column_does_not_shrink_score():
    df = pl.DataFrame({"a_signal": [1, -1], "b_signal": [1, -1]})
    out = score_signal_quality(df, weights={"a_signal": 1, "b_signal": 1, "gone_signal": 2})
    assert out["weighted_signal_score"].to_list() == [1.0, -1.0]
    with pytest.raises(ValueError):
        score_signal_quality(df, weights={"gone_signal": 1})
