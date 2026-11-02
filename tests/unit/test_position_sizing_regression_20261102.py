import polars as pl

from imst_quant.utils.position_sizing import calculate_position_sizes, kelly_criterion_size


def test_kelly_zero_avg_win_returns_zero():
    assert kelly_criterion_size(100000, 0.5, 0.0, 0.01) == 0.0


def test_atr_zero_gives_zero_units():
    df = pl.DataFrame({"close": [100.0, 100.0], "signal_direction": [1, 1], "atr": [0.0, 2.0]})
    out = calculate_position_sizes(df, 100000, "atr", atr_col="atr")
    assert out["position_units"].to_list() == [0.0, 500.0]
    assert out["position_size"].to_list() == [0.0, 50000.0]
