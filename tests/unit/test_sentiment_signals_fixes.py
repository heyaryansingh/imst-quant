import polars as pl
from imst_quant.utils.sentiment_signals import calculate_sentiment_divergence


def test_divergence_warmup_is_null_not_down():
    df = pl.DataFrame({"sentiment": [0.1] * 6, "returns": [0.01] * 6})
    r = calculate_sentiment_divergence(df, window=3)
    assert r["divergence_signal"].to_list()[:3] == [0, 0, 0]
    # row 2: sentiment shift is null -> must not be labeled "down"
    assert r["sentiment_direction"][2] is None
    assert r["divergence_signal"][2] == 0


def test_nan_or_null_sentiment_and_confidence_hold():
    import math

    import polars as pl

    from imst_quant.utils.sentiment_signals import sentiment_to_signal

    df = pl.DataFrame({
        "sentiment": [math.nan, None, 0.9, 0.9, -0.9],
        "conf": [0.9, 0.9, math.nan, None, 0.9],
    })
    out = sentiment_to_signal(df, confidence_col="conf")
    assert out["sentiment_signal"].to_list() == [0, 0, 0, 0, -1]
