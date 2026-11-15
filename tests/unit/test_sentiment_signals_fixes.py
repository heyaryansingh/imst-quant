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


def test_flat_sentiment_is_not_bearish_divergence():
    import polars as pl

    df = pl.DataFrame({"sentiment": [0.2] * 8, "returns": [0.01] * 8})
    r = calculate_sentiment_divergence(df, window=3)
    assert r["sentiment_direction"].drop_nulls().to_list() == [0] * 5
    assert r["divergence_signal"].to_list() == [0] * 8
    # Falling sentiment against rising price is still a bullish divergence.
    df = pl.DataFrame({"sentiment": [0.8, 0.7, 0.6, 0.5, 0.4], "returns": [0.01] * 5})
    assert calculate_sentiment_divergence(df, window=2)["divergence_signal"][-1] == 1


def test_composite_crossover_slow_window_exceeds_fast():
    import polars as pl

    from imst_quant.utils.sentiment_signals import SentimentSignalConfig, composite_sentiment_signal

    s = [(-1) ** (i // 40) * 0.5 for i in range(160)]
    for w in (10, 12):  # used to be fast == slow (never crosses) / fast > slow (inverted)
        cfg = SentimentSignalConfig(smoothing_window=w)
        x = [v for v in composite_sentiment_signal(pl.DataFrame({"sentiment": s}), config=cfg)["crossover_signal"] if v]
        assert x and x[0] == -1  # first regime change is a drop in sentiment
