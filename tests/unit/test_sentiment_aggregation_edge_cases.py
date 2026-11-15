"""Edge cases for sentiment aggregation: date dtypes, missing polarity, flat windows, zeros."""

from datetime import date, datetime

import polars as pl
import pytest

from imst_quant.sentiment.aggregation import (
    aggregate_daily_sentiment,
    calculate_sentiment_divergence,
    calculate_sentiment_z_score,
)


def _posts(dates):
    return pl.DataFrame({
        "date": dates,
        "asset_id": ["A", "A", "A", "B"],
        "author_id": ["u", "v", "w", "u"],
        "polarity": [0.5, float("nan"), None, -0.2],
    })


@pytest.mark.parametrize("dates", [
    [date(2026, 4, 7)] * 4,
    [datetime(2026, 4, 7, 15)] * 4,
    ["2026-04-07"] * 4,
])
def test_daily_handles_date_types_and_missing_polarity(dates):
    out = aggregate_daily_sentiment(_posts(dates), date(2026, 4, 7), {"u": 2.0})
    assert out == pytest.approx({"A": 0.5, "B": -0.2})
    assert aggregate_daily_sentiment(_posts(dates), date(2026, 4, 8)) == {}


def test_z_score_flat_window_is_null():
    df = pl.DataFrame({"asset_id": ["A"] * 4, "sentiment": [0.3, 0.3, 0.3, 0.6]})
    z = calculate_sentiment_z_score(df, window=3)["sentiment_z_score"].to_list()
    assert z[2] is None and z[3] > 0


def test_divergence_ignores_flat_side():
    sent = pl.DataFrame({"date": ["d1", "d2", "d3"], "asset_id": ["A"] * 3, "sentiment": [0.0, 0.5, 0.5]})
    rets = pl.DataFrame({"date": ["d1", "d2", "d3"], "asset_id": ["A"] * 3, "return_1d": [0.01, 0.0, -0.01]})
    out = calculate_sentiment_divergence(sent, rets).sort("date")
    assert out["sentiment_return_divergence"].to_list() == [0, 0, 1]
