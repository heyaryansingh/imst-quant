"""Tests for remove_outliers on columns with missing values."""

import polars as pl
import pytest

from imst_quant.utils.data_cleaning import remove_outliers

CLOSE = [1.5, 2.5, None, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, float("nan"), 400.0]


@pytest.mark.parametrize("method,threshold", [("iqr", 3.0), ("zscore", 2.0), ("percentile", 15.0)])
def test_missing_rows_are_kept_and_only_the_spike_is_removed(method, threshold):
    df = pl.DataFrame({"close": CLOSE})
    out = remove_outliers(df, "close", method=method, threshold=threshold)
    assert 400.0 not in out["close"].to_list()
    assert out["close"].null_count() == 1
    assert out["close"].is_nan().sum() == 1
    if method != "percentile":  # percentile also trims the low tail by design
        assert out.height == df.height - 1


def test_integer_column_and_all_missing_column():
    ints = pl.DataFrame({"v": [1, 2, 3, 4, 1000]})
    assert remove_outliers(ints, "v", method="iqr", threshold=1.5)["v"].to_list() == [1, 2, 3, 4]
    empty = pl.DataFrame({"v": [None, None]}, schema={"v": pl.Float64})
    assert remove_outliers(empty, "v").height == 2
