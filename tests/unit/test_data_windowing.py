from datetime import datetime, timedelta

import polars as pl

from imst_quant.utils.data_windowing import create_time_based_folds


def test_time_folds_with_gap_never_overlap_train():
    df = pl.DataFrame({
        "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)],
        "value": list(range(40)),
    })
    for gap in (timedelta(0), timedelta(days=2)):
        for train, val in create_time_based_folds(df, "date", n_folds=3, gap=gap):
            assert val["date"].min() > train["date"].max() + gap
