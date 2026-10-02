from datetime import date

import polars as pl

from imst_quant.utils.seasonality import analyze_day_of_week_effect


def test_day_of_week_names_are_correct():
    # 2024-01-01 is a Monday
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 7)],
            "return_1d": [0.05, -0.05, 0.0],
        }
    )
    r = analyze_day_of_week_effect(df)
    assert r.best_period == "Monday"
    assert r.worst_period == "Tuesday"
    assert "Sunday" in r.by_period["day_name"].to_list()
