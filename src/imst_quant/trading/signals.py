"""Signal generation from predictions (TRAD-03)."""

from typing import List

import polars as pl


def prediction_to_signal(
    df: pl.DataFrame,
    prob_col: str = "prob_up",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Add signal column: 1 long, -1 short, 0 neutral."""
    return df.with_columns(
        pl.when(pl.col(prob_col) > threshold)
        .then(1)
        .when(pl.col(prob_col) < (1 - threshold))
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )
