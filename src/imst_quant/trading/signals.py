"""Signal generation from ML predictions (TRAD-03).

This module converts model probability outputs into actionable trading signals.
Signals are discretized into long (+1), short (-1), and neutral (0) positions
based on configurable probability thresholds.

The signal generation is designed to work with any binary classification model
that outputs probability of upward price movement.

Functions:
    prediction_to_signal: Convert probability predictions to trading signals

Example:
    >>> import polars as pl
    >>> from imst_quant.trading.signals import prediction_to_signal
    >>> df = pl.DataFrame({"asset": ["BTC", "ETH"], "prob_up": [0.7, 0.3]})
    >>> signals = prediction_to_signal(df, threshold=0.6)
    >>> print(signals["signal"].to_list())  # [1, -1]
"""

from typing import List

import polars as pl


def prediction_to_signal(
    df: pl.DataFrame,
    prob_col: str = "prob_up",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Convert model probability predictions to discrete trading signals.

    Transforms continuous probability outputs from classification models
    into actionable trading signals. Uses symmetric thresholds around 0.5
    to determine long, short, or neutral positions.

    Args:
        df: DataFrame containing prediction probabilities. Must have
            the column specified by prob_col.
        prob_col: Name of the column containing upward movement probability.
            Values should be in range [0, 1]. Defaults to "prob_up".
        threshold: Probability threshold for signal generation.
            - prob > threshold: long signal (+1)
            - prob < (1 - threshold): short signal (-1)
            - otherwise: neutral signal (0)
            Defaults to 0.5 (equal threshold).

    Returns:
        DataFrame with new "signal" column containing integer values:
        1 for long, -1 for short, 0 for neutral.

    Example:
        >>> df = pl.DataFrame({"prob_up": [0.8, 0.2, 0.5]})
        >>> result = prediction_to_signal(df, threshold=0.6)
        >>> result["signal"].to_list()  # [1, -1, 0]
    """
    return df.with_columns(
        pl.when(pl.col(prob_col) > threshold)
        .then(1)
        .when(pl.col(prob_col) < (1 - threshold))
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )
