"""Trailing stop-loss signal generation.

This module computes trailing stop levels for long and short positions using
either an ATR-based (volatility-adjusted) or fixed-percentage method. Trailing
stops ratchet favorably with price (never retreat against the position) and
can be checked against price to detect a stop-out event.

Example:
    >>> import polars as pl
    >>> from imst_quant.trading.trailing_stop import atr_trailing_stop
    >>> df = pl.DataFrame({
    ...     "close": [100, 102, 105, 103, 108],
    ...     "atr": [2.0, 2.0, 2.0, 2.0, 2.0],
    ... })
    >>> result = atr_trailing_stop(df, multiplier=2.0)
    >>> print(result["trailing_stop"].to_list())
"""

from typing import Literal

import polars as pl

Direction = Literal["long", "short"]


def _ratchet_stops(raw_stops: list[float | None], direction: Direction) -> list[float | None]:
    """Ratchet raw stop levels so they only move in the favorable direction.

    For "long" positions the stop can only increase over time. For "short"
    positions the stop can only decrease over time. ``None`` values (e.g. from
    insufficient warm-up data) are passed through unchanged and do not reset
    the ratchet.
    """
    ratcheted: list[float | None] = []
    best: float | None = None

    for raw in raw_stops:
        if raw is None:
            ratcheted.append(None)
            continue

        if best is None:
            best = raw
        elif direction == "long":
            best = max(best, raw)
        else:
            best = min(best, raw)

        ratcheted.append(best)

    return ratcheted


def atr_trailing_stop(
    df: pl.DataFrame,
    price_col: str = "close",
    atr_col: str = "atr",
    multiplier: float = 2.0,
    direction: Direction = "long",
) -> pl.DataFrame:
    """Compute an ATR-based trailing stop level series.

    The raw stop at each bar is ``price - multiplier * atr`` for long
    positions (or ``price + multiplier * atr`` for short positions), then
    ratcheted so it only ever moves in the favorable direction.

    Args:
        df: DataFrame containing price and ATR columns.
        price_col: Name of the price column. Defaults to "close".
        atr_col: Name of the ATR column. Defaults to "atr".
        multiplier: ATR multiplier controlling stop distance. Defaults to 2.0.
        direction: "long" or "short". Defaults to "long".

    Returns:
        DataFrame with a new "trailing_stop" column.

    Example:
        >>> df = pl.DataFrame({"close": [100, 103, 101, 107], "atr": [1.0] * 4})
        >>> result = atr_trailing_stop(df, multiplier=2.0)
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if direction == "long":
        raw = (pl.col(price_col) - multiplier * pl.col(atr_col)).alias("_raw_stop")
    else:
        raw = (pl.col(price_col) + multiplier * pl.col(atr_col)).alias("_raw_stop")

    df = df.with_columns(raw)
    ratcheted = _ratchet_stops(df["_raw_stop"].to_list(), direction)
    return df.drop("_raw_stop").with_columns(pl.Series("trailing_stop", ratcheted))


def percent_trailing_stop(
    df: pl.DataFrame,
    price_col: str = "close",
    pct: float = 0.05,
    direction: Direction = "long",
) -> pl.DataFrame:
    """Compute a fixed-percentage trailing stop level series.

    The raw stop at each bar is ``price * (1 - pct)`` for long positions (or
    ``price * (1 + pct)`` for short positions), then ratcheted so it only
    ever moves in the favorable direction.

    Args:
        df: DataFrame containing a price column.
        price_col: Name of the price column. Defaults to "close".
        pct: Trailing stop distance as a fraction of price (e.g. 0.05 = 5%).
            Defaults to 0.05.
        direction: "long" or "short". Defaults to "long".

    Returns:
        DataFrame with a new "trailing_stop" column.

    Example:
        >>> df = pl.DataFrame({"close": [100, 110, 105, 120]})
        >>> result = percent_trailing_stop(df, pct=0.05)
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if not 0 < pct < 1:
        raise ValueError(f"pct must be between 0 and 1, got {pct}")

    if direction == "long":
        raw = (pl.col(price_col) * (1 - pct)).alias("_raw_stop")
    else:
        raw = (pl.col(price_col) * (1 + pct)).alias("_raw_stop")

    df = df.with_columns(raw)
    ratcheted = _ratchet_stops(df["_raw_stop"].to_list(), direction)
    return df.drop("_raw_stop").with_columns(pl.Series("trailing_stop", ratcheted))


def check_stop_triggered(
    df: pl.DataFrame,
    price_col: str = "close",
    stop_col: str = "trailing_stop",
    direction: Direction = "long",
) -> pl.DataFrame:
    """Flag bars where price has crossed the trailing stop level.

    Args:
        df: DataFrame containing price and stop level columns.
        price_col: Name of the price column. Defaults to "close".
        stop_col: Name of the trailing stop column. Defaults to "trailing_stop".
        direction: "long" or "short". Defaults to "long".

    Returns:
        DataFrame with a new "stop_triggered" boolean column. True when
        price has moved through the stop level (price <= stop for long,
        price >= stop for short).

    Example:
        >>> df = pl.DataFrame({"close": [100, 95], "trailing_stop": [90, 96]})
        >>> result = check_stop_triggered(df)
        >>> result["stop_triggered"].to_list()  # [False, True]
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if direction == "long":
        condition = pl.col(price_col) <= pl.col(stop_col)
    else:
        condition = pl.col(price_col) >= pl.col(stop_col)

    return df.with_columns(
        pl.when(pl.col(stop_col).is_not_null())
        .then(condition)
        .otherwise(False)
        .alias("stop_triggered")
    )
