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


def simulate_trailing_stop(
    prices: list[float],
    pct: float,
    cooldown: int = 5,
) -> dict[str, float]:
    """Simulate a long-only percent trailing stop with re-entry after a cooldown.

    Starts long at the first price. The stop trails the highest close since
    entry by ``pct``; a close at or below it exits at that close. After
    ``cooldown`` flat bars the position re-enters at the close.

    Args:
        prices: Close prices (positive, finite), oldest first.
        pct: Trailing distance as a fraction (0 < pct < 1).
        cooldown: Flat bars after a stop-out before re-entering (>= 0).

    Returns:
        Dict with total_return, buy_hold_return, max_drawdown (of the strategy
        equity, <= 0), n_stops and time_in_market (fraction of bar-to-bar
        returns held).
    """
    if not 0 < pct < 1:
        raise ValueError(f"pct must be between 0 and 1, got {pct}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be >= 0, got {cooldown}")
    if len(prices) < 2:
        raise ValueError("need at least 2 prices")
    if any(not (p > 0) or p == float("inf") for p in prices):
        raise ValueError("prices must be positive and finite")

    equity = peak_equity = 1.0
    max_dd = 0.0
    in_pos, high, flat_left = True, prices[0], 0
    n_stops = held = 0

    for prev, price in zip(prices, prices[1:]):
        if in_pos:
            equity *= price / prev
            held += 1
            high = max(high, price)
            if price <= high * (1 - pct):
                in_pos, flat_left = False, cooldown
                n_stops += 1
        elif flat_left > 0:
            flat_left -= 1
        if not in_pos and flat_left == 0:
            # Re-enter at this close; next bar's move is held.
            in_pos, high = True, price
        peak_equity = max(peak_equity, equity)
        max_dd = min(max_dd, equity / peak_equity - 1)

    return {
        "total_return": equity - 1,
        "buy_hold_return": prices[-1] / prices[0] - 1,
        "max_drawdown": max_dd,
        "n_stops": float(n_stops),
        "time_in_market": held / (len(prices) - 1),
    }
