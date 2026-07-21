"""Triple-barrier labeling for supervised learning on financial series.

The triple-barrier method (Lopez de Prado, "Advances in Financial Machine
Learning", 2018, ch. 3) labels each event by which of three barriers the
price path touches first: an upper profit-taking barrier, a lower stop-loss
barrier, or a vertical time barrier. Barrier widths scale with local
volatility so labels stay meaningful across regimes.

Labels: +1 (upper barrier hit first), -1 (lower barrier hit first),
0 (vertical barrier expired first).
"""

import numpy as np
from typing import Any, Dict, Optional, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def ewm_volatility(prices: ArrayLike, span: int = 20) -> np.ndarray:
    """Estimate per-bar return volatility with an exponentially weighted std.

    Args:
        prices: Price series.
        span: EWM span in bars (default 20).

    Returns:
        Array of volatility estimates, same length as prices. The first
        element is NaN (no return available).

    Raises:
        ValueError: If fewer than 2 prices or span < 2.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.size < 2:
        raise ValueError(f"Need at least 2 prices, got {arr.size}")
    if span < 2:
        raise ValueError(f"span must be >= 2, got {span}")

    returns = np.diff(arr) / arr[:-1]
    alpha = 2.0 / (span + 1.0)

    # EWM mean and mean-of-squares via recursive update
    vol = np.full(arr.size, np.nan)
    mean = returns[0]
    mean_sq = returns[0] ** 2
    for i, r in enumerate(returns):
        mean = alpha * r + (1 - alpha) * mean
        mean_sq = alpha * r * r + (1 - alpha) * mean_sq
        var = max(mean_sq - mean * mean, 0.0)
        vol[i + 1] = np.sqrt(var)
    return vol


def triple_barrier_labels(
    prices: ArrayLike,
    events: Optional[Sequence[int]] = None,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 10,
    volatility: Optional[ArrayLike] = None,
    vol_span: int = 20,
) -> Dict[str, Any]:
    """Label events with the triple-barrier method.

    For each event index t, sets an upper barrier at
    price[t] * (1 + pt_mult * vol[t]), a lower barrier at
    price[t] * (1 - sl_mult * vol[t]), and a vertical barrier at
    t + max_holding. The label is decided by whichever is touched first.

    Args:
        prices: Price series.
        events: Indices at which positions are opened. Default: every bar
            that has a valid volatility estimate and room before the end.
        pt_mult: Profit-taking barrier width in volatility units (default 2).
        sl_mult: Stop-loss barrier width in volatility units (default 2).
        max_holding: Vertical barrier in bars (default 10).
        volatility: Optional per-bar volatility (same length as prices).
            Default: ewm_volatility(prices, vol_span).
        vol_span: EWM span used when volatility is not supplied.

    Returns:
        Dict with:
            event_indices: Array of labeled event indices.
            labels: Array of +1 / -1 / 0 labels.
            touch_indices: Index where the deciding barrier was touched.
            returns: Realized return from event to touch.
            upper_barriers / lower_barriers: Barrier price levels.

    Raises:
        ValueError: If inputs are inconsistent or no event can be labeled.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.size < 3:
        raise ValueError(f"Need at least 3 prices, got {arr.size}")
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("pt_mult and sl_mult must be positive")
    if max_holding < 1:
        raise ValueError(f"max_holding must be >= 1, got {max_holding}")

    if volatility is None:
        vol = ewm_volatility(arr, span=vol_span)
    else:
        vol = np.asarray(volatility, dtype=float)
        if vol.size != arr.size:
            raise ValueError(
                f"volatility length {vol.size} != prices length {arr.size}"
            )

    if events is None:
        candidates = np.arange(arr.size - 1)
    else:
        candidates = np.asarray(events, dtype=int)
        if candidates.size and (candidates.min() < 0 or candidates.max() >= arr.size):
            raise ValueError("event indices out of range")

    event_indices = []
    labels = []
    touch_indices = []
    event_returns = []
    uppers = []
    lowers = []

    for t in candidates:
        if t >= arr.size - 1 or not np.isfinite(vol[t]) or vol[t] <= 0:
            continue
        upper = arr[t] * (1.0 + pt_mult * vol[t])
        lower = arr[t] * (1.0 - sl_mult * vol[t])
        end = min(t + max_holding, arr.size - 1)

        label = 0
        touch = end
        for j in range(t + 1, end + 1):
            if arr[j] >= upper:
                label, touch = 1, j
                break
            if arr[j] <= lower:
                label, touch = -1, j
                break

        event_indices.append(t)
        labels.append(label)
        touch_indices.append(touch)
        event_returns.append(arr[touch] / arr[t] - 1.0)
        uppers.append(upper)
        lowers.append(lower)

    if not event_indices:
        raise ValueError("No events could be labeled (check volatility warmup)")

    return {
        "event_indices": np.asarray(event_indices),
        "labels": np.asarray(labels),
        "touch_indices": np.asarray(touch_indices),
        "returns": np.asarray(event_returns),
        "upper_barriers": np.asarray(uppers),
        "lower_barriers": np.asarray(lowers),
    }


def label_distribution(labels: ArrayLike) -> Dict[str, Any]:
    """Summarize a triple-barrier label set.

    Args:
        labels: Array of +1 / -1 / 0 labels.

    Returns:
        Dict with counts, proportions, and class balance diagnostics
        (imbalance_ratio = largest class share / smallest nonzero share).

    Raises:
        ValueError: If labels is empty.
    """
    arr = np.asarray(labels, dtype=int)
    if arr.size == 0:
        raise ValueError("labels is empty")

    counts = {
        "upper": int(np.sum(arr == 1)),
        "lower": int(np.sum(arr == -1)),
        "vertical": int(np.sum(arr == 0)),
    }
    total = arr.size
    proportions = {k: v / total for k, v in counts.items()}
    nonzero = [v for v in counts.values() if v > 0]
    imbalance = max(nonzero) / min(nonzero) if len(nonzero) > 1 else float("inf")

    return {
        "n_labels": total,
        "counts": counts,
        "proportions": proportions,
        "imbalance_ratio": float(imbalance),
        "directional_share": (counts["upper"] + counts["lower"]) / total,
    }
