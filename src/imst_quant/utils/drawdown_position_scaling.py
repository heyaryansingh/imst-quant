"""Drawdown-based position scaling for dynamic risk management.

Reduces position sizes progressively as drawdown deepens, and scales
back up as equity recovers. Implements multiple scaling curves and
integrates with existing position sizing utilities.

Functions:
    calculate_current_drawdown: Compute drawdown from equity curve or HWM.
    linear_scale_factor: Linear reduction from 1.0 at no drawdown to floor at max.
    convex_scale_factor: Convex (aggressive) reduction that cuts faster.
    concave_scale_factor: Concave (gentle) reduction that holds longer.
    step_scale_factor: Discrete steps at configurable drawdown tiers.
    apply_drawdown_scaling: Apply scaling to a base position size.
    drawdown_scaling_report: Comprehensive report of current scaling state.

Example:
    >>> from imst_quant.utils.drawdown_position_scaling import apply_drawdown_scaling
    >>> scaled = apply_drawdown_scaling(
    ...     base_size=10_000, equity=95_000, high_water_mark=100_000
    ... )
    >>> print(f"Scaled position: ${scaled:.0f}")
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import structlog

logger = structlog.get_logger()

ScalingMethod = Literal["linear", "convex", "concave", "step"]


@dataclass
class ScalingConfig:
    """Configuration for drawdown-based position scaling.

    Attributes:
        method: Scaling curve type.
        max_drawdown: Drawdown level at which scaling reaches the floor.
        floor: Minimum scale factor (0.0 = full stop, 0.1 = 10% of base).
        kill_switch: Drawdown level to halt all trading (scale = 0).
        recovery_delay_pct: Required recovery from trough before scaling up.
        tiers: Step tiers as list of (drawdown_pct, scale_factor) for step method.
    """

    method: ScalingMethod = "linear"
    max_drawdown: float = 0.20
    floor: float = 0.25
    kill_switch: float = 0.30
    recovery_delay_pct: float = 0.02
    tiers: Optional[List[tuple]] = None


@dataclass
class ScalingState:
    """Current state of drawdown scaling.

    Attributes:
        equity: Current portfolio equity.
        high_water_mark: Peak equity value.
        drawdown_pct: Current drawdown as fraction (0-1).
        scale_factor: Current position scale factor (0-1).
        is_halted: Whether kill switch has been triggered.
        method: Active scaling method name.
    """

    equity: float
    high_water_mark: float
    drawdown_pct: float
    scale_factor: float
    is_halted: bool
    method: str


DEFAULT_CONFIG = ScalingConfig()

DEFAULT_TIERS = [
    (0.05, 0.80),
    (0.10, 0.50),
    (0.15, 0.30),
    (0.20, 0.10),
]


def calculate_current_drawdown(equity: float, high_water_mark: float) -> float:
    """Compute current drawdown as a fraction.

    Args:
        equity: Current portfolio value.
        high_water_mark: Peak portfolio value.

    Returns:
        Drawdown as a positive fraction (0.0 = at peak, 0.15 = 15% drawdown).
    """
    if high_water_mark <= 0:
        return 0.0
    dd = (high_water_mark - equity) / high_water_mark
    return max(0.0, dd)


def linear_scale_factor(
    drawdown_pct: float,
    max_drawdown: float = 0.20,
    floor: float = 0.25,
) -> float:
    """Linear reduction from 1.0 at zero drawdown to floor at max_drawdown.

    Args:
        drawdown_pct: Current drawdown fraction.
        max_drawdown: Drawdown at which floor is reached.
        floor: Minimum scaling factor.

    Returns:
        Scale factor between floor and 1.0.
    """
    if drawdown_pct <= 0:
        return 1.0
    if drawdown_pct >= max_drawdown:
        return floor
    t = drawdown_pct / max_drawdown
    return 1.0 - t * (1.0 - floor)


def convex_scale_factor(
    drawdown_pct: float,
    max_drawdown: float = 0.20,
    floor: float = 0.25,
    power: float = 2.0,
) -> float:
    """Convex scaling - cuts position size aggressively at first drawdown.

    Args:
        drawdown_pct: Current drawdown fraction.
        max_drawdown: Drawdown at which floor is reached.
        floor: Minimum scaling factor.
        power: Exponent controlling convexity (>1 = more aggressive).

    Returns:
        Scale factor between floor and 1.0.
    """
    if drawdown_pct <= 0:
        return 1.0
    if drawdown_pct >= max_drawdown:
        return floor
    t = drawdown_pct / max_drawdown
    # Convex curve: drops faster at the start of drawdown
    return 1.0 - (1.0 - (1.0 - t) ** power) * (1.0 - floor)


def concave_scale_factor(
    drawdown_pct: float,
    max_drawdown: float = 0.20,
    floor: float = 0.25,
    power: float = 2.0,
) -> float:
    """Concave scaling - holds position size longer, drops sharply near max.

    Args:
        drawdown_pct: Current drawdown fraction.
        max_drawdown: Drawdown at which floor is reached.
        floor: Minimum scaling factor.
        power: Exponent controlling concavity (>1 = more gradual early).

    Returns:
        Scale factor between floor and 1.0.
    """
    if drawdown_pct <= 0:
        return 1.0
    if drawdown_pct >= max_drawdown:
        return floor
    t = drawdown_pct / max_drawdown
    return 1.0 - (t ** power) * (1.0 - floor)


def step_scale_factor(
    drawdown_pct: float,
    tiers: Optional[List[tuple]] = None,
    kill_switch: float = 0.30,
) -> float:
    """Step-function scaling at discrete drawdown tiers.

    Args:
        drawdown_pct: Current drawdown fraction.
        tiers: List of (drawdown_threshold, scale_factor) pairs, sorted ascending.
        kill_switch: Drawdown level to return 0.0.

    Returns:
        Scale factor from the matching tier, or 0.0 if kill switch triggered.
    """
    if drawdown_pct >= kill_switch:
        return 0.0

    if tiers is None:
        tiers = DEFAULT_TIERS

    scale = 1.0
    for threshold, factor in sorted(tiers, key=lambda x: x[0]):
        if drawdown_pct >= threshold:
            scale = factor
        else:
            break
    return scale


def apply_drawdown_scaling(
    base_size: float,
    equity: float,
    high_water_mark: float,
    config: Optional[ScalingConfig] = None,
) -> float:
    """Apply drawdown-based scaling to a base position size.

    Args:
        base_size: Position size before drawdown adjustment.
        equity: Current portfolio value.
        high_water_mark: Peak portfolio value.
        config: Scaling configuration (uses defaults if None).

    Returns:
        Adjusted position size after drawdown scaling.
    """
    if config is None:
        config = DEFAULT_CONFIG

    dd = calculate_current_drawdown(equity, high_water_mark)

    # Kill switch check
    if dd >= config.kill_switch:
        logger.warning(
            "drawdown_kill_switch",
            drawdown_pct=dd,
            threshold=config.kill_switch,
        )
        return 0.0

    # Calculate scale factor based on method
    if config.method == "linear":
        scale = linear_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "convex":
        scale = convex_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "concave":
        scale = concave_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "step":
        scale = step_scale_factor(dd, config.tiers, config.kill_switch)
    else:
        scale = linear_scale_factor(dd, config.max_drawdown, config.floor)

    scaled_size = base_size * scale

    logger.info(
        "drawdown_scaling_applied",
        drawdown_pct=dd,
        method=config.method,
        scale_factor=scale,
        base_size=base_size,
        scaled_size=scaled_size,
    )

    return scaled_size


def get_scaling_state(
    equity: float,
    high_water_mark: float,
    config: Optional[ScalingConfig] = None,
) -> ScalingState:
    """Get the current drawdown scaling state without modifying positions.

    Args:
        equity: Current portfolio equity.
        high_water_mark: Peak equity value.
        config: Scaling configuration.

    Returns:
        ScalingState with current drawdown metrics and scale factor.
    """
    if config is None:
        config = DEFAULT_CONFIG

    dd = calculate_current_drawdown(equity, high_water_mark)
    is_halted = dd >= config.kill_switch

    if is_halted:
        scale = 0.0
    elif config.method == "linear":
        scale = linear_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "convex":
        scale = convex_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "concave":
        scale = concave_scale_factor(dd, config.max_drawdown, config.floor)
    elif config.method == "step":
        scale = step_scale_factor(dd, config.tiers, config.kill_switch)
    else:
        scale = linear_scale_factor(dd, config.max_drawdown, config.floor)

    return ScalingState(
        equity=equity,
        high_water_mark=high_water_mark,
        drawdown_pct=dd,
        scale_factor=scale,
        is_halted=is_halted,
        method=config.method,
    )


def drawdown_scaling_report(
    equity: float,
    high_water_mark: float,
    base_position_sizes: Dict[str, float],
    config: Optional[ScalingConfig] = None,
) -> Dict:
    """Generate a comprehensive drawdown scaling report.

    Args:
        equity: Current portfolio equity.
        high_water_mark: Peak equity value.
        base_position_sizes: Map of asset -> base position size in dollars.
        config: Scaling configuration.

    Returns:
        Dict with scaling state, per-asset adjustments, and aggregate metrics.
    """
    state = get_scaling_state(equity, high_water_mark, config)

    adjusted = {}
    total_base = 0.0
    total_scaled = 0.0

    for asset, base in sorted(base_position_sizes.items()):
        scaled = base * state.scale_factor
        adjusted[asset] = {
            "base_size": base,
            "scaled_size": scaled,
            "reduction": base - scaled,
        }
        total_base += base
        total_scaled += scaled

    return {
        "state": {
            "equity": state.equity,
            "high_water_mark": state.high_water_mark,
            "drawdown_pct": state.drawdown_pct,
            "scale_factor": state.scale_factor,
            "is_halted": state.is_halted,
            "method": state.method,
        },
        "positions": adjusted,
        "aggregate": {
            "total_base_exposure": total_base,
            "total_scaled_exposure": total_scaled,
            "total_reduction": total_base - total_scaled,
            "reduction_pct": (total_base - total_scaled) / total_base if total_base > 0 else 0.0,
        },
    }
