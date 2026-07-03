"""Trading policies, backtesting, position sizing, and validation.

This module provides core trading functionality including signal generation,
policy execution, backtesting, position sizing, and walk-forward validation.

Features:
    - Fixed and dynamic threshold trading policies
    - Signal generation from predictions
    - Event-driven backtesting engine
    - Position sizing strategies (fixed fractional, volatility-adjusted, Kelly)
    - Walk-forward validation for strategy robustness testing

Example:
    >>> from imst_quant.trading import run_backtest, PositionSizer
    >>> results = run_backtest(features_path, transaction_cost=0.001)
    >>> sizer = PositionSizer(account_equity=100000.0)
    >>> size = sizer.fixed_fractional(entry_price=150.0, stop_loss=145.0)
"""

from .backtest import run_backtest
from .policy import DynamicThresholdPolicy, FixedThresholdPolicy
from .position_pyramiding import (
    PyramidLevel,
    generate_pyramid_levels,
    next_pyramid_trigger,
    total_position_size,
)
from .position_sizing import (
    PositionConfig,
    PositionSizer,
    SizingMethod,
)
from .signals import (
    bollinger_band_signal,
    breakout_signal,
    composite_signal,
    crossover_signal,
    macd_signal,
    mean_reversion_signal,
    momentum_signal,
    prediction_to_signal,
    signal_strength,
    volatility_adjusted_signal,
)
from .trailing_stop import (
    atr_trailing_stop,
    check_stop_triggered,
    percent_trailing_stop,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    WindowResult,
)

__all__ = [
    # Policies
    "FixedThresholdPolicy",
    "DynamicThresholdPolicy",
    # Signals
    "prediction_to_signal",
    "momentum_signal",
    "mean_reversion_signal",
    "crossover_signal",
    "composite_signal",
    "breakout_signal",
    "volatility_adjusted_signal",
    "macd_signal",
    "signal_strength",
    "bollinger_band_signal",
    # Backtesting
    "run_backtest",
    # Position sizing
    "PositionSizer",
    "PositionConfig",
    "SizingMethod",
    # Walk-forward validation
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WindowResult",
    # Trailing stop
    "atr_trailing_stop",
    "percent_trailing_stop",
    "check_stop_triggered",
    # Position pyramiding
    "PyramidLevel",
    "generate_pyramid_levels",
    "next_pyramid_trigger",
    "total_position_size",
]
