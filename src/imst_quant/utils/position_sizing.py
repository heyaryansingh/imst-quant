"""Position sizing and risk management utilities.

This module provides utilities for calculating optimal position sizes based on
various risk management strategies including fixed fractional, Kelly criterion,
volatility-based, and ATR-based sizing.

Functions:
    fixed_fractional_size: Size based on fixed percentage of capital
    kelly_criterion_size: Optimal size using Kelly formula
    volatility_based_size: Size based on volatility targeting
    atr_based_size: Size based on Average True Range
    calculate_position_sizes: Apply sizing strategy to signals
    get_recommended_sizing: Get sizing recommendations for different risk levels

Example:
    >>> from imst_quant.utils.position_sizing import kelly_criterion_size
    >>> size = kelly_criterion_size(
    ...     capital=100000,
    ...     win_rate=0.55,
    ...     avg_win=0.02,
    ...     avg_loss=0.01,
    ... )
    >>> print(f"Kelly optimal position: ${size:.2f}")
"""

from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()


def fixed_fractional_size(
    capital: float,
    risk_per_trade: float = 0.02,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> float:
    """Calculate position size using fixed fractional method.

    Risks a fixed percentage of capital on each trade. If stop loss is
    provided, adjusts size to risk exactly that percentage.

    Args:
        capital: Total trading capital.
        risk_per_trade: Percentage of capital to risk (default: 2%).
        entry_price: Entry price for the trade.
        stop_loss: Stop loss price (optional).

    Returns:
        Position size in dollars (or units if entry_price provided).

    Example:
        >>> size = fixed_fractional_size(100000, risk_per_trade=0.02)
        >>> print(f"Risk ${size:.2f} per trade")
    """
    risk_amount = capital * risk_per_trade

    if entry_price and stop_loss:
        # Calculate position size in units to risk exact amount
        price_risk_per_unit = abs(entry_price - stop_loss)
        if price_risk_per_unit > 0:
            units = risk_amount / price_risk_per_unit
            return units * entry_price  # Return dollar value

    return risk_amount


def kelly_criterion_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Calculate position size using Kelly Criterion.

    The Kelly Criterion calculates the optimal fraction of capital to risk
    to maximize long-term growth. Uses a fractional Kelly to be conservative.

    Args:
        capital: Total trading capital.
        win_rate: Historical win rate (0-1).
        avg_win: Average winning trade return (as decimal).
        avg_loss: Average losing trade return (as positive decimal).
        kelly_fraction: Fraction of full Kelly to use (default: 0.25 = quarter Kelly).

    Returns:
        Position size in dollars.

    Example:
        >>> size = kelly_criterion_size(
        ...     capital=100000,
        ...     win_rate=0.55,
        ...     avg_win=0.03,
        ...     avg_loss=0.015,
        ... )
    """
    if win_rate >= 1.0 or win_rate <= 0:
        logger.warning(f"Invalid win rate: {win_rate}, using 0.5")
        win_rate = 0.5

    if avg_loss <= 0:
        logger.warning(f"avg_loss must be positive, got {avg_loss}")
        avg_loss = abs(avg_loss)

    # Kelly formula: f = (p * b - q) / b
    # where p = win rate, q = lose rate, b = avg_win / avg_loss
    lose_rate = 1 - win_rate
    b = avg_win / avg_loss if avg_loss > 0 else 1.0

    kelly_pct = (win_rate * b - lose_rate) / b

    # Apply Kelly fraction for conservative sizing
    kelly_pct = max(0, kelly_pct) * kelly_fraction

    position_size = capital * kelly_pct

    logger.debug(f"Kelly: {kelly_pct:.1%}, Position: ${position_size:.2f}")
    return position_size


def volatility_based_size(
    capital: float,
    target_volatility: float,
    asset_volatility: float,
    entry_price: Optional[float] = None,
) -> float:
    """Calculate position size to target a specific portfolio volatility.

    Scales position size inversely with asset volatility to maintain
    constant portfolio risk.

    Args:
        capital: Total trading capital.
        target_volatility: Target portfolio volatility (annualized, as decimal).
        asset_volatility: Asset's volatility (annualized, as decimal).
        entry_price: Entry price (optional, for unit-based sizing).

    Returns:
        Position size in dollars (or units if entry_price provided).

    Example:
        >>> size = volatility_based_size(
        ...     capital=100000,
        ...     target_volatility=0.15,  # 15% target vol
        ...     asset_volatility=0.30,   # 30% asset vol
        ... )
    """
    if asset_volatility <= 0:
        logger.warning(f"Invalid asset volatility: {asset_volatility}, using 0.2")
        asset_volatility = 0.2

    # Position size = (target_vol / asset_vol) * capital
    position_fraction = target_volatility / asset_volatility

    # Cap at 100% capital
    position_fraction = min(position_fraction, 1.0)

    position_size = capital * position_fraction

    if entry_price:
        units = position_size / entry_price
        return units

    return position_size


def atr_based_size(
    capital: float,
    atr: float,
    entry_price: float,
    atr_multiplier: float = 2.0,
    risk_per_trade: float = 0.02,
) -> float:
    """Calculate position size using Average True Range (ATR).

    Sizes positions to risk a fixed percentage of capital based on
    ATR-based stop loss.

    Args:
        capital: Total trading capital.
        atr: Current Average True Range value.
        entry_price: Entry price for the trade.
        atr_multiplier: ATR multiplier for stop loss (default: 2.0).
        risk_per_trade: Percentage of capital to risk (default: 2%).

    Returns:
        Position size in units (number of shares/contracts).

    Example:
        >>> units = atr_based_size(
        ...     capital=100000,
        ...     atr=2.5,
        ...     entry_price=50.0,
        ...     atr_multiplier=2.0,
        ... )
        >>> print(f"Buy {units:.0f} shares")
    """
    risk_amount = capital * risk_per_trade

    # Stop loss is ATR_multiplier * ATR away from entry
    stop_distance = atr * atr_multiplier

    if stop_distance <= 0:
        logger.warning(f"Invalid stop distance: {stop_distance}")
        return 0

    # Calculate units to risk exact amount
    units = risk_amount / stop_distance

    # Sanity check: don't exceed capital
    dollar_size = units * entry_price
    if dollar_size > capital:
        logger.warning(f"Position size ${dollar_size:.2f} exceeds capital, capping")
        units = capital / entry_price

    return units


def calculate_position_sizes(
    df: pl.DataFrame,
    capital: float,
    method: str = "fixed_fractional",
    risk_per_trade: float = 0.02,
    entry_col: str = "close",
    signal_col: str = "signal_direction",
    atr_col: Optional[str] = None,
    volatility_col: Optional[str] = None,
    **kwargs,
) -> pl.DataFrame:
    """Apply position sizing strategy to DataFrame with trading signals.

    Args:
        df: DataFrame containing price and signal data.
        capital: Total trading capital.
        method: Sizing method ("fixed_fractional", "atr", "volatility").
        risk_per_trade: Risk percentage for fixed fractional and ATR methods.
        entry_col: Column name for entry price.
        signal_col: Column name for trading signals.
        atr_col: Column name for ATR (required for "atr" method).
        volatility_col: Column name for volatility (required for "volatility" method).
        **kwargs: Additional parameters passed to sizing functions.

    Returns:
        DataFrame with added position_size and position_units columns.

    Example:
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 98],
        ...     "signal_direction": [1, 0, -1],
        ...     "atr": [2.5, 2.3, 2.7],
        ... })
        >>> result = calculate_position_sizes(
        ...     df, capital=100000, method="atr", atr_col="atr"
        ... )
    """
    if method == "fixed_fractional":
        size = fixed_fractional_size(capital, risk_per_trade)
        df = df.with_columns([
            pl.lit(size).alias("position_size"),
            (pl.lit(size) / pl.col(entry_col)).alias("position_units"),
        ])

    elif method == "atr":
        if not atr_col or atr_col not in df.columns:
            raise ValueError(f"ATR column '{atr_col}' required for atr method")

        atr_multiplier = kwargs.get("atr_multiplier", 2.0)

        # Vectorized ATR-based sizing
        risk_amount = capital * risk_per_trade
        stop_distance = df[atr_col] * atr_multiplier
        units = (risk_amount / stop_distance).fill_nan(0).fill_null(0)

        # Cap at capital limit
        dollar_size = units * df[entry_col]
        units = pl.when(dollar_size > capital).then(capital / pl.col(entry_col)).otherwise(units)

        df = df.with_columns([
            units.alias("position_units"),
            (units * pl.col(entry_col)).alias("position_size"),
        ])

    elif method == "volatility":
        if not volatility_col or volatility_col not in df.columns:
            raise ValueError(f"Volatility column '{volatility_col}' required for volatility method")

        target_vol = kwargs.get("target_volatility", 0.15)

        # Vectorized volatility-based sizing
        position_fraction = (target_vol / df[volatility_col]).clip(upper_bound=1.0)
        position_size = capital * position_fraction
        units = position_size / df[entry_col]

        df = df.with_columns([
            units.alias("position_units"),
            position_size.alias("position_size"),
        ])

    else:
        raise ValueError(f"Unknown sizing method: {method}")

    # Zero out position for no-signal rows
    df = df.with_columns([
        pl.when(pl.col(signal_col) == 0).then(0).otherwise(pl.col("position_units")).alias("position_units"),
        pl.when(pl.col(signal_col) == 0).then(0).otherwise(pl.col("position_size")).alias("position_size"),
    ])

    return df


def get_recommended_sizing(
    risk_tolerance: str = "medium",
    account_size: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Get recommended position sizing parameters for different risk levels.

    Args:
        risk_tolerance: Risk level ("conservative", "medium", "aggressive").
        account_size: Optional account size for specific recommendations.

    Returns:
        Dictionary with recommended sizing parameters.

    Example:
        >>> params = get_recommended_sizing("conservative", account_size=100000)
        >>> print(f"Risk per trade: {params['risk_per_trade']:.1%}")
    """
    profiles = {
        "conservative": {
            "method": "fixed_fractional",
            "risk_per_trade": 0.01,  # 1% per trade
            "kelly_fraction": 0.1,  # Very fractional Kelly
            "atr_multiplier": 3.0,  # Wider stops
            "target_volatility": 0.10,  # 10% annual vol
            "max_positions": 3,
        },
        "medium": {
            "method": "atr",
            "risk_per_trade": 0.02,  # 2% per trade
            "kelly_fraction": 0.25,  # Quarter Kelly
            "atr_multiplier": 2.0,  # Standard stops
            "target_volatility": 0.15,  # 15% annual vol
            "max_positions": 5,
        },
        "aggressive": {
            "method": "volatility",
            "risk_per_trade": 0.05,  # 5% per trade
            "kelly_fraction": 0.5,  # Half Kelly
            "atr_multiplier": 1.5,  # Tighter stops
            "target_volatility": 0.25,  # 25% annual vol
            "max_positions": 10,
        },
    }

    if risk_tolerance not in profiles:
        logger.warning(f"Unknown risk tolerance: {risk_tolerance}, using 'medium'")
        risk_tolerance = "medium"

    params = profiles[risk_tolerance].copy()

    if account_size:
        params["max_position_size"] = account_size * params["risk_per_trade"] * params["max_positions"]
        params["account_size"] = account_size

    return params


def calculate_optimal_units(
    entry_price: float,
    stop_loss: float,
    capital: float,
    risk_pct: float = 0.02,
    max_capital_pct: float = 0.25,
) -> Dict[str, float]:
    """Calculate optimal number of units to buy given entry, stop, and risk.

    Args:
        entry_price: Entry price per unit.
        stop_loss: Stop loss price.
        capital: Total trading capital.
        risk_pct: Percentage of capital to risk (default: 2%).
        max_capital_pct: Maximum percentage of capital for position (default: 25%).

    Returns:
        Dictionary with units, dollar_size, risk_amount, and risk_pct.

    Example:
        >>> result = calculate_optimal_units(
        ...     entry_price=50.0,
        ...     stop_loss=48.0,
        ...     capital=100000,
        ... )
        >>> print(f"Buy {result['units']:.0f} shares")
    """
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_loss)

    if price_risk <= 0:
        raise ValueError(f"Invalid price risk: {price_risk}")

    # Calculate units based on risk
    units = risk_amount / price_risk

    # Calculate dollar value of position
    dollar_size = units * entry_price

    # Cap at max capital percentage
    max_dollar_size = capital * max_capital_pct
    if dollar_size > max_dollar_size:
        logger.warning(f"Position ${dollar_size:.2f} exceeds {max_capital_pct:.0%} cap, reducing")
        dollar_size = max_dollar_size
        units = dollar_size / entry_price

    actual_risk_pct = (units * price_risk) / capital

    return {
        "units": units,
        "dollar_size": dollar_size,
        "risk_amount": units * price_risk,
        "risk_pct": actual_risk_pct,
        "position_pct": dollar_size / capital,
    }
