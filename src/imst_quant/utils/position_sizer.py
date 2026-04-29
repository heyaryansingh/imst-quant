"""Advanced position sizing strategies for risk management.

Implements Kelly Criterion, Risk Parity, Volatility Targeting, and Fixed Fraction methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal
import structlog

logger = structlog.get_logger()


class PositionSizer:
    """Calculate optimal position sizes using various strategies."""

    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = 0.10,
        max_portfolio_risk: float = 0.02
    ):
        """Initialize position sizer.

        Args:
            portfolio_value: Total portfolio value in dollars
            max_position_pct: Maximum position size as % of portfolio (default: 10%)
            max_portfolio_risk: Maximum portfolio risk per trade (default: 2%)
        """
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """Calculate position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of Kelly to use for safety (default: 0.25 = quarter Kelly)

        Returns:
            Position size in dollars
        """
        if avg_loss == 0:
            logger.warning("avg_loss_zero", using_fallback=True)
            return self.fixed_fraction(0.01)

        # Kelly formula: f = (p * b - q) / b
        # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        b = avg_win / avg_loss
        kelly_pct = (win_rate * b - (1 - win_rate)) / b

        # Apply safety fraction and bounds
        kelly_pct = max(0, min(kelly_pct, 1.0)) * kelly_fraction
        position_size = self.portfolio_value * kelly_pct

        # Apply position limits
        max_position = self.portfolio_value * self.max_position_pct
        final_size = min(position_size, max_position)

        logger.info(
            "kelly_position_calculated",
            kelly_pct=kelly_pct,
            position_size=final_size,
            win_rate=win_rate,
            kelly_fraction=kelly_fraction
        )
        return final_size

    def fixed_fraction(self, risk_fraction: float) -> float:
        """Calculate position size using fixed fractional method.

        Args:
            risk_fraction: Fraction of portfolio to risk (e.g., 0.02 = 2%)

        Returns:
            Position size in dollars
        """
        position_size = self.portfolio_value * risk_fraction
        max_position = self.portfolio_value * self.max_position_pct
        final_size = min(position_size, max_position)

        logger.info(
            "fixed_fraction_position_calculated",
            risk_fraction=risk_fraction,
            position_size=final_size
        )
        return final_size

    def volatility_targeting(
        self,
        asset_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """Calculate position size to target specific portfolio volatility.

        Args:
            asset_volatility: Annualized volatility of the asset
            target_volatility: Target portfolio volatility (default: 15%)

        Returns:
            Position size in dollars
        """
        if asset_volatility == 0:
            logger.warning("asset_volatility_zero", using_fallback=True)
            return self.fixed_fraction(0.01)

        # Scale position inversely with volatility
        position_pct = target_volatility / asset_volatility
        position_size = self.portfolio_value * position_pct

        # Apply position limits
        max_position = self.portfolio_value * self.max_position_pct
        final_size = min(position_size, max_position)

        logger.info(
            "volatility_target_position_calculated",
            asset_vol=asset_volatility,
            target_vol=target_volatility,
            position_size=final_size
        )
        return final_size

    def risk_based(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """Calculate position size based on stop-loss risk.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop-loss price
            risk_per_trade: Dollar amount to risk (if None, uses max_portfolio_risk)

        Returns:
            Number of shares to buy
        """
        if risk_per_trade is None:
            risk_per_trade = self.portfolio_value * self.max_portfolio_risk

        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            logger.warning("price_risk_zero", using_fallback=True)
            return 0

        shares = risk_per_trade / price_risk

        # Apply position limits
        max_shares = (self.portfolio_value * self.max_position_pct) / entry_price
        final_shares = min(shares, max_shares)

        logger.info(
            "risk_based_position_calculated",
            shares=final_shares,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            risk_amount=risk_per_trade
        )
        return final_shares

    def equal_weight(self, num_positions: int) -> float:
        """Calculate equal-weight position size.

        Args:
            num_positions: Number of positions in portfolio

        Returns:
            Position size in dollars
        """
        if num_positions <= 0:
            logger.warning("invalid_num_positions", num_positions=num_positions)
            return 0

        position_size = self.portfolio_value / num_positions
        max_position = self.portfolio_value * self.max_position_pct
        final_size = min(position_size, max_position)

        logger.info(
            "equal_weight_position_calculated",
            num_positions=num_positions,
            position_size=final_size
        )
        return final_size


def risk_parity_weights(
    returns: pd.DataFrame,
    target_risk: float = 0.10
) -> pd.Series:
    """Calculate risk parity portfolio weights.

    Each asset contributes equally to portfolio risk.

    Args:
        returns: DataFrame of asset returns (columns = assets)
        target_risk: Target portfolio volatility

    Returns:
        Series of portfolio weights (sums to 1)
    """
    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Calculate inverse volatility weights as starting point
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vol_weights = (1 / vols) / (1 / vols).sum()

    # Iteratively adjust to achieve risk parity
    weights = inv_vol_weights.copy()
    for _ in range(100):  # Max iterations
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights / portfolio_vol

        # Adjust weights to equalize risk contributions
        target_marginal_risk = marginal_risk.mean()
        weights *= target_marginal_risk / marginal_risk
        weights /= weights.sum()

        # Check convergence
        risk_contributions = weights * marginal_risk
        if risk_contributions.std() < 1e-6:
            break

    # Scale to target risk
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    weights *= target_risk / portfolio_vol

    logger.info(
        "risk_parity_weights_calculated",
        num_assets=len(weights),
        target_risk=target_risk,
        actual_risk=float(np.sqrt(weights @ cov_matrix @ weights))
    )

    return pd.Series(weights, index=returns.columns)


def calculate_leverage(
    position_sizes: Dict[str, float],
    portfolio_value: float
) -> Dict[str, float]:
    """Calculate leverage metrics for a portfolio.

    Args:
        position_sizes: Dictionary of asset -> position size (in dollars)
        portfolio_value: Total portfolio value

    Returns:
        Dictionary with leverage metrics
    """
    total_exposure = sum(abs(size) for size in position_sizes.values())
    gross_leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

    long_exposure = sum(size for size in position_sizes.values() if size > 0)
    short_exposure = abs(sum(size for size in position_sizes.values() if size < 0))
    net_leverage = (long_exposure - short_exposure) / portfolio_value if portfolio_value > 0 else 0

    logger.info(
        "leverage_calculated",
        gross_leverage=gross_leverage,
        net_leverage=net_leverage,
        num_positions=len(position_sizes)
    )

    return {
        "gross_leverage": gross_leverage,
        "net_leverage": net_leverage,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "total_exposure": total_exposure,
        "num_positions": len(position_sizes)
    }


def dynamic_position_sizing(
    base_size: float,
    market_regime: Literal["low_vol", "normal", "high_vol"],
    confidence_score: float = 1.0
) -> float:
    """Dynamically adjust position size based on market regime and signal confidence.

    Args:
        base_size: Base position size in dollars
        market_regime: Current market volatility regime
        confidence_score: Signal confidence (0-1)

    Returns:
        Adjusted position size
    """
    # Regime adjustments
    regime_multipliers = {
        "low_vol": 1.2,
        "normal": 1.0,
        "high_vol": 0.6
    }

    regime_mult = regime_multipliers.get(market_regime, 1.0)
    adjusted_size = base_size * regime_mult * confidence_score

    logger.info(
        "dynamic_position_sized",
        base_size=base_size,
        regime=market_regime,
        confidence=confidence_score,
        adjusted_size=adjusted_size
    )

    return adjusted_size
