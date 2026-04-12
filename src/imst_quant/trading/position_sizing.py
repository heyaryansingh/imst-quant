"""Position sizing strategies for portfolio management.

This module provides various position sizing algorithms to determine
optimal trade sizes based on risk parameters, account equity, and
signal strength. Supports fixed, volatility-adjusted, Kelly criterion,
and risk parity approaches.

Features:
    - Fixed fractional position sizing
    - Volatility-adjusted sizing (ATR-based)
    - Kelly criterion optimization
    - Risk parity allocation
    - Maximum position limits
    - Portfolio-level risk constraints

Example:
    >>> from imst_quant.trading.position_sizing import PositionSizer
    >>> sizer = PositionSizer(
    ...     account_equity=100000.0,
    ...     max_position_pct=0.10,
    ...     risk_per_trade_pct=0.02,
    ... )
    >>> size = sizer.calculate_position(
    ...     symbol="AAPL",
    ...     entry_price=150.0,
    ...     stop_loss=145.0,
    ...     signal_strength=0.8,
    ... )
    >>> print(f"Position size: {size['shares']} shares")

Note:
    Position sizing is critical for risk management. These methods
    provide starting points but should be combined with proper
    portfolio-level risk controls.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


class SizingMethod(str, Enum):
    """Position sizing method."""

    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class PositionConfig:
    """Configuration for position sizing.

    Attributes:
        max_position_pct: Maximum position size as % of equity (0.0-1.0).
        risk_per_trade_pct: Risk per trade as % of equity (0.0-1.0).
        max_portfolio_heat: Maximum total portfolio risk (0.0-1.0).
        volatility_lookback: Days for volatility calculation.
        kelly_fraction: Fraction of Kelly criterion to use (0.0-1.0).
        min_shares: Minimum shares per position.
        round_to_lots: Round to lot sizes (e.g., 100 shares).
    """

    max_position_pct: float = 0.10
    risk_per_trade_pct: float = 0.02
    max_portfolio_heat: float = 0.20
    volatility_lookback: int = 20
    kelly_fraction: float = 0.25
    min_shares: int = 1
    round_to_lots: int = 1


class PositionSizer:
    """Calculate optimal position sizes for trades.

    This class provides methods to determine trade sizes based on
    various risk-adjusted strategies. It considers account equity,
    risk tolerance, volatility, and portfolio constraints.

    Attributes:
        account_equity: Current account equity in USD.
        config: Position sizing configuration.
        open_positions: Dict of currently open positions by symbol.

    Example:
        >>> sizer = PositionSizer(account_equity=50000.0)
        >>> result = sizer.fixed_fractional(
        ...     entry_price=100.0,
        ...     stop_loss=95.0,
        ... )
        >>> print(f"Buy {result['shares']} shares")
    """

    def __init__(
        self,
        account_equity: float,
        config: Optional[PositionConfig] = None,
    ) -> None:
        """Initialize the position sizer.

        Args:
            account_equity: Current account equity in USD.
            config: Optional position sizing configuration.
        """
        self.account_equity = account_equity
        self.config = config or PositionConfig()
        self.open_positions: Dict[str, Dict] = {}

        logger.info(
            "position_sizer_initialized",
            equity=account_equity,
            max_position_pct=self.config.max_position_pct,
        )

    def update_equity(self, new_equity: float) -> None:
        """Update account equity for sizing calculations.

        Args:
            new_equity: New account equity value.
        """
        self.account_equity = new_equity
        logger.debug("equity_updated", new_equity=new_equity)

    def add_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
    ) -> None:
        """Track an open position for portfolio heat calculation.

        Args:
            symbol: Asset symbol.
            shares: Number of shares.
            entry_price: Entry price per share.
            stop_loss: Stop loss price.
        """
        risk_per_share = abs(entry_price - stop_loss)
        position_risk = risk_per_share * shares

        self.open_positions[symbol] = {
            "shares": shares,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_risk": position_risk,
        }

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position.

        Args:
            symbol: Asset symbol to remove.
        """
        self.open_positions.pop(symbol, None)

    def get_portfolio_heat(self) -> float:
        """Calculate total portfolio risk from open positions.

        Returns:
            Total risk as a percentage of equity.
        """
        total_risk = sum(
            pos["position_risk"] for pos in self.open_positions.values()
        )
        return total_risk / self.account_equity if self.account_equity > 0 else 0.0

    def get_available_risk(self) -> float:
        """Calculate available risk capacity.

        Returns:
            Available risk as a percentage of equity.
        """
        current_heat = self.get_portfolio_heat()
        return max(0.0, self.config.max_portfolio_heat - current_heat)

    def fixed_fractional(
        self,
        entry_price: float,
        stop_loss: float,
        signal_strength: float = 1.0,
    ) -> Dict:
        """Calculate position size using fixed fractional method.

        This method sizes positions based on a fixed percentage of
        equity risked per trade, adjusted by signal strength.

        Args:
            entry_price: Expected entry price.
            stop_loss: Stop loss price level.
            signal_strength: Signal strength multiplier (0.0-1.0).

        Returns:
            Dictionary containing:
                - shares: Number of shares to buy
                - position_value: Total position value
                - risk_amount: Dollar risk on the trade
                - risk_pct: Risk as percentage of equity
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return self._zero_position("Invalid stop loss")

        # Calculate risk amount
        base_risk = self.account_equity * self.config.risk_per_trade_pct
        adjusted_risk = base_risk * min(max(signal_strength, 0.0), 1.0)

        # Check portfolio heat
        available_risk = self.get_available_risk()
        risk_pct = adjusted_risk / self.account_equity
        if risk_pct > available_risk:
            adjusted_risk = available_risk * self.account_equity

        # Calculate shares
        shares = int(adjusted_risk / risk_per_share)

        # Apply position limits
        max_shares = int(
            self.account_equity * self.config.max_position_pct / entry_price
        )
        shares = min(shares, max_shares)

        # Round to lots
        if self.config.round_to_lots > 1:
            shares = (shares // self.config.round_to_lots) * self.config.round_to_lots

        # Enforce minimum
        if shares < self.config.min_shares:
            shares = 0

        position_value = shares * entry_price
        risk_amount = shares * risk_per_share

        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_pct": risk_amount / self.account_equity if self.account_equity > 0 else 0.0,
            "method": "fixed_fractional",
        }

    def volatility_adjusted(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        signal_strength: float = 1.0,
    ) -> Dict:
        """Calculate position size adjusted for volatility (ATR).

        This method uses Average True Range to set stop loss distance
        and size positions accordingly. Higher volatility = smaller position.

        Args:
            entry_price: Expected entry price.
            atr: Average True Range value.
            atr_multiplier: ATR multiplier for stop distance.
            signal_strength: Signal strength multiplier (0.0-1.0).

        Returns:
            Dictionary containing position sizing details.
        """
        if atr <= 0:
            return self._zero_position("Invalid ATR value")

        stop_distance = atr * atr_multiplier
        stop_loss = entry_price - stop_distance  # Assumes long position

        result = self.fixed_fractional(
            entry_price=entry_price,
            stop_loss=stop_loss,
            signal_strength=signal_strength,
        )
        result["method"] = "volatility_adjusted"
        result["atr"] = atr
        result["stop_distance"] = stop_distance

        return result

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
    ) -> Dict:
        """Calculate position size using Kelly criterion.

        The Kelly criterion optimizes position size for maximum
        geometric growth rate. This implementation uses a fractional
        Kelly for reduced volatility.

        Args:
            win_rate: Historical win rate (0.0-1.0).
            avg_win: Average winning trade (dollars or R-multiple).
            avg_loss: Average losing trade (positive value).
            entry_price: Expected entry price.

        Returns:
            Dictionary containing:
                - shares: Number of shares
                - kelly_pct: Full Kelly percentage
                - adjusted_pct: Fractional Kelly used
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self._zero_position("Invalid Kelly parameters")

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_pct = (b * p - q) / b

        # Cap at max position and apply Kelly fraction
        if kelly_pct <= 0:
            return self._zero_position("Negative Kelly - no edge")

        adjusted_pct = min(
            kelly_pct * self.config.kelly_fraction,
            self.config.max_position_pct,
        )

        position_value = self.account_equity * adjusted_pct
        shares = int(position_value / entry_price)

        if self.config.round_to_lots > 1:
            shares = (shares // self.config.round_to_lots) * self.config.round_to_lots

        return {
            "shares": shares,
            "position_value": shares * entry_price,
            "kelly_pct": kelly_pct,
            "adjusted_pct": adjusted_pct,
            "method": "kelly",
        }

    def risk_parity(
        self,
        assets: List[str],
        volatilities: Dict[str, float],
        prices: Dict[str, float],
        target_risk: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """Allocate capital using risk parity (equal risk contribution).

        Each asset contributes equally to total portfolio risk,
        so lower volatility assets get larger allocations.

        Args:
            assets: List of asset symbols to allocate.
            volatilities: Dict of symbol -> annualized volatility.
            prices: Dict of symbol -> current price.
            target_risk: Optional target portfolio volatility.

        Returns:
            Dictionary mapping symbols to allocation details.
        """
        if not assets:
            return {}

        target = target_risk or self.config.risk_per_trade_pct * 5

        # Calculate inverse volatility weights
        inv_vols = {}
        total_inv_vol = 0.0
        for asset in assets:
            vol = volatilities.get(asset, 0.20)  # Default 20% vol
            if vol > 0:
                inv_vols[asset] = 1.0 / vol
                total_inv_vol += inv_vols[asset]
            else:
                inv_vols[asset] = 0.0

        allocations = {}
        for asset in assets:
            if total_inv_vol > 0:
                weight = inv_vols[asset] / total_inv_vol
            else:
                weight = 1.0 / len(assets)

            # Cap at max position
            weight = min(weight, self.config.max_position_pct)

            position_value = self.account_equity * weight
            price = prices.get(asset, 0.0)
            shares = int(position_value / price) if price > 0 else 0

            allocations[asset] = {
                "shares": shares,
                "weight": weight,
                "position_value": shares * price if price > 0 else 0,
                "volatility": volatilities.get(asset, 0.0),
            }

        return allocations

    def equal_weight(
        self,
        assets: List[str],
        prices: Dict[str, float],
    ) -> Dict[str, Dict]:
        """Allocate equal dollar amounts to each asset.

        Args:
            assets: List of asset symbols to allocate.
            prices: Dict of symbol -> current price.

        Returns:
            Dictionary mapping symbols to allocation details.
        """
        if not assets:
            return {}

        weight_per_asset = min(
            1.0 / len(assets),
            self.config.max_position_pct,
        )

        allocations = {}
        for asset in assets:
            position_value = self.account_equity * weight_per_asset
            price = prices.get(asset, 0.0)
            shares = int(position_value / price) if price > 0 else 0

            allocations[asset] = {
                "shares": shares,
                "weight": weight_per_asset,
                "position_value": shares * price if price > 0 else 0,
            }

        return allocations

    def calculate_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        atr: Optional[float] = None,
        signal_strength: float = 1.0,
        method: SizingMethod = SizingMethod.FIXED_FRACTIONAL,
    ) -> Dict:
        """Calculate position size using the specified method.

        This is the main entry point for position sizing. It dispatches
        to the appropriate method based on the sizing strategy.

        Args:
            symbol: Asset symbol.
            entry_price: Expected entry price.
            stop_loss: Stop loss price (required for fixed_fractional).
            atr: Average True Range (required for volatility_adjusted).
            signal_strength: Signal strength multiplier (0.0-1.0).
            method: Position sizing method to use.

        Returns:
            Dictionary containing position sizing details.
        """
        result: Dict
        if method == SizingMethod.FIXED_FRACTIONAL:
            if stop_loss is None:
                return self._zero_position("Stop loss required for fixed_fractional")
            result = self.fixed_fractional(entry_price, stop_loss, signal_strength)

        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            if atr is None:
                return self._zero_position("ATR required for volatility_adjusted")
            result = self.volatility_adjusted(entry_price, atr, signal_strength=signal_strength)

        else:
            return self._zero_position(f"Unsupported method: {method}")

        result["symbol"] = symbol
        return result

    def _zero_position(self, reason: str) -> Dict:
        """Return a zero position with reason.

        Args:
            reason: Explanation for zero sizing.

        Returns:
            Dictionary with zero shares and explanation.
        """
        return {
            "shares": 0,
            "position_value": 0.0,
            "risk_amount": 0.0,
            "risk_pct": 0.0,
            "reason": reason,
        }
