"""Market impact modeling for trade execution cost estimation.

Implements multiple market impact models to estimate price impact
before executing trades, helping optimize trade sizing and timing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace


def _participation(trade_size: float, adv: float) -> float:
    """Order size as a fraction of ADV; side-agnostic so sells cost like buys.

    Raises:
        ValueError: If ``adv`` is not a positive finite number.
    """
    if not np.isfinite(adv) or adv <= 0:
        raise ValueError(f"adv must be a positive finite number, got {adv}")
    return abs(trade_size) / adv


@dataclass
class ImpactParameters:
    """Parameters for market impact models."""

    # Almgren-Chriss parameters
    permanent_impact: float = 0.1  # Permanent market impact coefficient
    temporary_impact: float = 0.5   # Temporary market impact coefficient
    volatility: float = 0.02        # Daily volatility

    # Square-root model parameters
    daily_volume: float = 1e6       # Average daily volume (shares)
    spread_bps: float = 5.0         # Bid-ask spread in basis points

    # Nonlinear impact parameters
    power_law_exponent: float = 0.6  # Exponent for nonlinear impact


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model.

    Models market impact as permanent + temporary components.
    Permanent impact: shifts market price permanently
    Temporary impact: dissipates after trade completes
    """

    def __init__(self, params: ImpactParameters):
        """Initialize with impact parameters."""
        self.params = params

    def permanent_impact(self, trade_size: float, adv: float) -> float:
        """Calculate permanent market impact.

        Args:
            trade_size: Size of the trade (shares)
            adv: Average daily volume (shares)

        Returns:
            Permanent impact as fraction of price
        """
        return self.params.permanent_impact * _participation(trade_size, adv)

    def temporary_impact(self, trade_rate: float, adv: float) -> float:
        """Calculate temporary market impact.

        Args:
            trade_rate: Trading rate (shares per unit time)
            adv: Average daily volume (shares)

        Returns:
            Temporary impact as fraction of price
        """
        return self.params.temporary_impact * _participation(trade_rate, adv)

    def optimal_trajectory(
        self,
        total_shares: float,
        time_horizon: float,
        adv: float,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """Calculate optimal trading trajectory.

        Args:
            total_shares: Total shares to trade
            time_horizon: Time to complete trade (in days)
            adv: Average daily volume
            risk_aversion: Risk aversion parameter (higher = trade slower)

        Returns:
            Array of optimal trade sizes over time; they sum to
            ``total_shares`` so the position is fully worked by the horizon.
        """
        if time_horizon <= 0:
            raise ValueError(f"time_horizon must be positive, got {time_horizon}")
        n_periods = max(1, int(time_horizon * 24))  # Hourly periods

        # Almgren-Chriss holdings x(t) = X sinh(kappa (T - t)) / sinh(kappa T);
        # kappa -> 0 (no risk aversion) is the linear TWAP schedule.
        kappa = np.sqrt(
            risk_aversion * self.params.volatility**2 /
            (2 * self.params.temporary_impact)
        )
        remaining = 1.0 - np.linspace(0.0, 1.0, n_periods + 1)
        if kappa * time_horizon > 1e-12:
            remaining = (
                np.sinh(kappa * time_horizon * remaining)
                / np.sinh(kappa * time_horizon)
            )
        trajectory = total_shares * remaining

        return -np.diff(trajectory)

    def expected_cost(
        self,
        total_shares: float,
        time_horizon: float,
        adv: float,
        current_price: float,
        risk_aversion: float = 1.0
    ) -> Dict[str, float]:
        """Calculate expected execution cost.

        Args:
            total_shares: Total shares to trade
            time_horizon: Time to complete trade (in days)
            adv: Average daily volume
            current_price: Current market price
            risk_aversion: Risk aversion parameter

        Returns:
            Dictionary with cost breakdown
        """
        trajectory = self.optimal_trajectory(
            total_shares, time_horizon, adv, risk_aversion
        )

        # Permanent impact cost
        perm_impact = self.permanent_impact(total_shares, adv)
        perm_cost = 0.5 * perm_impact * abs(total_shares) * current_price

        # Temporary impact cost
        temp_cost = 0
        for trade_size in trajectory:
            trade_rate = trade_size / (time_horizon / len(trajectory))
            temp_impact = self.temporary_impact(trade_rate, adv)
            temp_cost += temp_impact * abs(trade_size) * current_price

        total_cost = perm_cost + temp_cost

        return {
            'permanent_cost': perm_cost,
            'temporary_cost': temp_cost,
            'total_cost': total_cost,
            'cost_bps': (total_cost / abs(total_shares * current_price)) * 10000
            if total_shares and current_price else 0.0,
            'optimal_periods': len(trajectory)
        }


class SquareRootModel:
    """Square-root market impact model.

    Impact scales with square root of order size relative to volume.
    Commonly used for simple impact estimation.
    """

    def __init__(self, params: ImpactParameters):
        """Initialize with impact parameters."""
        self.params = params

    def estimate_impact(
        self,
        trade_size: float,
        current_price: float,
        adv: Optional[float] = None
    ) -> Dict[str, float]:
        """Estimate market impact using square-root model.

        Args:
            trade_size: Size of the trade (shares)
            current_price: Current market price
            adv: Average daily volume (shares), uses params default if None

        Returns:
            Dictionary with impact estimates
        """
        adv = self.params.daily_volume if adv is None else adv
        participation = _participation(trade_size, adv)
        trade_size = abs(trade_size)

        # Square-root impact
        impact_fraction = self.params.volatility * np.sqrt(participation)

        # Add spread cost
        spread_cost = (self.params.spread_bps / 10000) * current_price

        impact_dollars = impact_fraction * current_price * trade_size
        spread_dollars = spread_cost * trade_size / 2  # Half spread

        total_cost = impact_dollars + spread_dollars
        notional = trade_size * current_price

        return {
            'impact_fraction': impact_fraction,
            'impact_bps': impact_fraction * 10000,
            'spread_cost_bps': self.params.spread_bps / 2,
            'total_cost': total_cost,
            'total_cost_bps': (total_cost / notional * 10000) if notional else 0.0,
            'participation_rate': participation
        }


class NonlinearImpactModel:
    """Nonlinear market impact model with power-law scaling.

    Impact = alpha * (size / ADV)^beta
    where beta typically ranges from 0.5 to 0.7
    """

    def __init__(self, params: ImpactParameters):
        """Initialize with impact parameters."""
        self.params = params

    def estimate_impact(
        self,
        trade_size: float,
        current_price: float,
        adv: Optional[float] = None,
        alpha: float = 0.1
    ) -> Dict[str, float]:
        """Estimate nonlinear market impact.

        Args:
            trade_size: Size of the trade (shares)
            current_price: Current market price
            adv: Average daily volume (shares)
            alpha: Impact coefficient

        Returns:
            Dictionary with impact estimates
        """
        adv = self.params.daily_volume if adv is None else adv
        participation = _participation(trade_size, adv)
        trade_size = abs(trade_size)
        beta = self.params.power_law_exponent

        # Nonlinear impact
        impact_fraction = alpha * (participation ** beta)
        impact_dollars = impact_fraction * current_price * trade_size

        return {
            'impact_fraction': impact_fraction,
            'impact_bps': impact_fraction * 10000,
            'total_cost': impact_dollars,
            'participation_rate': participation,
            'power_exponent': beta
        }


class VolumeProfileAnalyzer:
    """Analyze intraday volume profiles for execution timing."""

    @staticmethod
    def fit_volume_curve(
        historical_volume: pd.DataFrame,
        intraday_intervals: int = 78  # 5-min intervals in trading day
    ) -> np.ndarray:
        """Fit typical intraday volume curve.

        Args:
            historical_volume: DataFrame with intraday volume data
            intraday_intervals: Number of intraday time periods

        Returns:
            Array of volume fractions for each interval
        """
        # Reshape to intraday intervals
        volume_by_interval = historical_volume.values.reshape(-1, intraday_intervals)

        # Average volume profile
        avg_profile = volume_by_interval.mean(axis=0)

        # Normalize to fractions
        volume_fractions = avg_profile / avg_profile.sum()

        return volume_fractions

    @staticmethod
    def vwap_optimal_schedule(
        total_shares: float,
        volume_curve: np.ndarray
    ) -> np.ndarray:
        """Calculate VWAP-optimal execution schedule.

        Trades proportionally to expected volume to minimize impact.

        Args:
            total_shares: Total shares to trade
            volume_curve: Expected volume fractions per interval

        Returns:
            Array of share quantities per interval
        """
        # Trade proportionally to volume
        schedule = total_shares * volume_curve

        return schedule


class AdaptiveImpactModel:
    """Adaptive market impact model combining multiple approaches.

    Selects appropriate model based on market conditions and order size.
    """

    def __init__(self, params: ImpactParameters):
        """Initialize with impact parameters."""
        self.params = params
        self.almgren_chriss = AlmgrenChrissModel(params)
        self.sqrt_model = SquareRootModel(params)
        self.nonlinear = NonlinearImpactModel(params)

    def estimate_impact(
        self,
        trade_size: float,
        current_price: float,
        adv: float,
        time_horizon: Optional[float] = None,
        market_volatility: Optional[float] = None
    ) -> Dict[str, any]:
        """Estimate impact using adaptive model selection.

        Args:
            trade_size: Size of the trade (shares)
            current_price: Current market price
            adv: Average daily volume
            time_horizon: Optional execution time horizon (days)
            market_volatility: Optional current market volatility

        Returns:
            Dictionary with impact estimates from multiple models
        """
        participation = _participation(trade_size, adv)
        # Current volatility overrides the calibrated one for this estimate;
        # it was documented but ignored, so a vol spike never raised costs.
        model = self
        if market_volatility is not None:
            if not np.isfinite(market_volatility) or market_volatility < 0:
                raise ValueError(
                    f"market_volatility must be a non-negative finite number, got {market_volatility}"
                )
            model = AdaptiveImpactModel(replace(self.params, volatility=market_volatility))

        results = {
            'trade_size': trade_size,
            'adv': adv,
            'participation_rate': participation,
            'current_price': current_price
        }

        # Small order (< 5% ADV): use simple square-root
        if participation < 0.05:
            results['recommended_model'] = 'square_root'
            results['square_root'] = model.sqrt_model.estimate_impact(
                trade_size, current_price, adv
            )

        # Medium order (5-20% ADV): use nonlinear if available
        elif participation < 0.20:
            results['recommended_model'] = 'nonlinear'
            results['nonlinear'] = model.nonlinear.estimate_impact(
                trade_size, current_price, adv
            )
            results['square_root'] = model.sqrt_model.estimate_impact(
                trade_size, current_price, adv
            )

        # Large order (> 20% ADV): use Almgren-Chriss with slicing
        else:
            results['recommended_model'] = 'almgren_chriss'
            if time_horizon is None:
                # Default: spread over multiple days based on participation
                time_horizon = max(1, participation / 0.10)  # ~10% ADV per day

            results['almgren_chriss'] = model.almgren_chriss.expected_cost(
                trade_size, time_horizon, adv, current_price
            )
            results['time_horizon_days'] = time_horizon

        return results


def estimate_slippage(
    orders: pd.DataFrame,
    execution_prices: pd.DataFrame,
    arrival_prices: pd.DataFrame
) -> pd.DataFrame:
    """Calculate realized slippage from executed orders.

    Args:
        orders: DataFrame with order details (size, side)
        execution_prices: Actual execution prices
        arrival_prices: Arrival prices when decision was made

    Returns:
        DataFrame with slippage analysis
    """
    slippage = pd.DataFrame({
        'order_id': orders.index,
        'side': orders['side'],
        'size': orders['size'],
        'arrival_price': arrival_prices.values,
        'execution_price': execution_prices.values
    })

    # Calculate slippage
    slippage['price_diff'] = slippage.apply(
        lambda x: (x['execution_price'] - x['arrival_price'])
        if x['side'] == 'buy'
        else (x['arrival_price'] - x['execution_price']),
        axis=1
    )

    slippage['slippage_bps'] = (
        slippage['price_diff'] / slippage['arrival_price'] * 10000
    )
    slippage['slippage_dollars'] = slippage['price_diff'] * slippage['size']

    # Summary statistics
    slippage['cumulative_slippage'] = slippage['slippage_dollars'].cumsum()

    return slippage
