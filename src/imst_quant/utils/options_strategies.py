"""Options strategy analysis and pricing utilities.

Provides tools for analyzing options strategies including hedging,
income generation, and portfolio protection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import norm


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


@dataclass
class Option:
    """Option contract specification."""
    strike: float
    expiry_days: int
    option_type: OptionType
    premium: Optional[float] = None
    quantity: int = 1


class BlackScholesModel:
    """Black-Scholes option pricing model."""

    @staticmethod
    def price(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """Calculate Black-Scholes option price.

        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Continuous dividend yield
            option_type: Call or Put

        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            if option_type == OptionType.CALL:
                return max(spot - strike, 0)
            return max(strike - spot, 0)

        d1 = (
            np.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))

        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type == OptionType.CALL:
            price = (
                spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
                - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            )
        else:
            price = (
                strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                - spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            )

        return price

    @staticmethod
    def greeks(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL
    ) -> Dict[str, float]:
        """Calculate option Greeks.

        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Continuous dividend yield
            option_type: Call or Put

        Returns:
            Dictionary with delta, gamma, vega, theta, rho
        """
        if time_to_expiry <= 0:
            return {
                'delta': 1.0 if spot > strike else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }

        d1 = (
            np.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))

        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Common terms
        pdf_d1 = norm.pdf(d1)
        discount_factor = np.exp(-dividend_yield * time_to_expiry)

        # Delta
        if option_type == OptionType.CALL:
            delta = discount_factor * norm.cdf(d1)
        else:
            delta = discount_factor * (norm.cdf(d1) - 1)

        # Gamma (same for calls and puts)
        gamma = (
            discount_factor * pdf_d1 / (spot * volatility * np.sqrt(time_to_expiry))
        )

        # Vega (same for calls and puts, convert to per 1% vol change)
        vega = spot * discount_factor * pdf_d1 * np.sqrt(time_to_expiry) / 100

        # Theta
        term1 = -(
            spot * pdf_d1 * volatility * discount_factor /
            (2 * np.sqrt(time_to_expiry))
        )

        if option_type == OptionType.CALL:
            term2 = -risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            term3 = dividend_yield * spot * discount_factor * norm.cdf(d1)
            theta = (term1 + term2 + term3) / 365
        else:
            term2 = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            term3 = -dividend_yield * spot * discount_factor * norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365

        # Rho (convert to per 1% rate change)
        if option_type == OptionType.CALL:
            rho = (
                strike * time_to_expiry *
                np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
            )
        else:
            rho = (
                -strike * time_to_expiry *
                np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
            )

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }


class CoveredCallStrategy:
    """Covered call strategy analyzer.

    Buy stock + sell call option for income generation.
    """

    @staticmethod
    def analyze(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        shares: int = 100
    ) -> Dict[str, float]:
        """Analyze covered call strategy.

        Args:
            spot: Current stock price
            strike: Call strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free rate
            shares: Number of shares (default 100)

        Returns:
            Strategy metrics including max profit, breakeven, etc.
        """
        # Price the call
        call_premium = BlackScholesModel.price(
            spot, strike, time_to_expiry, volatility, risk_free_rate,
            option_type=OptionType.CALL
        )

        # Strategy costs and payoffs
        initial_cost = spot * shares - call_premium * shares
        breakeven = spot - call_premium
        max_profit = (strike - spot + call_premium) * shares
        max_loss = initial_cost  # If stock goes to zero

        # Upside capture
        upside_capture = (strike - spot) / spot if strike > spot else 0

        return {
            'call_premium': call_premium,
            'premium_yield': call_premium / spot,
            'initial_cost': initial_cost,
            'breakeven': breakeven,
            'max_profit': max_profit,
            'max_profit_pct': max_profit / initial_cost,
            'max_loss': max_loss,
            'upside_capture': upside_capture,
            'annualized_return': (max_profit / initial_cost) / time_to_expiry
        }


class ProtectivePutStrategy:
    """Protective put strategy analyzer.

    Buy stock + buy put option for downside protection.
    """

    @staticmethod
    def analyze(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        shares: int = 100
    ) -> Dict[str, float]:
        """Analyze protective put strategy.

        Args:
            spot: Current stock price
            strike: Put strike price
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free rate
            shares: Number of shares

        Returns:
            Strategy metrics including protection cost, max loss, etc.
        """
        # Price the put
        put_premium = BlackScholesModel.price(
            spot, strike, time_to_expiry, volatility, risk_free_rate,
            option_type=OptionType.PUT
        )

        # Strategy costs and payoffs
        initial_cost = (spot + put_premium) * shares
        max_loss = (spot - strike + put_premium) * shares
        breakeven = spot + put_premium

        # Protection metrics
        protection_cost = put_premium / spot
        downside_protection = (spot - strike) / spot

        return {
            'put_premium': put_premium,
            'protection_cost_pct': protection_cost,
            'initial_cost': initial_cost,
            'breakeven': breakeven,
            'max_loss': max_loss,
            'max_loss_pct': max_loss / initial_cost,
            'downside_protected': downside_protection,
            'annualized_cost': protection_cost / time_to_expiry
        }


class CollarStrategy:
    """Collar strategy analyzer.

    Buy stock + buy put + sell call for costless or low-cost protection.
    """

    @staticmethod
    def analyze(
        spot: float,
        put_strike: float,
        call_strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        shares: int = 100
    ) -> Dict[str, float]:
        """Analyze collar strategy.

        Args:
            spot: Current stock price
            put_strike: Put strike price (downside protection)
            call_strike: Call strike price (upside limit)
            time_to_expiry: Time to expiry in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free rate
            shares: Number of shares

        Returns:
            Strategy metrics including net cost, range bounds, etc.
        """
        # Price the options
        put_premium = BlackScholesModel.price(
            spot, put_strike, time_to_expiry, volatility, risk_free_rate,
            option_type=OptionType.PUT
        )
        call_premium = BlackScholesModel.price(
            spot, call_strike, time_to_expiry, volatility, risk_free_rate,
            option_type=OptionType.CALL
        )

        # Net option cost
        net_option_cost = put_premium - call_premium

        # Strategy metrics
        initial_cost = (spot + net_option_cost) * shares
        max_profit = (call_strike - spot - net_option_cost) * shares
        max_loss = (spot - put_strike + net_option_cost) * shares

        # Range metrics
        profit_range = call_strike - put_strike
        range_pct = profit_range / spot

        return {
            'put_premium': put_premium,
            'call_premium': call_premium,
            'net_cost': net_option_cost,
            'net_cost_pct': net_option_cost / spot,
            'initial_cost': initial_cost,
            'max_profit': max_profit,
            'max_profit_pct': max_profit / initial_cost if initial_cost > 0 else 0,
            'max_loss': max_loss,
            'max_loss_pct': max_loss / initial_cost if initial_cost > 0 else 0,
            'profit_range': profit_range,
            'range_pct': range_pct,
            'is_zero_cost': abs(net_option_cost) < 0.01
        }


class DeltaHedging:
    """Delta hedging utilities for portfolio protection."""

    @staticmethod
    def calculate_hedge_ratio(
        portfolio_value: float,
        portfolio_beta: float,
        index_price: float,
        option_delta: float
    ) -> int:
        """Calculate number of option contracts needed to delta hedge.

        Args:
            portfolio_value: Total portfolio value
            portfolio_beta: Portfolio beta relative to index
            index_price: Current index price
            option_delta: Delta of the hedging option

        Returns:
            Number of option contracts needed
        """
        # Portfolio delta exposure
        portfolio_delta = portfolio_value * portfolio_beta / index_price

        # Contracts needed (assuming 100 shares per contract)
        contracts_needed = abs(portfolio_delta / (option_delta * 100))

        return int(np.ceil(contracts_needed))

    @staticmethod
    def rebalance_schedule(
        initial_delta: float,
        target_delta: float,
        gamma: float,
        spot_move_threshold: float = 0.02
    ) -> Dict[str, float]:
        """Calculate when to rebalance delta hedge.

        Args:
            initial_delta: Current portfolio delta
            target_delta: Target delta (usually 0 for delta-neutral)
            gamma: Portfolio gamma
            spot_move_threshold: Rebalance when spot moves by this fraction

        Returns:
            Rebalancing metrics
        """
        delta_drift_per_pct = gamma * 0.01
        max_delta_drift = abs(target_delta - initial_delta)

        # Spot move that causes max drift
        rebalance_trigger = max_delta_drift / abs(delta_drift_per_pct) if delta_drift_per_pct != 0 else float('inf')

        return {
            'current_delta': initial_delta,
            'target_delta': target_delta,
            'gamma': gamma,
            'delta_drift_per_1pct': delta_drift_per_pct,
            'rebalance_at_spot_move_pct': min(rebalance_trigger, spot_move_threshold * 100)
        }


class ImpliedVolatilityCalculator:
    """Calculate implied volatility from option prices."""

    @staticmethod
    def calculate_iv(
        option_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType = OptionType.CALL,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method.

        Args:
            option_price: Market price of option
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            option_type: Call or Put
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility or None if not converged
        """
        # Initial guess
        iv = 0.3

        for _ in range(max_iterations):
            # Calculate price and vega at current IV
            price = BlackScholesModel.price(
                spot, strike, time_to_expiry, iv, risk_free_rate,
                option_type=option_type
            )
            greeks = BlackScholesModel.greeks(
                spot, strike, time_to_expiry, iv, risk_free_rate,
                option_type=option_type
            )

            vega = greeks['vega'] * 100  # Convert back to per 1 vol change

            # Newton-Raphson update
            price_diff = price - option_price
            if abs(price_diff) < tolerance:
                return iv

            if abs(vega) < 1e-10:
                return None

            iv = iv - price_diff / vega

            # Keep IV in reasonable range
            iv = max(0.01, min(5.0, iv))

        return None  # Did not converge


def strategy_comparison(
    spot: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    put_strike: Optional[float] = None,
    call_strike: Optional[float] = None
) -> pd.DataFrame:
    """Compare common option strategies.

    Args:
        spot: Current stock price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        put_strike: Put strike (default 95% of spot)
        call_strike: Call strike (default 105% of spot)

    Returns:
        DataFrame comparing strategies
    """
    put_strike = put_strike or spot * 0.95
    call_strike = call_strike or spot * 1.05

    strategies = []

    # Covered call
    cc = CoveredCallStrategy.analyze(
        spot, call_strike, time_to_expiry, volatility, risk_free_rate
    )
    strategies.append({
        'strategy': 'Covered Call',
        'cost': cc['initial_cost'],
        'max_profit': cc['max_profit'],
        'max_loss': cc['max_loss'],
        'breakeven': cc['breakeven'],
        'return_on_risk': cc['max_profit'] / cc['max_loss'] if cc['max_loss'] > 0 else 0
    })

    # Protective put
    pp = ProtectivePutStrategy.analyze(
        spot, put_strike, time_to_expiry, volatility, risk_free_rate
    )
    strategies.append({
        'strategy': 'Protective Put',
        'cost': pp['initial_cost'],
        'max_profit': float('inf'),
        'max_loss': pp['max_loss'],
        'breakeven': pp['breakeven'],
        'return_on_risk': 0
    })

    # Collar
    collar = CollarStrategy.analyze(
        spot, put_strike, call_strike, time_to_expiry, volatility, risk_free_rate
    )
    strategies.append({
        'strategy': 'Collar',
        'cost': collar['initial_cost'],
        'max_profit': collar['max_profit'],
        'max_loss': collar['max_loss'],
        'breakeven': spot + collar['net_cost'],
        'return_on_risk': collar['max_profit'] / collar['max_loss'] if collar['max_loss'] > 0 else 0
    })

    return pd.DataFrame(strategies)
