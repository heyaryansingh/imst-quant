"""Transaction Cost Analyzer - Comprehensive trading cost analysis and breakdown.

This module provides deep analysis of transaction costs including:
- Market impact modeling (temporary and permanent)
- Timing cost analysis
- Opportunity cost calculation
- Cost attribution by trade characteristics
- Implementation shortfall analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransactionCostComponents:
    """Components of transaction costs."""

    spread_cost: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    commission: float
    slippage: float
    total_cost: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'spread_cost': self.spread_cost,
            'market_impact': self.market_impact,
            'timing_cost': self.timing_cost,
            'opportunity_cost': self.opportunity_cost,
            'commission': self.commission,
            'slippage': self.slippage,
            'total_cost': self.total_cost
        }


class TransactionCostAnalyzer:
    """Analyzes transaction costs with detailed breakdowns."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        spread_model: str = 'percentage',
        impact_model: str = 'sqrt'
    ):
        """Initialize transaction cost analyzer.

        Args:
            commission_rate: Commission rate as decimal
            spread_model: Bid-ask spread model ('percentage' or 'fixed')
            impact_model: Market impact model ('sqrt', 'linear', 'almgren_chriss')
        """
        self.commission_rate = commission_rate
        self.spread_model = spread_model
        self.impact_model = impact_model

    def analyze_trade(
        self,
        decision_price: float,
        execution_price: float,
        volume: float,
        adv: float,
        spread_bps: float,
        side: str = 'buy'
    ) -> TransactionCostComponents:
        """Analyze costs for a single trade.

        Args:
            decision_price: Price when decision was made
            execution_price: Actual execution price
            volume: Trade volume
            adv: Average daily volume
            spread_bps: Bid-ask spread in basis points
            side: 'buy' or 'sell'

        Returns:
            TransactionCostComponents with detailed breakdown
        """
        notional = volume * execution_price

        # Commission cost
        commission = notional * self.commission_rate

        # Spread cost (pay half-spread on average)
        spread_cost = notional * (spread_bps / 10000) * 0.5

        # Market impact
        participation_rate = volume / adv
        market_impact = self._calculate_market_impact(
            notional, participation_rate, execution_price
        )

        # Timing cost (delay cost)
        timing_cost = abs(execution_price - decision_price) * volume

        # Slippage (difference from mid-price)
        # Assuming execution_price includes slippage
        slippage = abs(execution_price - decision_price) * volume - timing_cost

        # Opportunity cost (for partial fills)
        # Simplified - assumes full fill for now
        opportunity_cost = 0.0

        total_cost = (
            commission + spread_cost + market_impact +
            timing_cost + slippage + opportunity_cost
        )

        return TransactionCostComponents(
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            commission=commission,
            slippage=slippage,
            total_cost=total_cost
        )

    def _calculate_market_impact(
        self,
        notional: float,
        participation_rate: float,
        price: float
    ) -> float:
        """Calculate market impact based on model.

        Args:
            notional: Trade notional value
            participation_rate: Volume / ADV
            price: Execution price

        Returns:
            Market impact cost
        """
        if self.impact_model == 'sqrt':
            # Square-root model (Almgren-Chriss simplified)
            impact_factor = 0.01  # 1% for 100% participation
            impact = notional * impact_factor * np.sqrt(participation_rate)
        elif self.impact_model == 'linear':
            # Linear model
            impact_factor = 0.02  # 2% for 100% participation
            impact = notional * impact_factor * participation_rate
        elif self.impact_model == 'almgren_chriss':
            # Full Almgren-Chriss model
            sigma = 0.02  # Daily volatility
            tau = 1.0 / 252  # One day in years
            eta = 0.1  # Temporary impact parameter
            gamma = 0.05  # Permanent impact parameter

            # Temporary impact
            temp_impact = eta * sigma * np.sqrt(participation_rate / tau)
            # Permanent impact
            perm_impact = gamma * participation_rate

            impact = notional * (temp_impact + perm_impact)
        else:
            impact = 0.0

        return impact

    def analyze_portfolio_trades(
        self,
        trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze transaction costs for a portfolio of trades.

        Args:
            trades_df: DataFrame with columns:
                - decision_price: Price at decision time
                - execution_price: Actual execution price
                - volume: Trade volume
                - adv: Average daily volume
                - spread_bps: Bid-ask spread in bps
                - side: 'buy' or 'sell'
                - symbol: Asset symbol

        Returns:
            DataFrame with cost analysis per trade
        """
        results = []

        for idx, trade in trades_df.iterrows():
            costs = self.analyze_trade(
                decision_price=trade['decision_price'],
                execution_price=trade['execution_price'],
                volume=trade['volume'],
                adv=trade['adv'],
                spread_bps=trade['spread_bps'],
                side=trade.get('side', 'buy')
            )

            result = {
                'symbol': trade.get('symbol', f'trade_{idx}'),
                'notional': trade['volume'] * trade['execution_price'],
                **costs.to_dict(),
                'total_bps': (costs.total_cost / (trade['volume'] * trade['execution_price'])) * 10000
            }
            results.append(result)

        return pd.DataFrame(results)

    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        execution_prices: List[float],
        volumes: List[float],
        benchmark_price: float
    ) -> Dict[str, float]:
        """Calculate implementation shortfall analysis.

        Args:
            decision_price: Price when decision was made
            execution_prices: List of execution prices for child orders
            volumes: List of volumes for child orders
            benchmark_price: Benchmark price (e.g., arrival price)

        Returns:
            Dictionary with IS components:
                - delay_cost: Cost of waiting to trade
                - execution_cost: Cost during execution
                - opportunity_cost: Cost of unfilled orders
                - total_is: Total implementation shortfall
        """
        total_volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(execution_prices, volumes)) / total_volume

        # Delay cost (decision to start of execution)
        delay_cost = (benchmark_price - decision_price) * total_volume

        # Execution cost (during execution)
        execution_cost = (vwap - benchmark_price) * total_volume

        # Opportunity cost (unfilled orders)
        # For simplicity, assume all orders filled
        opportunity_cost = 0.0

        total_is = delay_cost + execution_cost + opportunity_cost

        return {
            'delay_cost': delay_cost,
            'execution_cost': execution_cost,
            'opportunity_cost': opportunity_cost,
            'total_is': total_is,
            'is_bps': (total_is / (decision_price * total_volume)) * 10000
        }

    def attribute_costs(
        self,
        trades_df: pd.DataFrame,
        by: str = 'symbol'
    ) -> pd.DataFrame:
        """Attribute transaction costs by various dimensions.

        Args:
            trades_df: DataFrame with trade and cost data
            by: Dimension to attribute by ('symbol', 'strategy', 'time_period')

        Returns:
            DataFrame with cost attribution
        """
        if by not in trades_df.columns:
            raise ValueError(f"Column '{by}' not found in trades_df")

        cost_cols = [
            'spread_cost', 'market_impact', 'timing_cost',
            'opportunity_cost', 'commission', 'slippage', 'total_cost'
        ]

        # Group and sum costs
        attribution = trades_df.groupby(by)[cost_cols + ['notional']].sum()

        # Calculate cost as % of notional
        for col in cost_cols:
            attribution[f'{col}_pct'] = (attribution[col] / attribution['notional']) * 100

        return attribution

    def estimate_optimal_execution_horizon(
        self,
        target_volume: float,
        adv: float,
        volatility: float,
        urgency: float = 0.5
    ) -> Dict[str, float]:
        """Estimate optimal execution horizon using Almgren-Chriss.

        Args:
            target_volume: Total volume to execute
            adv: Average daily volume
            volatility: Daily volatility
            urgency: Trading urgency (0=patient, 1=urgent)

        Returns:
            Dictionary with:
                - optimal_horizon_days: Optimal execution time
                - expected_cost_bps: Expected total cost in bps
                - recommended_rate: Recommended participation rate
        """
        # Simplified Almgren-Chriss
        eta = 0.1  # Temporary impact
        gamma = 0.05  # Permanent impact
        lambda_risk = 2e-6  # Risk aversion

        # Adjust for urgency
        lambda_risk = lambda_risk / (urgency + 0.1)

        # Optimal execution time (in days)
        sigma_annual = volatility * np.sqrt(252)
        optimal_horizon = np.sqrt(
            (eta / (gamma * lambda_risk * sigma_annual**2)) *
            (target_volume / adv)
        )

        # Expected cost
        participation_rate = target_volume / (adv * optimal_horizon)
        temp_cost = eta * np.sqrt(participation_rate) * volatility
        perm_cost = gamma * participation_rate
        expected_cost_bps = (temp_cost + perm_cost) * 10000

        return {
            'optimal_horizon_days': optimal_horizon,
            'expected_cost_bps': expected_cost_bps,
            'recommended_participation_rate': participation_rate,
            'recommended_daily_volume': adv * participation_rate
        }

    def compare_execution_strategies(
        self,
        target_volume: float,
        adv: float,
        volatility: float,
        current_price: float
    ) -> pd.DataFrame:
        """Compare different execution strategies.

        Args:
            target_volume: Total volume to execute
            adv: Average daily volume
            volatility: Daily volatility
            current_price: Current market price

        Returns:
            DataFrame comparing strategies
        """
        strategies = {
            'Aggressive (0.5 days)': 0.5,
            'Moderate (2 days)': 2.0,
            'Patient (5 days)': 5.0,
            'Very Patient (10 days)': 10.0,
            'TWAP (1 day)': 1.0
        }

        results = []
        for name, horizon_days in strategies.items():
            participation = target_volume / (adv * horizon_days)

            # Estimate costs
            if name.startswith('TWAP'):
                # Time-weighted average price
                impact = 0.01 * np.sqrt(participation)
            else:
                # Volume-weighted
                impact = 0.01 * np.sqrt(participation) + 0.005 * participation

            timing_risk = volatility * np.sqrt(horizon_days) * current_price * target_volume

            results.append({
                'strategy': name,
                'horizon_days': horizon_days,
                'participation_rate': participation,
                'expected_impact_bps': impact * 10000,
                'timing_risk_usd': timing_risk,
                'total_cost_bps': impact * 10000 + (timing_risk / (current_price * target_volume)) * 10000
            })

        return pd.DataFrame(results).sort_values('total_cost_bps')


def analyze_historical_transaction_costs(
    trades_df: pd.DataFrame,
    groupby: str = 'date'
) -> pd.DataFrame:
    """Analyze historical transaction costs over time.

    Args:
        trades_df: DataFrame with historical trades and costs
        groupby: Time period to group by ('date', 'week', 'month')

    Returns:
        DataFrame with time series of costs
    """
    if 'timestamp' not in trades_df.columns:
        raise ValueError("trades_df must have 'timestamp' column")

    trades_df = trades_df.copy()

    # Create time grouping
    if groupby == 'date':
        trades_df['period'] = pd.to_datetime(trades_df['timestamp']).dt.date
    elif groupby == 'week':
        trades_df['period'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('W')
    elif groupby == 'month':
        trades_df['period'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')

    # Aggregate costs
    cost_cols = ['spread_cost', 'market_impact', 'timing_cost', 'commission', 'total_cost']
    agg_dict = {col: 'sum' for col in cost_cols if col in trades_df.columns}
    agg_dict['notional'] = 'sum'
    agg_dict['volume'] = 'sum'

    historical = trades_df.groupby('period').agg(agg_dict).reset_index()

    # Calculate cost as % of notional
    for col in cost_cols:
        if col in historical.columns:
            historical[f'{col}_pct'] = (historical[col] / historical['notional']) * 100

    return historical


if __name__ == '__main__':
    # Example usage
    analyzer = TransactionCostAnalyzer(
        commission_rate=0.001,
        impact_model='sqrt'
    )

    # Analyze single trade
    costs = analyzer.analyze_trade(
        decision_price=100.0,
        execution_price=100.15,
        volume=10000,
        adv=1000000,
        spread_bps=5.0,
        side='buy'
    )
    print("Single Trade Costs:")
    print(costs.to_dict())

    # Optimal execution horizon
    optimal = analyzer.estimate_optimal_execution_horizon(
        target_volume=50000,
        adv=1000000,
        volatility=0.02,
        urgency=0.5
    )
    print("\nOptimal Execution:")
    print(optimal)
