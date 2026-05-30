"""Trade Execution Quality Analyzer - Comprehensive execution performance analysis.

Analyzes trade execution quality including slippage, fill rates, timing,
market impact, and execution cost breakdown for performance optimization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum


class ExecutionVenue(Enum):
    """Execution venues/brokers."""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ROBINHOOD = "robinhood"
    TD_AMERITRADE = "td_ameritrade"
    PAPER = "paper_trading"


@dataclass
class ExecutionMetrics:
    """Comprehensive execution quality metrics."""
    avg_slippage_bps: float
    median_slippage_bps: float
    slippage_std_bps: float
    fill_rate: float
    partial_fill_rate: float
    avg_fill_time_seconds: float
    median_fill_time_seconds: float
    market_impact_bps: float
    effective_spread_bps: float
    realized_spread_bps: float
    price_improvement_bps: float
    adverse_selection_bps: float
    implementation_shortfall_bps: float
    total_execution_cost_bps: float


@dataclass
class TradeExecution:
    """Individual trade execution record."""
    symbol: str
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: int
    limit_price: Optional[float]
    fill_price: float
    benchmark_price: float  # VWAP, arrival price, or mid
    venue: ExecutionVenue
    fill_time_seconds: float
    partial_fill: bool = False


class ExecutionQualityAnalyzer:
    """Analyze and optimize trade execution quality."""

    def __init__(self, executions: List[TradeExecution]):
        """Initialize analyzer with execution data.

        Args:
            executions: List of TradeExecution records
        """
        self.executions = executions

    def calculate_slippage(self, execution: TradeExecution) -> float:
        """Calculate slippage in basis points.

        Args:
            execution: TradeExecution record

        Returns:
            Slippage in bps (positive = worse execution)
        """
        if execution.side == 'buy':
            slippage = (execution.fill_price - execution.benchmark_price) / execution.benchmark_price
        else:  # sell
            slippage = (execution.benchmark_price - execution.fill_price) / execution.benchmark_price

        return slippage * 10000  # Convert to bps

    def calculate_market_impact(
        self,
        execution: TradeExecution,
        post_trade_price: float,
        horizon_minutes: int = 5
    ) -> float:
        """Calculate market impact after trade.

        Args:
            execution: TradeExecution record
            post_trade_price: Price N minutes after execution
            horizon_minutes: Minutes after trade to measure

        Returns:
            Market impact in bps
        """
        if execution.side == 'buy':
            impact = (post_trade_price - execution.fill_price) / execution.fill_price
        else:
            impact = (execution.fill_price - post_trade_price) / execution.fill_price

        return abs(impact) * 10000

    def calculate_effective_spread(self, execution: TradeExecution) -> float:
        """Calculate effective spread (difference from mid to execution).

        Args:
            execution: TradeExecution record

        Returns:
            Effective spread in bps
        """
        # Using benchmark as mid price
        spread = abs(execution.fill_price - execution.benchmark_price) / execution.benchmark_price
        return spread * 10000 * 2  # Multiply by 2 for half-spread convention

    def calculate_implementation_shortfall(
        self,
        execution: TradeExecution,
        decision_price: float
    ) -> float:
        """Calculate implementation shortfall.

        Args:
            execution: TradeExecution record
            decision_price: Price when decision was made

        Returns:
            Implementation shortfall in bps
        """
        if execution.side == 'buy':
            shortfall = (execution.fill_price - decision_price) / decision_price
        else:
            shortfall = (decision_price - execution.fill_price) / decision_price

        return shortfall * 10000

    def analyze_fill_rates(self) -> Dict[str, float]:
        """Analyze order fill rates.

        Returns:
            Dict with fill rate metrics
        """
        total_orders = len(self.executions)
        if total_orders == 0:
            return {
                'fill_rate': 0.0,
                'partial_fill_rate': 0.0,
                'complete_fill_rate': 0.0
            }

        partial_fills = sum(1 for e in self.executions if e.partial_fill)
        complete_fills = total_orders - partial_fills

        return {
            'fill_rate': 1.0,  # Assuming all orders in list were filled
            'partial_fill_rate': partial_fills / total_orders,
            'complete_fill_rate': complete_fills / total_orders
        }

    def analyze_fill_times(self) -> Dict[str, float]:
        """Analyze order fill times.

        Returns:
            Dict with fill time statistics
        """
        fill_times = [e.fill_time_seconds for e in self.executions]

        if not fill_times:
            return {
                'avg_fill_time': 0.0,
                'median_fill_time': 0.0,
                'p95_fill_time': 0.0,
                'max_fill_time': 0.0
            }

        return {
            'avg_fill_time': np.mean(fill_times),
            'median_fill_time': np.median(fill_times),
            'p95_fill_time': np.percentile(fill_times, 95),
            'max_fill_time': np.max(fill_times)
        }

    def analyze_slippage(self) -> Dict[str, float]:
        """Analyze slippage statistics.

        Returns:
            Dict with slippage metrics
        """
        slippages = [self.calculate_slippage(e) for e in self.executions]

        if not slippages:
            return {
                'avg_slippage': 0.0,
                'median_slippage': 0.0,
                'std_slippage': 0.0,
                'p95_slippage': 0.0,
                'worst_slippage': 0.0
            }

        return {
            'avg_slippage': np.mean(slippages),
            'median_slippage': np.median(slippages),
            'std_slippage': np.std(slippages),
            'p95_slippage': np.percentile(slippages, 95),
            'worst_slippage': np.max(slippages)
        }

    def analyze_by_order_type(self) -> pd.DataFrame:
        """Analyze execution quality by order type.

        Returns:
            DataFrame with metrics per order type
        """
        results = []

        order_types = set(e.order_type for e in self.executions)

        for order_type in order_types:
            type_execs = [e for e in self.executions if e.order_type == order_type]

            if not type_execs:
                continue

            slippages = [self.calculate_slippage(e) for e in type_execs]
            fill_times = [e.fill_time_seconds for e in type_execs]

            results.append({
                'order_type': order_type,
                'count': len(type_execs),
                'avg_slippage_bps': np.mean(slippages),
                'median_slippage_bps': np.median(slippages),
                'avg_fill_time_seconds': np.mean(fill_times),
                'partial_fill_rate': sum(1 for e in type_execs if e.partial_fill) / len(type_execs)
            })

        return pd.DataFrame(results)

    def analyze_by_venue(self) -> pd.DataFrame:
        """Analyze execution quality by venue.

        Returns:
            DataFrame with metrics per venue
        """
        results = []

        venues = set(e.venue for e in self.executions)

        for venue in venues:
            venue_execs = [e for e in self.executions if e.venue == venue]

            if not venue_execs:
                continue

            slippages = [self.calculate_slippage(e) for e in venue_execs]
            fill_times = [e.fill_time_seconds for e in venue_execs]

            results.append({
                'venue': venue.value,
                'count': len(venue_execs),
                'avg_slippage_bps': np.mean(slippages),
                'median_slippage_bps': np.median(slippages),
                'std_slippage_bps': np.std(slippages),
                'avg_fill_time_seconds': np.mean(fill_times),
                'partial_fill_rate': sum(1 for e in venue_execs if e.partial_fill) / len(venue_execs)
            })

        return pd.DataFrame(results)

    def analyze_by_side(self) -> Dict[str, Dict[str, float]]:
        """Analyze execution quality by trade side (buy/sell).

        Returns:
            Dict with metrics per side
        """
        results = {}

        for side in ['buy', 'sell']:
            side_execs = [e for e in self.executions if e.side == side]

            if not side_execs:
                results[side] = {
                    'count': 0,
                    'avg_slippage_bps': 0.0,
                    'median_slippage_bps': 0.0
                }
                continue

            slippages = [self.calculate_slippage(e) for e in side_execs]

            results[side] = {
                'count': len(side_execs),
                'avg_slippage_bps': np.mean(slippages),
                'median_slippage_bps': np.median(slippages),
                'std_slippage_bps': np.std(slippages)
            }

        return results

    def generate_quality_metrics(self) -> ExecutionMetrics:
        """Generate comprehensive execution quality metrics.

        Returns:
            ExecutionMetrics object
        """
        slippage_stats = self.analyze_slippage()
        fill_stats = self.analyze_fill_rates()
        time_stats = self.analyze_fill_times()

        # Calculate aggregated metrics
        all_slippages = [self.calculate_slippage(e) for e in self.executions]
        avg_effective_spread = np.mean([
            self.calculate_effective_spread(e) for e in self.executions
        ]) if self.executions else 0.0

        # Estimate total execution cost
        total_cost = slippage_stats['avg_slippage'] + avg_effective_spread / 2

        return ExecutionMetrics(
            avg_slippage_bps=slippage_stats['avg_slippage'],
            median_slippage_bps=slippage_stats['median_slippage'],
            slippage_std_bps=slippage_stats['std_slippage'],
            fill_rate=fill_stats['fill_rate'],
            partial_fill_rate=fill_stats['partial_fill_rate'],
            avg_fill_time_seconds=time_stats['avg_fill_time'],
            median_fill_time_seconds=time_stats['median_fill_time'],
            market_impact_bps=0.0,  # Would need post-trade data
            effective_spread_bps=avg_effective_spread,
            realized_spread_bps=0.0,  # Would need post-trade data
            price_improvement_bps=max(0, -slippage_stats['avg_slippage']),
            adverse_selection_bps=max(0, slippage_stats['avg_slippage']),
            implementation_shortfall_bps=0.0,  # Would need decision prices
            total_execution_cost_bps=total_cost
        )

    def detect_execution_issues(self) -> List[str]:
        """Detect potential execution quality issues.

        Returns:
            List of issue descriptions
        """
        issues = []
        slippage_stats = self.analyze_slippage()
        time_stats = self.analyze_fill_times()

        # Check slippage
        if slippage_stats['avg_slippage'] > 10:
            issues.append(f"HIGH SLIPPAGE: Average {slippage_stats['avg_slippage']:.1f} bps")

        if slippage_stats['std_slippage'] > 15:
            issues.append(f"INCONSISTENT EXECUTION: Slippage std {slippage_stats['std_slippage']:.1f} bps")

        # Check fill times
        if time_stats['avg_fill_time'] > 5.0:
            issues.append(f"SLOW FILLS: Average {time_stats['avg_fill_time']:.1f} seconds")

        if time_stats['p95_fill_time'] > 30.0:
            issues.append(f"EXTREME DELAYS: 95th percentile {time_stats['p95_fill_time']:.1f} seconds")

        # Check partial fills
        partial_rate = sum(1 for e in self.executions if e.partial_fill) / len(self.executions) if self.executions else 0
        if partial_rate > 0.1:
            issues.append(f"HIGH PARTIAL FILLS: {partial_rate:.1%} of orders")

        return issues

    def create_execution_report(self) -> pd.DataFrame:
        """Create detailed execution quality report.

        Returns:
            DataFrame with comprehensive metrics
        """
        metrics = self.generate_quality_metrics()

        report_data = {
            'Metric': [
                'Avg Slippage', 'Median Slippage', 'Slippage Std Dev',
                'Fill Rate', 'Partial Fill Rate',
                'Avg Fill Time', 'Median Fill Time',
                'Effective Spread', 'Price Improvement',
                'Total Execution Cost'
            ],
            'Value': [
                f"{metrics.avg_slippage_bps:.2f} bps",
                f"{metrics.median_slippage_bps:.2f} bps",
                f"{metrics.slippage_std_bps:.2f} bps",
                f"{metrics.fill_rate:.1%}",
                f"{metrics.partial_fill_rate:.1%}",
                f"{metrics.avg_fill_time_seconds:.2f} sec",
                f"{metrics.median_fill_time_seconds:.2f} sec",
                f"{metrics.effective_spread_bps:.2f} bps",
                f"{metrics.price_improvement_bps:.2f} bps",
                f"{metrics.total_execution_cost_bps:.2f} bps"
            ]
        }

        return pd.DataFrame(report_data)


def compare_venues(
    executions: List[TradeExecution],
    min_sample_size: int = 10
) -> pd.DataFrame:
    """Compare execution quality across venues.

    Args:
        executions: List of executions
        min_sample_size: Minimum executions per venue for comparison

    Returns:
        DataFrame ranking venues by quality
    """
    venue_groups = {}

    for venue in ExecutionVenue:
        venue_execs = [e for e in executions if e.venue == venue]
        if len(venue_execs) >= min_sample_size:
            venue_groups[venue.value] = venue_execs

    results = []

    for venue_name, execs in venue_groups.items():
        analyzer = ExecutionQualityAnalyzer(execs)
        metrics = analyzer.generate_quality_metrics()

        results.append({
            'venue': venue_name,
            'sample_size': len(execs),
            'avg_slippage_bps': metrics.avg_slippage_bps,
            'avg_fill_time': metrics.avg_fill_time_seconds,
            'total_cost_bps': metrics.total_execution_cost_bps,
            'quality_score': -metrics.total_execution_cost_bps  # Higher is better
        })

    df = pd.DataFrame(results)
    df = df.sort_values('quality_score', ascending=False)

    return df
