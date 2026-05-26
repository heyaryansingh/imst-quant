"""Dynamic portfolio hedging using correlation analysis and factor models.

This module implements advanced hedging strategies including:
- Correlation-based hedging
- Beta-neutral hedging
- Factor-based risk decomposition
- Dynamic hedge ratio adjustment
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger()


@dataclass
class HedgeConfig:
    """Configuration for dynamic hedging."""

    # Correlation parameters
    correlation_lookback: int = 60  # Days for correlation calculation
    min_correlation: float = 0.5  # Minimum correlation for hedge candidates
    correlation_threshold: float = -0.7  # Threshold for negative correlation

    # Beta parameters
    target_beta: float = 0.0  # Target beta (0 = market neutral)
    beta_tolerance: float = 0.1  # Tolerance around target beta
    beta_lookback: int = 90  # Days for beta calculation

    # Hedge ratio parameters
    hedge_ratio_method: str = "minimum_variance"  # or "beta_adjusted", "correlation"
    rebalance_threshold: float = 0.15  # Rebalance when hedge ratio deviates by 15%

    # Factor model parameters
    use_factor_model: bool = True
    factors: List[str] = None  # Factor names (e.g., ["SPY", "TLT", "GLD"])

    def __post_init__(self):
        """Initialize default factors if None."""
        if self.factors is None:
            self.factors = ["SPY"]  # Default to S&P 500


class DynamicHedger:
    """Dynamic portfolio hedging with correlation analysis and factor models.

    This class provides sophisticated hedging strategies that dynamically adjust
    hedge ratios based on changing market conditions, correlations, and factor exposures.

    Examples:
        >>> config = HedgeConfig(target_beta=0.0, min_correlation=0.6)
        >>> hedger = DynamicHedger(config)
        >>>
        >>> # Find hedge candidates for a position
        >>> position_returns = pd.Series([...])  # Historical returns
        >>> market_returns = pd.Series([...])    # Market returns
        >>> hedge_ratio = hedger.calculate_hedge_ratio(position_returns, market_returns)
        >>>
        >>> # Construct hedged portfolio
        >>> portfolio_returns = {"AAPL": returns1, "MSFT": returns2}
        >>> hedges = hedger.construct_hedges(portfolio_returns, market_data)
    """

    def __init__(self, config: Optional[HedgeConfig] = None):
        """Initialize dynamic hedger.

        Args:
            config: Configuration for hedging strategies
        """
        self.config = config or HedgeConfig()
        logger.info(
            "dynamic_hedger_initialized",
            target_beta=self.config.target_beta,
            hedge_ratio_method=self.config.hedge_ratio_method,
        )

    def calculate_hedge_ratio(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series,
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate optimal hedge ratio for a position.

        Args:
            position_returns: Returns of the position to hedge
            hedge_returns: Returns of the hedge instrument
            method: Hedge ratio calculation method (overrides config)

        Returns:
            Dict containing hedge_ratio and related statistics
        """
        method = method or self.config.hedge_ratio_method

        # Align series
        aligned = pd.concat([position_returns, hedge_returns], axis=1).dropna()
        if len(aligned) < 20:
            logger.warning("insufficient_data_for_hedge", n_samples=len(aligned))
            return {"hedge_ratio": 0.0, "method": method, "error": "insufficient_data"}

        pos_ret = aligned.iloc[:, 0]
        hdg_ret = aligned.iloc[:, 1]

        if method == "minimum_variance":
            hedge_ratio = self._minimum_variance_hedge(pos_ret, hdg_ret)
        elif method == "beta_adjusted":
            hedge_ratio = self._beta_adjusted_hedge(pos_ret, hdg_ret)
        elif method == "correlation":
            hedge_ratio = self._correlation_hedge(pos_ret, hdg_ret)
        else:
            logger.error("unknown_hedge_method", method=method)
            return {"hedge_ratio": 0.0, "method": method, "error": "unknown_method"}

        # Calculate hedge effectiveness
        effectiveness = self._calculate_hedge_effectiveness(pos_ret, hdg_ret, hedge_ratio)

        result = {
            "hedge_ratio": hedge_ratio,
            "method": method,
            "effectiveness": effectiveness,
            "correlation": pos_ret.corr(hdg_ret),
            "beta": self._calculate_beta(pos_ret, hdg_ret),
        }

        logger.info("hedge_ratio_calculated", **result)
        return result

    def _minimum_variance_hedge(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series
    ) -> float:
        """Calculate minimum variance hedge ratio.

        This minimizes the variance of the hedged portfolio:
        h* = Cov(R_p, R_h) / Var(R_h)

        Args:
            position_returns: Position returns
            hedge_returns: Hedge instrument returns

        Returns:
            Optimal hedge ratio
        """
        covariance = position_returns.cov(hedge_returns)
        hedge_variance = hedge_returns.var()

        if hedge_variance == 0:
            return 0.0

        hedge_ratio = covariance / hedge_variance
        return hedge_ratio

    def _beta_adjusted_hedge(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series
    ) -> float:
        """Calculate beta-adjusted hedge ratio.

        This uses the beta of the position relative to the hedge instrument
        to achieve beta-neutral exposure.

        Args:
            position_returns: Position returns
            hedge_returns: Hedge instrument returns

        Returns:
            Beta-adjusted hedge ratio
        """
        beta = self._calculate_beta(position_returns, hedge_returns)

        # Adjust to achieve target beta
        hedge_ratio = beta - self.config.target_beta

        return hedge_ratio

    def _correlation_hedge(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series
    ) -> float:
        """Calculate correlation-based hedge ratio.

        This uses correlation and relative volatilities:
        h* = ρ * (σ_p / σ_h)

        Args:
            position_returns: Position returns
            hedge_returns: Hedge instrument returns

        Returns:
            Correlation-based hedge ratio
        """
        correlation = position_returns.corr(hedge_returns)
        pos_vol = position_returns.std()
        hdg_vol = hedge_returns.std()

        if hdg_vol == 0:
            return 0.0

        hedge_ratio = correlation * (pos_vol / hdg_vol)
        return hedge_ratio

    def _calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate beta of asset relative to market.

        Args:
            asset_returns: Asset returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance
        return beta

    def _calculate_hedge_effectiveness(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series,
        hedge_ratio: float
    ) -> float:
        """Calculate hedge effectiveness (reduction in variance).

        Args:
            position_returns: Position returns
            hedge_returns: Hedge instrument returns
            hedge_ratio: Applied hedge ratio

        Returns:
            Hedge effectiveness (0 = no reduction, 1 = perfect hedge)
        """
        # Variance of unhedged position
        unhedged_var = position_returns.var()

        # Variance of hedged position
        hedged_returns = position_returns - hedge_ratio * hedge_returns
        hedged_var = hedged_returns.var()

        if unhedged_var == 0:
            return 0.0

        # Variance reduction ratio
        effectiveness = 1.0 - (hedged_var / unhedged_var)
        return max(0.0, effectiveness)

    def find_hedge_candidates(
        self,
        position_returns: pd.Series,
        candidate_returns: Dict[str, pd.Series],
        top_n: int = 5
    ) -> pd.DataFrame:
        """Find best hedge candidates based on correlation and effectiveness.

        Args:
            position_returns: Returns of position to hedge
            candidate_returns: Dict of candidate hedge instrument returns
            top_n: Number of top candidates to return

        Returns:
            DataFrame with ranked hedge candidates
        """
        candidates = []

        for name, hedge_returns in candidate_returns.items():
            # Calculate hedge ratio and effectiveness
            hedge_info = self.calculate_hedge_ratio(position_returns, hedge_returns)

            if "error" in hedge_info:
                continue

            # Add candidate info
            candidates.append({
                "instrument": name,
                "hedge_ratio": hedge_info["hedge_ratio"],
                "effectiveness": hedge_info["effectiveness"],
                "correlation": hedge_info["correlation"],
                "beta": hedge_info["beta"],
            })

        if not candidates:
            logger.warning("no_valid_hedge_candidates")
            return pd.DataFrame()

        df = pd.DataFrame(candidates)

        # Rank by effectiveness
        df = df.sort_values("effectiveness", ascending=False)
        df = df.head(top_n)

        logger.info("hedge_candidates_found", n_candidates=len(df))
        return df

    def construct_portfolio_hedge(
        self,
        portfolio_returns: Dict[str, pd.Series],
        hedge_candidates: Dict[str, pd.Series],
        portfolio_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Construct hedges for an entire portfolio.

        Args:
            portfolio_returns: Dict of portfolio position returns
            hedge_candidates: Dict of hedge instrument returns
            portfolio_weights: Optional portfolio weights (equal weight if None)

        Returns:
            Dict mapping hedge instruments to hedge sizes
        """
        if portfolio_weights is None:
            # Equal weight
            n = len(portfolio_returns)
            portfolio_weights = {k: 1.0 / n for k in portfolio_returns.keys()}

        # Calculate portfolio returns
        portfolio_df = pd.DataFrame(portfolio_returns)
        weights_series = pd.Series(portfolio_weights)
        portfolio_ret = (portfolio_df * weights_series).sum(axis=1)

        # Find best hedges for portfolio
        hedge_df = self.find_hedge_candidates(
            portfolio_ret,
            hedge_candidates,
            top_n=min(3, len(hedge_candidates))
        )

        if hedge_df.empty:
            return {}

        # Build hedge dict
        hedges = {}
        for _, row in hedge_df.iterrows():
            hedges[row["instrument"]] = row["hedge_ratio"]

        logger.info("portfolio_hedge_constructed", hedges=hedges)
        return hedges

    def calculate_factor_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Calculate portfolio exposures to risk factors.

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Dict of factor returns

        Returns:
            Dict mapping factor names to exposure coefficients
        """
        # Prepare factor matrix
        factor_df = pd.DataFrame(factor_returns)

        # Align with portfolio returns
        aligned = pd.concat([portfolio_returns, factor_df], axis=1).dropna()
        if len(aligned) < 30:
            logger.warning("insufficient_data_for_factor_analysis", n_samples=len(aligned))
            return {}

        y = aligned.iloc[:, 0].values  # Portfolio returns
        X = aligned.iloc[:, 1:].values  # Factor returns

        # Multiple linear regression: R_p = α + Σ(β_i * F_i) + ε
        # Using normal equations: β = (X'X)^-1 X'y
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]

            exposures = {}
            for i, factor_name in enumerate(factor_df.columns):
                exposures[factor_name] = betas[i]

            logger.info("factor_exposures_calculated", exposures=exposures)
            return exposures

        except np.linalg.LinAlgError:
            logger.error("factor_regression_failed")
            return {}

    def neutralize_factor_exposure(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
        target_exposures: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate hedge ratios to neutralize factor exposures.

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Dict of factor returns
            target_exposures: Target exposures (None = zero exposure to all factors)

        Returns:
            Dict mapping factors to hedge ratios
        """
        if target_exposures is None:
            target_exposures = {k: 0.0 for k in factor_returns.keys()}

        # Calculate current exposures
        current_exposures = self.calculate_factor_exposures(
            portfolio_returns,
            factor_returns
        )

        if not current_exposures:
            return {}

        # Calculate hedge ratios needed to achieve target exposures
        hedge_ratios = {}
        for factor, current_beta in current_exposures.items():
            target_beta = target_exposures.get(factor, 0.0)
            hedge_ratio = current_beta - target_beta
            hedge_ratios[factor] = hedge_ratio

        logger.info(
            "factor_hedge_ratios_calculated",
            current_exposures=current_exposures,
            target_exposures=target_exposures,
            hedge_ratios=hedge_ratios,
        )

        return hedge_ratios

    def optimize_dynamic_hedge(
        self,
        position_returns: pd.Series,
        hedge_returns: pd.Series,
        transaction_cost: float = 0.001,
        current_hedge_ratio: float = 0.0
    ) -> Dict[str, float]:
        """Optimize hedge ratio considering transaction costs and rebalancing.

        Args:
            position_returns: Position returns
            hedge_returns: Hedge instrument returns
            transaction_cost: Transaction cost (as fraction)
            current_hedge_ratio: Current hedge ratio

        Returns:
            Dict with optimal hedge ratio and rebalancing decision
        """
        # Calculate optimal hedge ratio without transaction costs
        optimal = self.calculate_hedge_ratio(position_returns, hedge_returns)
        optimal_ratio = optimal["hedge_ratio"]

        # Calculate deviation from current ratio
        deviation = abs(optimal_ratio - current_hedge_ratio)
        deviation_pct = deviation / max(abs(current_hedge_ratio), 1e-6)

        # Determine if rebalancing is worthwhile
        should_rebalance = deviation_pct > self.config.rebalance_threshold

        if should_rebalance:
            # Calculate expected benefit of rebalancing
            current_effectiveness = self._calculate_hedge_effectiveness(
                position_returns,
                hedge_returns,
                current_hedge_ratio
            )

            new_effectiveness = optimal["effectiveness"]

            effectiveness_gain = new_effectiveness - current_effectiveness

            # Cost of rebalancing
            rebalance_cost = deviation * transaction_cost

            # Net benefit
            net_benefit = effectiveness_gain - rebalance_cost

            if net_benefit < 0:
                # Not worth rebalancing
                should_rebalance = False
                logger.info(
                    "rebalance_not_worthwhile",
                    effectiveness_gain=effectiveness_gain,
                    rebalance_cost=rebalance_cost,
                    net_benefit=net_benefit,
                )

        result = {
            "optimal_ratio": optimal_ratio,
            "current_ratio": current_hedge_ratio,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
            "should_rebalance": should_rebalance,
            "effectiveness": optimal["effectiveness"],
        }

        if should_rebalance:
            logger.info("rebalance_recommended", **result)
        else:
            logger.debug("no_rebalance_needed", **result)

        return result

    def calculate_tail_hedge(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.Series,
        tail_quantile: float = 0.05
    ) -> Dict[str, float]:
        """Calculate hedge ratio optimized for tail risk protection.

        Args:
            portfolio_returns: Portfolio returns
            hedge_returns: Hedge instrument returns
            tail_quantile: Quantile for tail risk (e.g., 0.05 for 5% worst cases)

        Returns:
            Dict with tail hedge ratio and tail risk metrics
        """
        # Identify tail events (worst returns)
        threshold = portfolio_returns.quantile(tail_quantile)
        tail_events = portfolio_returns <= threshold

        # Calculate conditional correlation during tail events
        tail_portfolio = portfolio_returns[tail_events]
        tail_hedge = hedge_returns[tail_events]

        if len(tail_portfolio) < 5:
            logger.warning("insufficient_tail_events", n_events=len(tail_portfolio))
            return {"hedge_ratio": 0.0, "error": "insufficient_tail_events"}

        # Calculate hedge ratio for tail events
        tail_hedge_ratio = self._minimum_variance_hedge(tail_portfolio, tail_hedge)

        # Calculate tail risk reduction
        tail_var_unhedged = tail_portfolio.var()
        hedged_tail = tail_portfolio - tail_hedge_ratio * tail_hedge
        tail_var_hedged = hedged_tail.var()

        tail_risk_reduction = 1.0 - (tail_var_hedged / tail_var_unhedged) if tail_var_unhedged > 0 else 0.0

        result = {
            "hedge_ratio": tail_hedge_ratio,
            "tail_risk_reduction": tail_risk_reduction,
            "tail_correlation": tail_portfolio.corr(tail_hedge),
            "n_tail_events": len(tail_portfolio),
            "threshold": threshold,
        }

        logger.info("tail_hedge_calculated", **result)
        return result
