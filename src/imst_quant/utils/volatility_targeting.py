"""Volatility-targeting position sizing for risk-adjusted portfolio management.

This module implements volatility targeting (vol-targeting) strategies that dynamically
adjust position sizes to maintain constant portfolio volatility exposure.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Union
import structlog

logger = structlog.get_logger()


@dataclass
class VolTargetConfig:
    """Configuration for volatility targeting."""

    target_vol: float = 0.15  # 15% annualized portfolio volatility
    lookback_days: int = 20  # Rolling window for vol calculation
    min_leverage: float = 0.0  # Minimum leverage (0 = no shorting)
    max_leverage: float = 2.0  # Maximum leverage
    vol_floor: float = 0.01  # Minimum volatility to avoid division by zero
    rebalance_threshold: float = 0.10  # Rebalance when exposure deviates by 10%
    clip_outliers: bool = True  # Clip extreme volatility estimates
    outlier_std: float = 3.0  # Standard deviations for outlier detection


class VolatilityTargeter:
    """Dynamically size positions to achieve target portfolio volatility.

    This implementation uses realized volatility and adjusts position sizes
    inversely proportional to volatility: when vol increases, reduce position;
    when vol decreases, increase position.

    Examples:
        >>> config = VolTargetConfig(target_vol=0.12, lookback_days=30)
        >>> targeter = VolatilityTargeter(config)
        >>>
        >>> # Calculate position size for an asset
        >>> returns = pd.Series([0.01, -0.015, 0.02, -0.005, 0.012])
        >>> position_size = targeter.calculate_position_size(returns)
        >>>
        >>> # Rebalance entire portfolio
        >>> portfolio_returns = pd.Series([...])  # Historical returns
        >>> current_exposure = 1.0
        >>> new_exposure = targeter.rebalance_portfolio(
        ...     portfolio_returns,
        ...     current_exposure
        ... )
    """

    def __init__(self, config: Optional[VolTargetConfig] = None):
        """Initialize volatility targeter.

        Args:
            config: Configuration for vol targeting. Uses defaults if None.
        """
        self.config = config or VolTargetConfig()
        logger.info(
            "vol_targeter_initialized",
            target_vol=self.config.target_vol,
            lookback_days=self.config.lookback_days,
        )

    def calculate_realized_vol(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """Calculate realized volatility from return series.

        Args:
            returns: Time series of returns (linear, not log)
            annualize: If True, annualize the volatility (assumes daily returns)

        Returns:
            Realized volatility (standard deviation of returns)
        """
        if len(returns) < 2:
            logger.warning("insufficient_data_for_vol", n_returns=len(returns))
            return self.config.vol_floor

        # Calculate standard deviation
        vol = returns.std()

        # Annualize if requested (sqrt(252) for daily returns)
        if annualize:
            vol = vol * np.sqrt(252)

        # Apply floor to avoid division by zero
        vol = max(vol, self.config.vol_floor)

        # Clip outliers if configured
        if self.config.clip_outliers:
            vol = self._clip_outlier_vol(returns, vol, annualize)

        return vol

    def _clip_outlier_vol(
        self,
        returns: pd.Series,
        vol: float,
        annualize: bool
    ) -> float:
        """Clip extreme volatility estimates using rolling statistics.

        Args:
            returns: Time series of returns
            vol: Current volatility estimate
            annualize: Whether volatility is annualized

        Returns:
            Clipped volatility estimate
        """
        # Calculate rolling volatility statistics
        rolling_vol = returns.rolling(window=self.config.lookback_days).std()
        if annualize:
            rolling_vol = rolling_vol * np.sqrt(252)

        # Calculate mean and std of rolling vol
        mean_vol = rolling_vol.mean()
        std_vol = rolling_vol.std()

        # Clip to mean ± N standard deviations
        lower_bound = max(mean_vol - self.config.outlier_std * std_vol, self.config.vol_floor)
        upper_bound = mean_vol + self.config.outlier_std * std_vol

        clipped_vol = np.clip(vol, lower_bound, upper_bound)

        if clipped_vol != vol:
            logger.warning(
                "vol_clipped",
                original_vol=vol,
                clipped_vol=clipped_vol,
                bounds=(lower_bound, upper_bound),
            )

        return clipped_vol

    def calculate_position_size(
        self,
        returns: pd.Series,
        current_price: Optional[float] = None,
        portfolio_value: Optional[float] = None
    ) -> float:
        """Calculate position size to achieve target volatility.

        Args:
            returns: Historical returns for the asset
            current_price: Current asset price (optional, for share calculation)
            portfolio_value: Total portfolio value (optional, for share calculation)

        Returns:
            Position size as fraction of portfolio (or number of shares if prices provided)
        """
        # Use last N days for volatility calculation
        recent_returns = returns.tail(self.config.lookback_days)
        realized_vol = self.calculate_realized_vol(recent_returns, annualize=True)

        # Calculate position size: target_vol / realized_vol
        # Higher volatility → smaller position, lower volatility → larger position
        position_fraction = self.config.target_vol / realized_vol

        # Apply leverage constraints
        position_fraction = np.clip(
            position_fraction,
            self.config.min_leverage,
            self.config.max_leverage
        )

        logger.info(
            "position_size_calculated",
            realized_vol=realized_vol,
            target_vol=self.config.target_vol,
            position_fraction=position_fraction,
        )

        # Convert to number of shares if prices provided
        if current_price is not None and portfolio_value is not None:
            dollar_allocation = portfolio_value * position_fraction
            shares = dollar_allocation / current_price
            logger.info(
                "shares_calculated",
                dollar_allocation=dollar_allocation,
                current_price=current_price,
                shares=shares,
            )
            return shares

        return position_fraction

    def rebalance_portfolio(
        self,
        portfolio_returns: pd.Series,
        current_exposure: float
    ) -> Dict[str, Union[float, bool]]:
        """Determine if portfolio rebalancing is needed.

        Args:
            portfolio_returns: Historical portfolio returns
            current_exposure: Current portfolio exposure (fraction)

        Returns:
            Dict containing:
                - new_exposure: Recommended new exposure
                - rebalance_needed: Whether rebalancing is recommended
                - current_vol: Current realized volatility
                - deviation: Deviation from target exposure
        """
        # Calculate current realized vol
        recent_returns = portfolio_returns.tail(self.config.lookback_days)
        realized_vol = self.calculate_realized_vol(recent_returns, annualize=True)

        # Calculate target exposure
        target_exposure = self.config.target_vol / realized_vol
        target_exposure = np.clip(
            target_exposure,
            self.config.min_leverage,
            self.config.max_leverage
        )

        # Calculate deviation
        deviation = abs(target_exposure - current_exposure) / target_exposure
        rebalance_needed = deviation > self.config.rebalance_threshold

        result = {
            "new_exposure": target_exposure,
            "rebalance_needed": rebalance_needed,
            "current_vol": realized_vol,
            "target_vol": self.config.target_vol,
            "current_exposure": current_exposure,
            "deviation": deviation,
            "deviation_pct": deviation * 100,
        }

        if rebalance_needed:
            logger.warning(
                "rebalance_recommended",
                **result
            )
        else:
            logger.info(
                "no_rebalance_needed",
                **result
            )

        return result

    def calculate_multi_asset_positions(
        self,
        asset_returns: Dict[str, pd.Series],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate position sizes for multiple assets accounting for correlations.

        Args:
            asset_returns: Dict mapping asset names to return series
            correlation_matrix: Pairwise correlation matrix (optional)

        Returns:
            Dict mapping asset names to position fractions
        """
        positions = {}

        # Calculate individual volatilities
        asset_vols = {}
        for asset, returns in asset_returns.items():
            recent_returns = returns.tail(self.config.lookback_days)
            asset_vols[asset] = self.calculate_realized_vol(recent_returns, annualize=True)

        # If no correlation matrix provided, assume uncorrelated
        if correlation_matrix is None:
            for asset in asset_returns:
                positions[asset] = self.config.target_vol / asset_vols[asset]
                positions[asset] = np.clip(
                    positions[asset],
                    self.config.min_leverage,
                    self.config.max_leverage
                )
        else:
            # Adjust for correlations using portfolio variance formula
            # This is a simplified approach; full mean-variance optimization
            # would be more sophisticated
            n_assets = len(asset_returns)
            equal_weight = 1.0 / n_assets

            for asset in asset_returns:
                # Start with volatility-adjusted position
                base_position = self.config.target_vol / asset_vols[asset]

                # Adjust for average correlation with other assets
                avg_correlation = correlation_matrix.loc[asset].drop(asset).mean()

                # Reduce position if highly correlated with others
                correlation_adjustment = 1.0 / (1.0 + abs(avg_correlation))
                adjusted_position = base_position * correlation_adjustment

                positions[asset] = np.clip(
                    adjusted_position,
                    self.config.min_leverage,
                    self.config.max_leverage
                )

        # Normalize positions to sum to 1.0 (fully invested)
        total_position = sum(positions.values())
        if total_position > 0:
            positions = {k: v / total_position for k, v in positions.items()}

        logger.info(
            "multi_asset_positions_calculated",
            n_assets=len(positions),
            positions=positions,
        )

        return positions

    def generate_rebalance_schedule(
        self,
        portfolio_returns: pd.Series,
        current_exposure: float,
        forecast_days: int = 20
    ) -> pd.DataFrame:
        """Generate a rebalancing schedule based on recent volatility trends.

        Args:
            portfolio_returns: Historical portfolio returns
            current_exposure: Current portfolio exposure
            forecast_days: Number of days to forecast

        Returns:
            DataFrame with columns: date, realized_vol, target_exposure, rebalance
        """
        schedule = []

        # Get rolling volatility over recent history
        rolling_vol = portfolio_returns.rolling(
            window=self.config.lookback_days
        ).std() * np.sqrt(252)

        # Calculate current trend (simple linear regression on recent vols)
        recent_vols = rolling_vol.tail(self.config.lookback_days).dropna()
        if len(recent_vols) >= 5:
            x = np.arange(len(recent_vols))
            coeffs = np.polyfit(x, recent_vols.values, deg=1)
            vol_trend = coeffs[0]  # Slope
        else:
            vol_trend = 0.0

        # Forecast future volatility (simple extrapolation)
        last_vol = rolling_vol.iloc[-1]
        for day in range(forecast_days):
            forecast_vol = last_vol + vol_trend * day
            forecast_vol = max(forecast_vol, self.config.vol_floor)

            # Calculate target exposure
            target_exposure = self.config.target_vol / forecast_vol
            target_exposure = np.clip(
                target_exposure,
                self.config.min_leverage,
                self.config.max_leverage
            )

            # Check if rebalance needed
            deviation = abs(target_exposure - current_exposure) / target_exposure
            rebalance = deviation > self.config.rebalance_threshold

            schedule.append({
                "day": day + 1,
                "forecast_vol": forecast_vol,
                "target_exposure": target_exposure,
                "current_exposure": current_exposure,
                "deviation_pct": deviation * 100,
                "rebalance": rebalance,
            })

            # Update current exposure if rebalancing
            if rebalance:
                current_exposure = target_exposure

        return pd.DataFrame(schedule)
