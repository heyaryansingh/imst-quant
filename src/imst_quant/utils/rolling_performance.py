"""
Rolling performance metrics calculator.

Provides methods to calculate rolling performance metrics for portfolio analysis
including returns, risk-adjusted metrics, and drawdowns.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def calculate_rolling_returns(
    prices: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling annualized returns.

    Args:
        prices: Price series
        window: Rolling window size
        annualization_factor: Periods per year (252 for daily, 12 for monthly)

    Returns:
        Rolling returns series
    """
    returns = prices.pct_change()
    rolling_returns = returns.rolling(window).apply(
        lambda x: (1 + x).prod() ** (annualization_factor / len(x)) - 1,
        raw=True
    )
    return rolling_returns


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling annualized volatility.

    Args:
        returns: Returns series
        window: Rolling window size
        annualization_factor: Periods per year

    Returns:
        Rolling volatility series
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(annualization_factor)
    return rolling_vol


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Returns series
        window: Rolling window size
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Periods per year

    Returns:
        Rolling Sharpe ratio series
    """
    excess_returns = returns - risk_free_rate / annualization_factor

    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = excess_returns.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization_factor)

    return rolling_sharpe


def calculate_rolling_sortino(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling Sortino ratio.

    Args:
        returns: Returns series
        window: Rolling window size
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Periods per year

    Returns:
        Rolling Sortino ratio series
    """
    excess_returns = returns - risk_free_rate / annualization_factor

    rolling_mean = excess_returns.rolling(window).mean()

    # Downside deviation
    def downside_std(x):
        downside = x[x < 0]
        if len(downside) < 2:
            return np.nan
        return np.std(downside)

    rolling_downside_std = excess_returns.rolling(window).apply(downside_std, raw=False)

    rolling_sortino = (rolling_mean / rolling_downside_std) * np.sqrt(annualization_factor)

    return rolling_sortino


def calculate_rolling_calmar(
    prices: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling Calmar ratio (return / max drawdown).

    Args:
        prices: Price series
        window: Rolling window size
        annualization_factor: Periods per year

    Returns:
        Rolling Calmar ratio series
    """
    returns = prices.pct_change()

    def calmar_ratio(price_window):
        if len(price_window) < 2:
            return np.nan

        # Annualized return
        total_return = (price_window[-1] / price_window[0]) ** (annualization_factor / len(price_window)) - 1

        # Max drawdown
        running_max = np.maximum.accumulate(price_window)
        drawdowns = (price_window - running_max) / running_max
        max_dd = abs(drawdowns.min())

        if max_dd == 0:
            return np.nan

        return total_return / max_dd

    rolling_calmar = prices.rolling(window).apply(calmar_ratio, raw=True)

    return rolling_calmar


def calculate_rolling_drawdown(
    prices: pd.Series,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate rolling maximum drawdown metrics.

    Args:
        prices: Price series
        window: Rolling window size (None for expanding window)

    Returns:
        DataFrame with drawdown, running_max, and max_drawdown
    """
    if window is None:
        # Expanding window
        running_max = prices.expanding().max()
    else:
        # Rolling window
        running_max = prices.rolling(window).max()

    drawdown = (prices - running_max) / running_max

    if window is None:
        max_drawdown = drawdown.expanding().min()
    else:
        max_drawdown = drawdown.rolling(window).min()

    return pd.DataFrame({
        'drawdown': drawdown,
        'running_max': running_max,
        'max_drawdown': max_drawdown,
    })


def calculate_rolling_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling beta against market.

    Args:
        returns: Asset returns
        market_returns: Market returns
        window: Rolling window size

    Returns:
        Rolling beta series
    """
    def beta(asset_window, market_window):
        if len(asset_window) < 2:
            return np.nan
        cov = np.cov(asset_window, market_window)[0, 1]
        market_var = np.var(market_window)
        if market_var == 0:
            return np.nan
        return cov / market_var

    rolling_beta = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        asset_window = returns.iloc[i - window:i].values
        market_window = market_returns.iloc[i - window:i].values
        rolling_beta.iloc[i] = beta(asset_window, market_window)

    return rolling_beta


def calculate_rolling_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling alpha (Jensen's alpha).

    Alpha = Portfolio Return - (Risk-Free Rate + Beta * (Market Return - Risk-Free Rate))

    Args:
        returns: Asset returns
        market_returns: Market returns
        window: Rolling window size
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Periods per year

    Returns:
        Rolling alpha series
    """
    rf_period = risk_free_rate / annualization_factor

    rolling_beta = calculate_rolling_beta(returns, market_returns, window)

    excess_returns = returns - rf_period
    excess_market = market_returns - rf_period

    rolling_portfolio_return = excess_returns.rolling(window).mean()
    rolling_market_return = excess_market.rolling(window).mean()

    rolling_alpha = rolling_portfolio_return - rolling_beta * rolling_market_return

    # Annualize
    rolling_alpha = rolling_alpha * annualization_factor

    return rolling_alpha


def calculate_rolling_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling information ratio.

    IR = (Portfolio Return - Benchmark Return) / Tracking Error

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size
        annualization_factor: Periods per year

    Returns:
        Rolling information ratio series
    """
    active_returns = returns - benchmark_returns

    rolling_mean = active_returns.rolling(window).mean()
    rolling_std = active_returns.rolling(window).std()

    rolling_ir = (rolling_mean / rolling_std) * np.sqrt(annualization_factor)

    return rolling_ir


def calculate_rolling_omega(
    returns: pd.Series,
    window: int,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Calculate rolling Omega ratio.

    Omega = Sum of returns above threshold / Sum of returns below threshold

    Args:
        returns: Returns series
        window: Rolling window size
        threshold: Return threshold (default 0)

    Returns:
        Rolling Omega ratio series
    """
    def omega_ratio(window_returns):
        if len(window_returns) < 2:
            return np.nan

        gains = window_returns[window_returns > threshold] - threshold
        losses = threshold - window_returns[window_returns < threshold]

        gain_sum = gains.sum()
        loss_sum = losses.sum()

        if loss_sum == 0:
            return np.nan

        return gain_sum / loss_sum

    rolling_omega = returns.rolling(window).apply(omega_ratio, raw=False)

    return rolling_omega


def calculate_rolling_win_rate(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling win rate (percentage of positive returns).

    Args:
        returns: Returns series
        window: Rolling window size

    Returns:
        Rolling win rate series
    """
    def win_rate(window_returns):
        if len(window_returns) == 0:
            return np.nan
        return (window_returns > 0).sum() / len(window_returns)

    rolling_win_rate = returns.rolling(window).apply(win_rate, raw=False)

    return rolling_win_rate


def calculate_rolling_profit_factor(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling profit factor (gross profit / gross loss).

    Args:
        returns: Returns series
        window: Rolling window size

    Returns:
        Rolling profit factor series
    """
    def profit_factor(window_returns):
        if len(window_returns) < 2:
            return np.nan

        gross_profit = window_returns[window_returns > 0].sum()
        gross_loss = abs(window_returns[window_returns < 0].sum())

        if gross_loss == 0:
            return np.nan

        return gross_profit / gross_loss

    rolling_pf = returns.rolling(window).apply(profit_factor, raw=False)

    return rolling_pf


def calculate_rolling_ulcer_index(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling Ulcer Index (measure of downside volatility).

    Args:
        prices: Price series
        window: Rolling window size

    Returns:
        Rolling Ulcer Index series
    """
    def ulcer_index(price_window):
        if len(price_window) < 2:
            return np.nan

        running_max = np.maximum.accumulate(price_window)
        drawdowns = ((price_window - running_max) / running_max) * 100

        squared_dd = drawdowns ** 2
        ulcer = np.sqrt(squared_dd.mean())

        return ulcer

    rolling_ui = prices.rolling(window).apply(ulcer_index, raw=True)

    return rolling_ui


def calculate_comprehensive_rolling_metrics(
    prices: pd.Series,
    market_prices: Optional[pd.Series] = None,
    windows: list[int] = [20, 60, 120, 252],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive rolling performance metrics for multiple windows.

    Args:
        prices: Price series
        market_prices: Market price series (optional, for beta/alpha)
        windows: List of window sizes to calculate
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Periods per year

    Returns:
        Dictionary mapping metric names to DataFrames with columns for each window
    """
    returns = prices.pct_change()

    metrics = {}

    # Calculate for each window
    for window in windows:
        col_name = f"window_{window}"

        # Returns and volatility
        if 'returns' not in metrics:
            metrics['returns'] = pd.DataFrame()
        metrics['returns'][col_name] = calculate_rolling_returns(
            prices, window, annualization_factor
        )

        if 'volatility' not in metrics:
            metrics['volatility'] = pd.DataFrame()
        metrics['volatility'][col_name] = calculate_rolling_volatility(
            returns, window, annualization_factor
        )

        # Risk-adjusted metrics
        if 'sharpe' not in metrics:
            metrics['sharpe'] = pd.DataFrame()
        metrics['sharpe'][col_name] = calculate_rolling_sharpe(
            returns, window, risk_free_rate, annualization_factor
        )

        if 'sortino' not in metrics:
            metrics['sortino'] = pd.DataFrame()
        metrics['sortino'][col_name] = calculate_rolling_sortino(
            returns, window, risk_free_rate, annualization_factor
        )

        if 'calmar' not in metrics:
            metrics['calmar'] = pd.DataFrame()
        metrics['calmar'][col_name] = calculate_rolling_calmar(
            prices, window, annualization_factor
        )

        # Market-relative metrics (if market data provided)
        if market_prices is not None:
            market_returns = market_prices.pct_change()

            if 'beta' not in metrics:
                metrics['beta'] = pd.DataFrame()
            metrics['beta'][col_name] = calculate_rolling_beta(
                returns, market_returns, window
            )

            if 'alpha' not in metrics:
                metrics['alpha'] = pd.DataFrame()
            metrics['alpha'][col_name] = calculate_rolling_alpha(
                returns, market_returns, window, risk_free_rate, annualization_factor
            )

    return metrics
