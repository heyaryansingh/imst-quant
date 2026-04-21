"""Factor analysis and risk decomposition utilities.

Provides tools for analyzing portfolio factor exposures, including:
- Fama-French factor model estimation (MKT, SMB, HML, MOM, etc.)
- Beta decomposition (market, sector, idiosyncratic)
- Factor return attribution
- Rolling factor exposure tracking
- Factor mimicking portfolio construction

Example:
    >>> from imst_quant.utils.factor_analysis import FactorAnalyzer
    >>> analyzer = FactorAnalyzer(returns, factor_returns)
    >>> exposures = analyzer.estimate_exposures()
    >>> print(f"Market beta: {exposures['MKT']:.2f}")
    Market beta: 1.15

References:
    - Fama, French (1993): Common risk factors in stock/bond returns
    - Carhart (1997): On persistence in mutual fund performance
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class FactorExposures:
    """Container for factor model results.

    Attributes:
        betas: Dict mapping factor names to beta coefficients.
        alpha: Jensen's alpha (intercept).
        r_squared: Model R-squared (explained variance).
        adj_r_squared: Adjusted R-squared.
        residual_vol: Idiosyncratic volatility.
        t_stats: T-statistics for each beta.
        p_values: P-values for significance tests.
        factor_contributions: Variance contribution by factor.
    """
    betas: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    residual_vol: float = 0.0
    t_stats: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    factor_contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskDecomposition:
    """Portfolio risk broken down by source.

    Attributes:
        total_risk: Total portfolio volatility.
        systematic_risk: Risk from factor exposures.
        idiosyncratic_risk: Residual risk.
        factor_risks: Risk contribution by factor.
        diversification_benefit: Risk reduction from diversification.
    """
    total_risk: float = 0.0
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    factor_risks: dict[str, float] = field(default_factory=dict)
    diversification_benefit: float = 0.0


def estimate_factor_exposures(
    returns: pl.Series,
    factor_returns: pl.DataFrame,
    risk_free_rate: float = 0.0,
) -> FactorExposures:
    """Estimate factor exposures using OLS regression.

    Regresses portfolio returns on factor returns to estimate betas
    and alpha using the standard factor model:

        R_p - R_f = alpha + sum(beta_i * F_i) + epsilon

    Args:
        returns: Portfolio or asset returns series.
        factor_returns: DataFrame with factor return columns.
        risk_free_rate: Risk-free rate for excess return calculation.

    Returns:
        FactorExposures with estimated betas, alpha, and statistics.

    Example:
        >>> factors = pl.DataFrame({
        ...     "MKT": market_excess_returns,
        ...     "SMB": smb_returns,
        ...     "HML": hml_returns,
        ... })
        >>> exposures = estimate_factor_exposures(portfolio_returns, factors)
    """
    # Convert to numpy for regression
    y = returns.drop_nulls().to_numpy() - risk_free_rate

    factor_cols = [c for c in factor_returns.columns if c not in ["date", "timestamp"]]

    # Build design matrix
    X_data = []
    for col in factor_cols:
        X_data.append(factor_returns[col].drop_nulls().to_numpy())

    # Align lengths
    min_len = min(len(y), min(len(x) for x in X_data))
    y = y[-min_len:]
    X = np.column_stack([x[-min_len:] for x in X_data])

    # Add intercept
    X = np.column_stack([np.ones(len(y)), X])

    # OLS regression: (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        # Handle singular matrix
        betas = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calculate residuals and statistics
    y_pred = X @ betas
    residuals = y - y_pred

    n = len(y)
    k = len(factor_cols)

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0.0

    # Standard errors and t-stats
    mse = ss_res / (n - k - 1) if n > k + 1 else ss_res
    se = np.sqrt(np.diag(XtX_inv) * mse) if mse > 0 else np.zeros(k + 1)
    t_stats = betas / se if np.all(se > 0) else np.zeros(k + 1)

    # P-values (approximate using normal distribution for large samples)
    from scipy import stats
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

    # Idiosyncratic volatility
    residual_vol = np.std(residuals) * np.sqrt(252)  # Annualized

    # Factor variance contributions
    factor_var = np.var(y_pred)
    total_var = np.var(y)

    factor_contributions = {}
    for i, col in enumerate(factor_cols):
        factor_contribution = betas[i + 1] ** 2 * np.var(X[:, i + 1])
        factor_contributions[col] = factor_contribution / total_var if total_var > 0 else 0.0

    # Build result
    result = FactorExposures(
        betas={col: float(betas[i + 1]) for i, col in enumerate(factor_cols)},
        alpha=float(betas[0]) * 252,  # Annualized
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        residual_vol=float(residual_vol),
        t_stats={col: float(t_stats[i + 1]) for i, col in enumerate(factor_cols)},
        p_values={col: float(p_values[i + 1]) for i, col in enumerate(factor_cols)},
        factor_contributions=factor_contributions,
    )

    return result


def decompose_risk(
    returns: pl.Series,
    factor_returns: pl.DataFrame,
    exposures: Optional[FactorExposures] = None,
) -> RiskDecomposition:
    """Decompose total risk into systematic and idiosyncratic components.

    Args:
        returns: Portfolio returns.
        factor_returns: Factor return series.
        exposures: Pre-computed exposures (computed if not provided).

    Returns:
        RiskDecomposition with risk breakdown.
    """
    if exposures is None:
        exposures = estimate_factor_exposures(returns, factor_returns)

    # Total risk (annualized volatility)
    total_risk = float(returns.std()) * np.sqrt(252)

    # Systematic risk from factor exposures
    factor_cols = [c for c in factor_returns.columns if c not in ["date", "timestamp"]]

    factor_risks = {}
    systematic_var = 0.0

    for col in factor_cols:
        factor_vol = float(factor_returns[col].std()) * np.sqrt(252)
        beta = exposures.betas.get(col, 0.0)
        factor_risk = abs(beta) * factor_vol
        factor_risks[col] = factor_risk
        systematic_var += (beta * factor_vol) ** 2

    systematic_risk = np.sqrt(systematic_var)
    idiosyncratic_risk = exposures.residual_vol

    # Diversification benefit
    sum_factor_risks = sum(factor_risks.values())
    diversification_benefit = sum_factor_risks - systematic_risk if sum_factor_risks > 0 else 0.0

    return RiskDecomposition(
        total_risk=total_risk,
        systematic_risk=systematic_risk,
        idiosyncratic_risk=idiosyncratic_risk,
        factor_risks=factor_risks,
        diversification_benefit=diversification_benefit,
    )


def rolling_factor_exposures(
    returns: pl.Series,
    factor_returns: pl.DataFrame,
    window: int = 60,
) -> pl.DataFrame:
    """Calculate rolling factor exposures over time.

    Args:
        returns: Portfolio returns.
        factor_returns: Factor return series.
        window: Rolling window size in periods.

    Returns:
        DataFrame with rolling betas and alpha.
    """
    factor_cols = [c for c in factor_returns.columns if c not in ["date", "timestamp"]]

    y = returns.drop_nulls().to_numpy()
    X_data = {col: factor_returns[col].drop_nulls().to_numpy() for col in factor_cols}

    # Align lengths
    min_len = min(len(y), min(len(x) for x in X_data.values()))
    y = y[-min_len:]
    X_data = {k: v[-min_len:] for k, v in X_data.items()}

    results = {col: [] for col in factor_cols}
    results["alpha"] = []
    results["r_squared"] = []

    for i in range(window, len(y) + 1):
        y_window = y[i - window:i]
        X_window = np.column_stack([
            np.ones(window),
            *[X_data[col][i - window:i] for col in factor_cols]
        ])

        try:
            betas = np.linalg.lstsq(X_window, y_window, rcond=None)[0]
            y_pred = X_window @ betas
            ss_res = np.sum((y_window - y_pred) ** 2)
            ss_tot = np.sum((y_window - np.mean(y_window)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            betas = np.zeros(len(factor_cols) + 1)
            r_sq = 0.0

        results["alpha"].append(float(betas[0]) * 252)
        results["r_squared"].append(float(r_sq))
        for j, col in enumerate(factor_cols):
            results[col].append(float(betas[j + 1]))

    # Pad beginning with NaN
    pad_len = min_len - len(results["alpha"])
    for key in results:
        results[key] = [None] * pad_len + results[key]

    return pl.DataFrame(results)


def generate_synthetic_factors(
    market_returns: pl.Series,
    n_periods: Optional[int] = None,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """Generate synthetic factor returns for testing.

    Creates approximate SMB, HML, and MOM factors based on
    market returns with appropriate correlations.

    Args:
        market_returns: Market excess returns.
        n_periods: Number of periods (uses market length if None).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with MKT, SMB, HML, MOM columns.
    """
    if seed is not None:
        np.random.seed(seed)

    mkt = market_returns.drop_nulls().to_numpy()
    n = n_periods or len(mkt)

    # Generate correlated factors
    # SMB: Low correlation with market, positive mean
    smb = np.random.normal(0.0001, 0.005, n) + 0.1 * mkt[:n]

    # HML: Low correlation with market, positive mean
    hml = np.random.normal(0.00015, 0.006, n) + 0.05 * mkt[:n]

    # MOM: Moderate correlation, higher vol
    mom = np.random.normal(0.0002, 0.008, n) + 0.2 * mkt[:n]

    return pl.DataFrame({
        "MKT": mkt[:n],
        "SMB": smb,
        "HML": hml,
        "MOM": mom,
    })


def factor_attribution(
    returns: pl.Series,
    factor_returns: pl.DataFrame,
    exposures: Optional[FactorExposures] = None,
) -> dict:
    """Attribute portfolio returns to factor exposures.

    Decomposes total return into contributions from:
    - Alpha (skill/luck)
    - Each factor exposure
    - Residual (unexplained)

    Args:
        returns: Portfolio returns.
        factor_returns: Factor return series.
        exposures: Pre-computed exposures.

    Returns:
        Dict with return attribution breakdown.
    """
    if exposures is None:
        exposures = estimate_factor_exposures(returns, factor_returns)

    factor_cols = [c for c in factor_returns.columns if c not in ["date", "timestamp"]]

    # Total return (annualized)
    total_return = float(returns.mean()) * 252

    # Factor contributions
    contributions = {"alpha": exposures.alpha}

    for col in factor_cols:
        factor_mean = float(factor_returns[col].mean()) * 252
        beta = exposures.betas.get(col, 0.0)
        contributions[col] = beta * factor_mean

    # Residual
    explained = sum(contributions.values())
    contributions["residual"] = total_return - explained
    contributions["total"] = total_return

    # Percentages
    contributions["pct_alpha"] = contributions["alpha"] / total_return if total_return != 0 else 0
    for col in factor_cols:
        contributions[f"pct_{col}"] = contributions[col] / total_return if total_return != 0 else 0

    return contributions
