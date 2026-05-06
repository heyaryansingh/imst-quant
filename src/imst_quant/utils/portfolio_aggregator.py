"""Portfolio-level analytics aggregation and multi-asset reporting.

This module provides utilities to aggregate metrics across multiple assets,
compute portfolio-level statistics, and generate comprehensive performance
reports for multi-asset strategies.

Functions:
    aggregate_portfolio_metrics: Compute portfolio-level performance metrics
    calculate_portfolio_weights: Determine optimal portfolio weights
    compute_cross_asset_correlations: Calculate correlation matrix
    generate_portfolio_report: Create comprehensive portfolio report
    compare_asset_contributions: Analyze individual asset contributions

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.portfolio_aggregator import aggregate_portfolio_metrics
    >>> df = pl.DataFrame({
    ...     "asset": ["BTC", "ETH", "BTC", "ETH"],
    ...     "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    ...     "returns": [0.02, 0.01, -0.01, 0.03],
    ... })
    >>> metrics = aggregate_portfolio_metrics(df, asset_col="asset")
"""

from typing import Dict, List, Optional, Tuple

import polars as pl


def aggregate_portfolio_metrics(
    df: pl.DataFrame,
    asset_col: str = "asset",
    date_col: str = "date",
    returns_col: str = "returns",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute portfolio-level performance metrics from individual assets.

    Args:
        df: DataFrame with multi-asset returns data.
        asset_col: Name of asset identifier column.
        date_col: Name of date column.
        returns_col: Name of returns column.
        weights: Optional dict of asset weights (default: equal weight).

    Returns:
        Dictionary containing:
        - portfolio_return: Aggregate return
        - portfolio_volatility: Portfolio volatility
        - sharpe_ratio: Sharpe ratio (assuming 0% risk-free rate)
        - max_drawdown: Maximum drawdown
        - win_rate: Percentage of positive return days
        - num_assets: Number of assets in portfolio

    Example:
        >>> df = pl.DataFrame({
        ...     "asset": ["BTC", "ETH", "BTC", "ETH"],
        ...     "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        ...     "returns": [0.02, 0.01, -0.01, 0.03],
        ... })
        >>> metrics = aggregate_portfolio_metrics(df)
        >>> print(metrics["portfolio_return"])
    """
    # Get unique assets
    unique_assets = df[asset_col].unique().to_list()
    num_assets = len(unique_assets)

    # Default to equal weights
    if weights is None:
        weights = {asset: 1.0 / num_assets for asset in unique_assets}

    # Pivot to wide format: dates x assets
    pivot_df = df.pivot(
        index=date_col,
        on=asset_col,
        values=returns_col,
    ).fill_null(0)

    # Calculate weighted portfolio returns for each date
    portfolio_returns = pl.lit(0.0)
    for asset in unique_assets:
        if asset in weights and asset in pivot_df.columns:
            portfolio_returns = portfolio_returns + (pl.col(asset) * weights[asset])

    portfolio_df = pivot_df.with_columns(portfolio_returns.alias("portfolio_returns"))

    # Compute metrics
    total_return = portfolio_df["portfolio_returns"].sum()
    volatility = portfolio_df["portfolio_returns"].std()
    mean_return = portfolio_df["portfolio_returns"].mean()

    # Sharpe ratio (annualized)
    sharpe = (mean_return * 252) / (volatility * (252**0.5)) if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + portfolio_df["portfolio_returns"]).cum_prod()
    running_max = cumulative.cum_max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    positive_days = (portfolio_df["portfolio_returns"] > 0).sum()
    total_days = len(portfolio_df)
    win_rate = positive_days / total_days if total_days > 0 else 0.0

    return {
        "portfolio_return": float(total_return),
        "portfolio_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_assets": num_assets,
        "mean_daily_return": float(mean_return),
        "total_days": total_days,
    }


def calculate_portfolio_weights(
    df: pl.DataFrame,
    asset_col: str = "asset",
    returns_col: str = "returns",
    method: str = "equal",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Dict[str, float]:
    """Calculate optimal portfolio weights using various methods.

    Args:
        df: DataFrame with asset returns.
        asset_col: Name of asset identifier column.
        returns_col: Name of returns column.
        method: Weighting method:
            - "equal": Equal weight (1/N)
            - "volatility": Inverse volatility weighting
            - "sharpe": Sharpe ratio weighting
            - "returns": Based on mean returns
        min_weight: Minimum weight per asset (default: 0.0).
        max_weight: Maximum weight per asset (default: 1.0).

    Returns:
        Dictionary mapping asset names to weights (sum = 1.0).

    Example:
        >>> df = pl.DataFrame({
        ...     "asset": ["BTC", "ETH", "BTC", "ETH"],
        ...     "returns": [0.02, 0.01, -0.01, 0.03],
        ... })
        >>> weights = calculate_portfolio_weights(df, method="volatility")
    """
    # Compute per-asset statistics
    asset_stats = (
        df.group_by(asset_col)
        .agg(
            [
                pl.col(returns_col).mean().alias("mean_return"),
                pl.col(returns_col).std().alias("volatility"),
            ]
        )
        .with_columns(
            (pl.col("mean_return") / pl.col("volatility"))
            .fill_null(0)
            .alias("sharpe")
        )
    )

    weights = {}

    if method == "equal":
        num_assets = len(asset_stats)
        for asset in asset_stats[asset_col].to_list():
            weights[asset] = 1.0 / num_assets

    elif method == "volatility":
        # Inverse volatility weighting
        inv_vols = 1.0 / asset_stats["volatility"].fill_null(1.0)
        total = inv_vols.sum()
        for i, asset in enumerate(asset_stats[asset_col].to_list()):
            weights[asset] = float(inv_vols[i] / total)

    elif method == "sharpe":
        # Sharpe ratio weighting (only positive Sharpe)
        sharpe_vals = asset_stats["sharpe"].clip(lower_bound=0)
        total = sharpe_vals.sum()
        if total > 0:
            for i, asset in enumerate(asset_stats[asset_col].to_list()):
                weights[asset] = float(sharpe_vals[i] / total)
        else:
            # Fallback to equal weight
            num_assets = len(asset_stats)
            for asset in asset_stats[asset_col].to_list():
                weights[asset] = 1.0 / num_assets

    elif method == "returns":
        # Mean return weighting (only positive returns)
        return_vals = asset_stats["mean_return"].clip(lower_bound=0)
        total = return_vals.sum()
        if total > 0:
            for i, asset in enumerate(asset_stats[asset_col].to_list()):
                weights[asset] = float(return_vals[i] / total)
        else:
            # Fallback to equal weight
            num_assets = len(asset_stats)
            for asset in asset_stats[asset_col].to_list():
                weights[asset] = 1.0 / num_assets

    # Apply weight constraints
    for asset in weights:
        weights[asset] = max(min_weight, min(max_weight, weights[asset]))

    # Renormalize to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {asset: w / total_weight for asset, w in weights.items()}

    return weights


def compute_cross_asset_correlations(
    df: pl.DataFrame,
    asset_col: str = "asset",
    date_col: str = "date",
    returns_col: str = "returns",
    method: str = "pearson",
) -> pl.DataFrame:
    """Compute correlation matrix between assets.

    Args:
        df: DataFrame with multi-asset returns.
        asset_col: Name of asset identifier column.
        date_col: Name of date column.
        returns_col: Name of returns column.
        method: Correlation method ("pearson" or "spearman").

    Returns:
        Correlation matrix as DataFrame (assets x assets).

    Example:
        >>> df = pl.DataFrame({
        ...     "asset": ["BTC", "ETH", "BTC", "ETH"],
        ...     "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        ...     "returns": [0.02, 0.01, -0.01, 0.03],
        ... })
        >>> corr_matrix = compute_cross_asset_correlations(df)
    """
    # Pivot to wide format
    pivot_df = df.pivot(
        index=date_col,
        on=asset_col,
        values=returns_col,
    ).fill_null(0)

    # Drop date column for correlation calculation
    returns_only = pivot_df.drop(date_col)

    # Calculate correlation matrix
    asset_names = returns_only.columns
    n = len(asset_names)

    # Create correlation matrix
    corr_data = {}
    for col in asset_names:
        correlations = []
        for other_col in asset_names:
            if method == "pearson":
                corr = returns_only.select(pl.corr(col, other_col)).item()
            else:  # spearman
                # Rank transform
                rank1 = returns_only[col].rank()
                rank2 = returns_only[other_col].rank()
                corr = rank1.corr(rank2)
            correlations.append(corr)
        corr_data[col] = correlations

    corr_matrix = pl.DataFrame(corr_data)
    corr_matrix = corr_matrix.with_columns(pl.Series("asset", asset_names))

    # Reorder columns: asset, then all correlation columns
    cols = ["asset"] + asset_names
    corr_matrix = corr_matrix.select(cols)

    return corr_matrix


def generate_portfolio_report(
    df: pl.DataFrame,
    asset_col: str = "asset",
    date_col: str = "date",
    returns_col: str = "returns",
    weights: Optional[Dict[str, float]] = None,
    output_format: str = "dict",
) -> Dict:
    """Generate comprehensive portfolio performance report.

    Args:
        df: DataFrame with multi-asset returns data.
        asset_col: Name of asset identifier column.
        date_col: Name of date column.
        returns_col: Name of returns column.
        weights: Optional asset weights (default: equal weight).
        output_format: Output format ("dict", "dataframe", or "text").

    Returns:
        Dictionary or DataFrame with complete portfolio analytics.

    Example:
        >>> df = pl.DataFrame({
        ...     "asset": ["BTC", "ETH"] * 50,
        ...     "date": [f"2024-01-{i:02d}" for i in range(1, 51)] * 2,
        ...     "returns": [0.01, -0.02] * 50,
        ... })
        >>> report = generate_portfolio_report(df)
    """
    # Get unique assets
    unique_assets = df[asset_col].unique().to_list()

    # Portfolio-level metrics
    portfolio_metrics = aggregate_portfolio_metrics(
        df, asset_col=asset_col, date_col=date_col, returns_col=returns_col, weights=weights
    )

    # Per-asset metrics
    asset_metrics = {}
    for asset in unique_assets:
        asset_data = df.filter(pl.col(asset_col) == asset)
        asset_metrics[asset] = {
            "mean_return": float(asset_data[returns_col].mean()),
            "volatility": float(asset_data[returns_col].std()),
            "sharpe": float(
                asset_data[returns_col].mean()
                / asset_data[returns_col].std()
                if asset_data[returns_col].std() > 0
                else 0
            ),
            "min_return": float(asset_data[returns_col].min()),
            "max_return": float(asset_data[returns_col].max()),
            "weight": weights.get(asset, 1.0 / len(unique_assets))
            if weights
            else 1.0 / len(unique_assets),
        }

    # Correlation matrix
    corr_matrix = compute_cross_asset_correlations(
        df, asset_col=asset_col, date_col=date_col, returns_col=returns_col
    )

    report = {
        "portfolio": portfolio_metrics,
        "assets": asset_metrics,
        "correlation_matrix": corr_matrix,
        "weights": weights
        if weights
        else {asset: 1.0 / len(unique_assets) for asset in unique_assets},
    }

    if output_format == "dict":
        return report
    elif output_format == "dataframe":
        # Convert to DataFrame format
        # This is a simplified version - could be expanded
        return corr_matrix
    else:  # text
        # Return formatted text report
        lines = ["=" * 60]
        lines.append("PORTFOLIO PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append("\nPortfolio Metrics:")
        for key, value in portfolio_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        lines.append("\nAsset Metrics:")
        for asset, metrics in asset_metrics.items():
            lines.append(f"\n  {asset}:")
            for key, value in metrics.items():
                lines.append(f"    {key}: {value:.4f}")
        return "\n".join(lines)


def compare_asset_contributions(
    df: pl.DataFrame,
    asset_col: str = "asset",
    returns_col: str = "returns",
    weights: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Analyze individual asset contributions to portfolio performance.

    Args:
        df: DataFrame with multi-asset returns.
        asset_col: Name of asset identifier column.
        returns_col: Name of returns column.
        weights: Optional asset weights (default: equal weight).

    Returns:
        DataFrame with contribution analysis per asset.

    Example:
        >>> df = pl.DataFrame({
        ...     "asset": ["BTC", "ETH", "SOL"] * 10,
        ...     "returns": [0.02, 0.01, 0.03] * 10,
        ... })
        >>> contributions = compare_asset_contributions(df)
    """
    # Get unique assets
    unique_assets = df[asset_col].unique().to_list()
    num_assets = len(unique_assets)

    # Default weights
    if weights is None:
        weights = {asset: 1.0 / num_assets for asset in unique_assets}

    # Calculate contributions
    contributions = []

    for asset in unique_assets:
        asset_data = df.filter(pl.col(asset_col) == asset)
        mean_return = asset_data[returns_col].mean()
        volatility = asset_data[returns_col].std()
        weight = weights.get(asset, 1.0 / num_assets)

        contribution = {
            "asset": asset,
            "weight": weight,
            "mean_return": mean_return,
            "volatility": volatility,
            "weighted_return": mean_return * weight,
            "risk_contribution": volatility * weight,
        }
        contributions.append(contribution)

    result_df = pl.DataFrame(contributions)

    # Add percentage contributions
    total_weighted_return = result_df["weighted_return"].sum()
    total_risk = result_df["risk_contribution"].sum()

    result_df = result_df.with_columns(
        [
            ((pl.col("weighted_return") / total_weighted_return) * 100).alias(
                "return_contribution_pct"
            ),
            ((pl.col("risk_contribution") / total_risk) * 100).alias("risk_contribution_pct"),
        ]
    )

    return result_df.sort("weighted_return", descending=True)
