"""Sector and industry sentiment aggregation utilities.

This module provides tools for aggregating ticker-level sentiment to
sector level, calculating sector rotation signals, and analyzing
cross-sector sentiment dispersion for quantitative trading strategies.

Functions:
    aggregate_sector_sentiment: Aggregate ticker sentiment to sector level
    sector_sentiment_momentum: Calculate rolling sector sentiment momentum
    sector_rotation_signal: Generate overweight/underweight sector signals
    cross_sector_dispersion: Analyze sentiment dispersion across sectors
    sector_sentiment_summary: Full summary with all sector metrics

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.sector_sentiment import (
    ...     aggregate_sector_sentiment,
    ...     sector_rotation_signal,
    ...     sector_sentiment_summary,
    ... )
    >>> df = pl.DataFrame({
    ...     "ticker": ["AAPL", "MSFT", "JPM", "XOM"],
    ...     "sentiment": [0.8, 0.6, -0.2, 0.1],
    ... })
    >>> sector_results = aggregate_sector_sentiment(df)
    >>> for s in sector_results:
    ...     print(f"{s.sector}: avg={s.avg_sentiment:.2f}")
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl


# GICS Sector Mapping for common tickers
# Covers all 11 GICS sectors with representative stocks
SECTOR_MAP: Dict[str, str] = {
    # Technology (Information Technology)
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "ADBE": "Technology",
    # Health Care
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "PFE": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "LLY": "Health Care",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "BRK.B": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "PM": "Consumer Staples",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    # Industrials
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "UNP": "Industrials",
    "GE": "Industrials",
    "RTX": "Industrials",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "DD": "Materials",
    "NEM": "Materials",
    "FCX": "Materials",
    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "SPG": "Real Estate",
    "O": "Real Estate",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "XEL": "Utilities",
    # Communication Services
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "CMCSA": "Communication Services",
}

# All GICS sectors for reference
GICS_SECTORS: List[str] = [
    "Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Industrials",
    "Materials",
    "Real Estate",
    "Utilities",
    "Communication Services",
]


@dataclass
class SectorSentiment:
    """Aggregated sentiment metrics for a single sector.

    Attributes:
        sector: GICS sector name.
        avg_sentiment: Mean sentiment score across sector assets.
        median_sentiment: Median sentiment score (robust to outliers).
        std_sentiment: Standard deviation of sentiment within sector.
        num_assets: Number of assets contributing to the aggregation.
        bullish_pct: Percentage of assets with positive sentiment.
        bearish_pct: Percentage of assets with negative sentiment.
        neutral_pct: Percentage of assets with neutral sentiment (~0).
        strongest_ticker: Ticker with highest sentiment in sector.
        weakest_ticker: Ticker with lowest sentiment in sector.
    """

    sector: str
    avg_sentiment: float
    median_sentiment: float
    std_sentiment: float
    num_assets: int
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    strongest_ticker: str
    weakest_ticker: str


@dataclass
class SectorRotationSignal:
    """Sector rotation signal based on relative sentiment strength.

    Attributes:
        date: Signal generation date (ISO format string).
        overweight_sectors: Sectors with above-threshold sentiment.
        underweight_sectors: Sectors with below-threshold sentiment.
        neutral_sectors: Sectors within neutral threshold.
        rotation_strength: Magnitude of rotation signal (0-1 scale).
    """

    date: str
    overweight_sectors: List[str]
    underweight_sectors: List[str]
    neutral_sectors: List[str]
    rotation_strength: float


def aggregate_sector_sentiment(
    df: pl.DataFrame,
    ticker_col: str = "ticker",
    sentiment_col: str = "sentiment",
    sector_map: Optional[Dict[str, str]] = None,
    neutral_threshold: float = 0.05,
) -> List[SectorSentiment]:
    """Aggregate ticker-level sentiment to sector level.

    Computes sector-level sentiment statistics by grouping tickers
    according to their GICS sector classification. Calculates mean,
    median, standard deviation, and sentiment distribution metrics.

    Args:
        df: DataFrame with ticker and sentiment columns.
        ticker_col: Column name for ticker symbols.
        sentiment_col: Column name for sentiment values (typically -1 to 1).
        sector_map: Custom ticker-to-sector mapping. Uses SECTOR_MAP if None.
        neutral_threshold: Threshold for classifying sentiment as neutral.
            Values in [-threshold, +threshold] are neutral.

    Returns:
        List of SectorSentiment objects, one per sector present in data.

    Example:
        >>> df = pl.DataFrame({
        ...     "ticker": ["AAPL", "MSFT", "JPM", "BAC"],
        ...     "sentiment": [0.5, 0.7, -0.1, 0.2]
        ... })
        >>> results = aggregate_sector_sentiment(df)
        >>> tech = next(s for s in results if s.sector == "Technology")
        >>> print(f"Tech avg: {tech.avg_sentiment:.2f}, n={tech.num_assets}")
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    if df.is_empty():
        return []

    # Validate required columns
    if ticker_col not in df.columns or sentiment_col not in df.columns:
        return []

    # Map tickers to sectors
    df = df.with_columns(
        pl.col(ticker_col)
        .replace(sector_map, default=None)
        .alias("_sector")
    )

    # Filter to only mapped tickers
    df_mapped = df.filter(pl.col("_sector").is_not_null())

    if df_mapped.is_empty():
        return []

    # Group by sector and compute aggregations
    results: List[SectorSentiment] = []

    for sector in df_mapped["_sector"].unique().sort().to_list():
        sector_df = df_mapped.filter(pl.col("_sector") == sector)
        sentiments = sector_df[sentiment_col].drop_nulls()

        if sentiments.is_empty():
            continue

        n_assets = sentiments.len()
        avg_sent = sentiments.mean()
        median_sent = sentiments.median()
        std_sent = sentiments.std() if n_assets > 1 else 0.0

        # Handle None values from statistics
        if avg_sent is None:
            avg_sent = 0.0
        if median_sent is None:
            median_sent = 0.0
        if std_sent is None:
            std_sent = 0.0

        # Calculate sentiment distribution
        bullish_count = sentiments.filter(sentiments > neutral_threshold).len()
        bearish_count = sentiments.filter(sentiments < -neutral_threshold).len()
        neutral_count = n_assets - bullish_count - bearish_count

        bullish_pct = bullish_count / n_assets * 100 if n_assets > 0 else 0.0
        bearish_pct = bearish_count / n_assets * 100 if n_assets > 0 else 0.0
        neutral_pct = neutral_count / n_assets * 100 if n_assets > 0 else 0.0

        # Find strongest and weakest tickers
        sector_with_sent = sector_df.select([ticker_col, sentiment_col]).drop_nulls()

        if sector_with_sent.is_empty():
            strongest = ""
            weakest = ""
        else:
            max_idx = sector_with_sent[sentiment_col].arg_max()
            min_idx = sector_with_sent[sentiment_col].arg_min()

            strongest = sector_with_sent[ticker_col][max_idx] if max_idx is not None else ""
            weakest = sector_with_sent[ticker_col][min_idx] if min_idx is not None else ""

        results.append(
            SectorSentiment(
                sector=str(sector),
                avg_sentiment=float(avg_sent),
                median_sentiment=float(median_sent),
                std_sentiment=float(std_sent),
                num_assets=int(n_assets),
                bullish_pct=float(bullish_pct),
                bearish_pct=float(bearish_pct),
                neutral_pct=float(neutral_pct),
                strongest_ticker=str(strongest),
                weakest_ticker=str(weakest),
            )
        )

    return results


def sector_sentiment_momentum(
    df: pl.DataFrame,
    ticker_col: str = "ticker",
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    window: int = 5,
    sector_map: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Calculate rolling sector sentiment momentum.

    Computes the change in average sector sentiment over a rolling
    window period. Positive momentum indicates improving sector
    sentiment, useful for trend-following sector rotation strategies.

    Args:
        df: DataFrame with ticker, sentiment, and date columns.
        ticker_col: Column name for ticker symbols.
        sentiment_col: Column name for sentiment values.
        date_col: Column name for dates.
        window: Rolling window size in periods for momentum calculation.
        sector_map: Custom ticker-to-sector mapping.

    Returns:
        DataFrame with columns: date, sector, avg_sentiment,
        sentiment_momentum (change over window).

    Example:
        >>> df = pl.DataFrame({
        ...     "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
        ...     "ticker": ["AAPL", "MSFT", "JPM", "BAC"] * 2,
        ...     "sentiment": [0.5, 0.6, 0.1, 0.2, 0.7, 0.8, 0.0, 0.1]
        ... })
        >>> momentum = sector_sentiment_momentum(df, window=1)
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    if df.is_empty():
        return pl.DataFrame({
            date_col: [],
            "sector": [],
            "avg_sentiment": [],
            "sentiment_momentum": [],
        })

    # Validate required columns
    required_cols = [ticker_col, sentiment_col, date_col]
    if not all(col in df.columns for col in required_cols):
        return pl.DataFrame({
            date_col: [],
            "sector": [],
            "avg_sentiment": [],
            "sentiment_momentum": [],
        })

    # Map tickers to sectors
    df = df.with_columns(
        pl.col(ticker_col)
        .replace(sector_map, default=None)
        .alias("_sector")
    )

    # Filter to mapped tickers only
    df_mapped = df.filter(pl.col("_sector").is_not_null())

    if df_mapped.is_empty():
        return pl.DataFrame({
            date_col: [],
            "sector": [],
            "avg_sentiment": [],
            "sentiment_momentum": [],
        })

    # Aggregate sentiment by date and sector
    daily_sector = (
        df_mapped
        .group_by([date_col, "_sector"])
        .agg(pl.col(sentiment_col).mean().alias("avg_sentiment"))
        .sort([date_col, "_sector"])
    )

    # Calculate rolling momentum per sector
    result = (
        daily_sector
        .with_columns(
            pl.col("avg_sentiment")
            .diff(n=window)
            .over("_sector")
            .alias("sentiment_momentum")
        )
        .rename({"_sector": "sector"})
    )

    return result


def sector_rotation_signal(
    sector_sentiments: List[SectorSentiment],
    threshold: float = 0.1,
) -> SectorRotationSignal:
    """Generate sector rotation signals based on relative sentiment.

    Classifies sectors into overweight, underweight, or neutral based
    on their average sentiment relative to the cross-sector mean.
    Used for tactical sector allocation decisions.

    Args:
        sector_sentiments: List of SectorSentiment from aggregate_sector_sentiment.
        threshold: Deviation from mean required for overweight/underweight.
            Sectors within +/- threshold of mean are neutral.

    Returns:
        SectorRotationSignal with sector classifications and rotation strength.

    Example:
        >>> sectors = aggregate_sector_sentiment(df)
        >>> signal = sector_rotation_signal(sectors, threshold=0.1)
        >>> print(f"Overweight: {signal.overweight_sectors}")
        >>> print(f"Rotation strength: {signal.rotation_strength:.2f}")
    """
    if not sector_sentiments:
        return SectorRotationSignal(
            date=datetime.now().strftime("%Y-%m-%d"),
            overweight_sectors=[],
            underweight_sectors=[],
            neutral_sectors=[],
            rotation_strength=0.0,
        )

    # Calculate cross-sector mean sentiment (weighted by num_assets)
    total_assets = sum(s.num_assets for s in sector_sentiments)
    if total_assets == 0:
        mean_sentiment = 0.0
    else:
        weighted_sum = sum(s.avg_sentiment * s.num_assets for s in sector_sentiments)
        mean_sentiment = weighted_sum / total_assets

    # Classify sectors relative to mean
    overweight: List[str] = []
    underweight: List[str] = []
    neutral: List[str] = []

    deviations: List[float] = []

    for s in sector_sentiments:
        deviation = s.avg_sentiment - mean_sentiment
        deviations.append(abs(deviation))

        if deviation > threshold:
            overweight.append(s.sector)
        elif deviation < -threshold:
            underweight.append(s.sector)
        else:
            neutral.append(s.sector)

    # Rotation strength: normalized measure of sector differentiation
    # Higher values indicate more conviction in rotation signals
    if deviations:
        max_deviation = max(deviations)
        avg_deviation = sum(deviations) / len(deviations)
        # Scale to 0-1 range (cap at 1.0 for very extreme readings)
        rotation_strength = min(1.0, avg_deviation / 0.5) if avg_deviation > 0 else 0.0
    else:
        rotation_strength = 0.0

    return SectorRotationSignal(
        date=datetime.now().strftime("%Y-%m-%d"),
        overweight_sectors=sorted(overweight),
        underweight_sectors=sorted(underweight),
        neutral_sectors=sorted(neutral),
        rotation_strength=float(rotation_strength),
    )


def cross_sector_dispersion(
    df: pl.DataFrame,
    ticker_col: str = "ticker",
    sentiment_col: str = "sentiment",
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Calculate cross-sector sentiment dispersion.

    Measures how differentiated sector sentiments are from each other.
    High dispersion indicates sector-specific drivers (stock picking
    matters). Low dispersion indicates macro-driven markets (beta matters).

    Args:
        df: DataFrame with ticker and sentiment columns.
        ticker_col: Column name for ticker symbols.
        sentiment_col: Column name for sentiment values.
        sector_map: Custom ticker-to-sector mapping.

    Returns:
        Dictionary containing:
        - dispersion: Standard deviation of sector average sentiments
        - range: Spread between highest and lowest sector sentiment
        - top_sector: Sector with highest average sentiment
        - bottom_sector: Sector with lowest average sentiment
        - interpretation: Human-readable market regime interpretation

    Example:
        >>> dispersion = cross_sector_dispersion(df)
        >>> print(f"Dispersion: {dispersion['dispersion']:.3f}")
        >>> print(f"Regime: {dispersion['interpretation']}")
    """
    sector_sentiments = aggregate_sector_sentiment(
        df, ticker_col, sentiment_col, sector_map
    )

    if not sector_sentiments:
        return {
            "dispersion": 0.0,
            "range": 0.0,
            "top_sector": None,
            "bottom_sector": None,
            "sector_sentiments": {},
            "interpretation": "Insufficient data for dispersion analysis",
        }

    # Extract average sentiments
    avg_sentiments = [s.avg_sentiment for s in sector_sentiments]

    if len(avg_sentiments) < 2:
        return {
            "dispersion": 0.0,
            "range": 0.0,
            "top_sector": sector_sentiments[0].sector if sector_sentiments else None,
            "bottom_sector": sector_sentiments[0].sector if sector_sentiments else None,
            "sector_sentiments": {s.sector: s.avg_sentiment for s in sector_sentiments},
            "interpretation": "Insufficient sectors for dispersion analysis",
        }

    # Calculate dispersion metrics
    mean_sent = sum(avg_sentiments) / len(avg_sentiments)
    variance = sum((s - mean_sent) ** 2 for s in avg_sentiments) / len(avg_sentiments)
    dispersion = variance ** 0.5

    max_sent = max(avg_sentiments)
    min_sent = min(avg_sentiments)
    sent_range = max_sent - min_sent

    # Find top and bottom sectors
    top_sector = max(sector_sentiments, key=lambda s: s.avg_sentiment).sector
    bottom_sector = min(sector_sentiments, key=lambda s: s.avg_sentiment).sector

    # Generate interpretation
    if dispersion > 0.3:
        interpretation = "High dispersion: Strong sector differentiation. Sector rotation and stock selection likely rewarded."
    elif dispersion > 0.15:
        interpretation = "Moderate dispersion: Some sector differentiation. Mixed regime favoring selective exposure."
    elif dispersion > 0.05:
        interpretation = "Low dispersion: Limited sector differentiation. Market largely driven by macro factors."
    else:
        interpretation = "Very low dispersion: Uniform sentiment across sectors. Beta-driven market, sector allocation less important."

    return {
        "dispersion": float(dispersion),
        "range": float(sent_range),
        "top_sector": top_sector,
        "bottom_sector": bottom_sector,
        "sector_sentiments": {s.sector: s.avg_sentiment for s in sector_sentiments},
        "interpretation": interpretation,
    }


def sector_sentiment_summary(
    df: pl.DataFrame,
    ticker_col: str = "ticker",
    sentiment_col: str = "sentiment",
    sector_map: Optional[Dict[str, str]] = None,
    rotation_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Generate comprehensive sector sentiment summary.

    Combines all sector analysis functions into a single summary report
    including sector metrics, rotation signals, and dispersion analysis.

    Args:
        df: DataFrame with ticker and sentiment columns.
        ticker_col: Column name for ticker symbols.
        sentiment_col: Column name for sentiment values.
        sector_map: Custom ticker-to-sector mapping.
        rotation_threshold: Threshold for rotation signal generation.

    Returns:
        Dictionary containing:
        - sector_metrics: List of SectorSentiment as dicts
        - rotation_signal: SectorRotationSignal as dict
        - dispersion: Cross-sector dispersion analysis
        - coverage: Stats on ticker/sector coverage
        - top_picks: Best sentiment tickers per sector
        - bottom_picks: Worst sentiment tickers per sector

    Example:
        >>> summary = sector_sentiment_summary(df)
        >>> print(f"Sectors covered: {summary['coverage']['sectors_covered']}")
        >>> print(f"Rotation strength: {summary['rotation_signal']['rotation_strength']:.2f}")
        >>> for sector in summary['rotation_signal']['overweight_sectors']:
        ...     print(f"  Overweight: {sector}")
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    # Aggregate sector sentiment
    sector_sentiments = aggregate_sector_sentiment(
        df, ticker_col, sentiment_col, sector_map
    )

    # Generate rotation signal
    rotation = sector_rotation_signal(sector_sentiments, rotation_threshold)

    # Calculate dispersion
    dispersion = cross_sector_dispersion(df, ticker_col, sentiment_col, sector_map)

    # Coverage statistics
    total_tickers = df[ticker_col].n_unique() if ticker_col in df.columns else 0
    mapped_tickers = sum(s.num_assets for s in sector_sentiments)
    sectors_covered = len(sector_sentiments)

    # Extract top and bottom picks
    top_picks: Dict[str, str] = {}
    bottom_picks: Dict[str, str] = {}

    for s in sector_sentiments:
        if s.strongest_ticker:
            top_picks[s.sector] = s.strongest_ticker
        if s.weakest_ticker:
            bottom_picks[s.sector] = s.weakest_ticker

    # Convert dataclasses to dicts for JSON serialization
    sector_metrics_dicts = [
        {
            "sector": s.sector,
            "avg_sentiment": s.avg_sentiment,
            "median_sentiment": s.median_sentiment,
            "std_sentiment": s.std_sentiment,
            "num_assets": s.num_assets,
            "bullish_pct": s.bullish_pct,
            "bearish_pct": s.bearish_pct,
            "neutral_pct": s.neutral_pct,
            "strongest_ticker": s.strongest_ticker,
            "weakest_ticker": s.weakest_ticker,
        }
        for s in sector_sentiments
    ]

    rotation_dict = {
        "date": rotation.date,
        "overweight_sectors": rotation.overweight_sectors,
        "underweight_sectors": rotation.underweight_sectors,
        "neutral_sectors": rotation.neutral_sectors,
        "rotation_strength": rotation.rotation_strength,
    }

    return {
        "sector_metrics": sector_metrics_dicts,
        "rotation_signal": rotation_dict,
        "dispersion": dispersion,
        "coverage": {
            "total_tickers": total_tickers,
            "mapped_tickers": mapped_tickers,
            "unmapped_tickers": total_tickers - mapped_tickers,
            "coverage_pct": mapped_tickers / total_tickers * 100 if total_tickers > 0 else 0.0,
            "sectors_covered": sectors_covered,
            "total_sectors": len(GICS_SECTORS),
        },
        "top_picks": top_picks,
        "bottom_picks": bottom_picks,
    }


def get_sector_for_ticker(
    ticker: str,
    sector_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Get the GICS sector for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        sector_map: Custom mapping. Uses SECTOR_MAP if None.

    Returns:
        Sector name or None if ticker not in mapping.

    Example:
        >>> sector = get_sector_for_ticker("AAPL")
        >>> print(sector)
        Technology
    """
    if sector_map is None:
        sector_map = SECTOR_MAP
    return sector_map.get(ticker.upper())


def filter_by_sector(
    df: pl.DataFrame,
    sectors: List[str],
    ticker_col: str = "ticker",
    sector_map: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Filter DataFrame to include only tickers from specified sectors.

    Args:
        df: DataFrame with ticker column.
        sectors: List of sector names to include.
        ticker_col: Column name for ticker symbols.
        sector_map: Custom ticker-to-sector mapping.

    Returns:
        Filtered DataFrame with only tickers from specified sectors.

    Example:
        >>> tech_energy = filter_by_sector(df, ["Technology", "Energy"])
        >>> print(tech_energy[ticker_col].unique().to_list())
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    if ticker_col not in df.columns:
        return df

    # Create set of valid tickers for specified sectors
    valid_tickers = {
        ticker for ticker, sector in sector_map.items()
        if sector in sectors
    }

    return df.filter(pl.col(ticker_col).is_in(valid_tickers))
