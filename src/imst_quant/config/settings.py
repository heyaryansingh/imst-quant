"""Pydantic Settings for IMST-Quant configuration.

This module provides type-safe configuration management using Pydantic BaseSettings.
All configuration values can be overridden via environment variables or .env files.

Example:
    >>> from imst_quant.config.settings import Settings
    >>> settings = Settings()
    >>> print(settings.data.raw_dir)
    data/raw

Environment Variables:
    REDDIT_CLIENT_ID: Reddit API client ID
    REDDIT_CLIENT_SECRET: Reddit API client secret
    DATA_RAW_DIR: Path to raw data directory
    MARKET_EQUITY_TICKERS: Comma-separated list of equity tickers
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedditSettings(BaseSettings):
    """Reddit API configuration for social media data ingestion.

    Attributes:
        client_id: Reddit application client ID.
        client_secret: Reddit application client secret (stored securely).
        user_agent: User agent string for Reddit API requests.
    """

    model_config = SettingsConfigDict(
        env_prefix="REDDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    client_id: str = ""
    client_secret: SecretStr = SecretStr("")
    user_agent: str = "IMST-Quant/1.0"


class DataSettings(BaseSettings):
    """Data paths configuration for the medallion architecture.

    Follows the bronze-silver-gold data lake pattern for organizing
    processed data at different stages of the pipeline.

    Attributes:
        raw_dir: Path to raw, unprocessed data.
        bronze_dir: Path to bronze layer (raw data with schema).
        silver_dir: Path to silver layer (cleaned, normalized data).
        sentiment_dir: Path to sentiment analysis outputs.
        influence_dir: Path to influence graph outputs.
        credibility_dir: Path to credibility scores.
        gold_dir: Path to gold layer (aggregated, ready for ML).
    """

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
    )

    raw_dir: Path = Path("data/raw")
    bronze_dir: Path = Path("data/bronze")
    silver_dir: Path = Path("data/silver")
    sentiment_dir: Path = Path("data/sentiment")
    influence_dir: Path = Path("data/influence")
    credibility_dir: Path = Path("data/credibility")
    gold_dir: Path = Path("data/gold")


class MarketSettings(BaseSettings):
    """Market data configuration for financial data sources.

    Attributes:
        equity_tickers: List of equity ticker symbols to track.
        crypto_pairs: List of cryptocurrency trading pairs to track.
    """

    model_config = SettingsConfigDict(
        env_prefix="MARKET_",
        env_file=".env",
    )

    equity_tickers: list[str] = ["AAPL", "JNJ", "JPM", "XOM"]
    crypto_pairs: list[str] = ["BTC/USDT", "ETH/USDT"]


class Settings(BaseSettings):
    """Root settings composing all configuration sections.

    This is the main entry point for accessing application configuration.
    It composes all subsection settings into a single unified interface.

    Attributes:
        reddit: Reddit API configuration.
        data: Data directory paths configuration.
        market: Market data tickers and pairs configuration.

    Example:
        >>> settings = Settings()
        >>> print(settings.market.equity_tickers)
        ['AAPL', 'JNJ', 'JPM', 'XOM']
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    reddit: RedditSettings = RedditSettings()
    data: DataSettings = DataSettings()
    market: MarketSettings = MarketSettings()
