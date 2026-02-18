"""Pydantic Settings for IMST-Quant configuration."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedditSettings(BaseSettings):
    """Reddit API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REDDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    client_id: str = ""
    client_secret: SecretStr = SecretStr("")
    user_agent: str = "IMST-Quant/1.0"


class DataSettings(BaseSettings):
    """Data paths configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
    )

    raw_dir: Path = Path("data/raw")
    bronze_dir: Path = Path("data/bronze")


class MarketSettings(BaseSettings):
    """Market data configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MARKET_",
        env_file=".env",
    )

    equity_tickers: list[str] = ["AAPL", "JNJ", "JPM", "XOM"]
    crypto_pairs: list[str] = ["BTC/USDT", "ETH/USDT"]


class Settings(BaseSettings):
    """Root settings composing all config sections."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    reddit: RedditSettings = RedditSettings()
    data: DataSettings = DataSettings()
    market: MarketSettings = MarketSettings()
