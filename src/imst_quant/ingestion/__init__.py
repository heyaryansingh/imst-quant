"""Data ingestion module for IMST-Quant trading platform.

This package provides data ingestion pipelines for various financial data sources
including cryptocurrency exchanges, traditional markets, and social media sentiment.

Submodules:
    crypto: Cryptocurrency price and volume data from exchanges (Binance, Coinbase, etc.)
    market: Traditional market data including stocks, ETFs, and indices.
    reddit: Social sentiment data from Reddit for market sentiment analysis.

Example:
    >>> from imst_quant.ingestion import crypto, reddit
    >>> btc_data = crypto.fetch_ohlcv('BTC/USDT', timeframe='1h')
    >>> sentiment = reddit.fetch_posts('wallstreetbets', limit=100)
"""
