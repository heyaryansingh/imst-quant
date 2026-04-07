"""Text normalization utilities for social media financial posts.

This module provides text preprocessing for financial sentiment analysis,
including URL removal, cashtag standardization, and emoji extraction.

Classes:
    NormalizedText: Named tuple containing normalized text and metadata.
    TextNormalizer: Stateless text normalizer with configurable patterns.

Example:
    >>> normalizer = TextNormalizer()
    >>> result = normalizer.normalize("Check out $aapl! https://example.com 🚀")
    >>> print(result.cleaned)
    'Check out $AAPL! [URL] 🚀'
"""

import re
from typing import List, NamedTuple, Pattern


class NormalizedText(NamedTuple):
    """Container for normalized text with extraction metadata.

    Attributes:
        raw: Original unmodified text.
        cleaned: Normalized text with URLs replaced and cashtags uppercased.
        urls_removed: Count of URLs that were replaced with [URL] placeholder.
        emojis: List of emoji characters extracted from the text.
    """

    raw: str
    cleaned: str
    urls_removed: int
    emojis: List[str]


class TextNormalizer:
    """Text normalizer for financial social media posts.

    Normalizes text by:
    - Replacing URLs with [URL] placeholder
    - Uppercasing cashtags (e.g., $aapl -> $AAPL)
    - Collapsing whitespace
    - Extracting emojis for separate analysis

    Attributes:
        URL_PATTERN: Regex pattern matching HTTP(S) URLs.
        WHITESPACE_PATTERN: Regex pattern matching consecutive whitespace.
        CASHTAG_PATTERN: Regex pattern matching stock ticker cashtags.
    """

    URL_PATTERN: Pattern[str] = re.compile(r"https?://\S+|www\.\S+")
    WHITESPACE_PATTERN: Pattern[str] = re.compile(r"\s+")
    CASHTAG_PATTERN: Pattern[str] = re.compile(r"\$([a-zA-Z]{1,5})\b")

    def normalize(self, text: str) -> NormalizedText:
        """Normalize text for sentiment analysis.

        Args:
            text: Raw input text from social media post.

        Returns:
            NormalizedText with cleaned text and metadata.

        Example:
            >>> normalizer = TextNormalizer()
            >>> result = normalizer.normalize("Buy $tsla now! https://link.com")
            >>> result.cleaned
            'Buy $TSLA now! [URL]'
        """
        raw = text
        urls = self.URL_PATTERN.findall(text)
        cleaned = self.URL_PATTERN.sub(" [URL] ", text)
        cleaned = self.CASHTAG_PATTERN.sub(
            lambda m: f"${m.group(1).upper()}", cleaned
        )
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        emojis = self._extract_emojis(text)
        return NormalizedText(
            raw=raw,
            cleaned=cleaned,
            urls_removed=len(urls),
            emojis=emojis,
        )

    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emoji characters from text.

        Uses the emoji library if available; returns empty list otherwise.

        Args:
            text: Text potentially containing emoji characters.

        Returns:
            List of individual emoji characters found in the text.
        """
        try:
            import emoji
            return [c for c in text if c in emoji.EMOJI_DATA]
        except (ImportError, AttributeError):
            return []
