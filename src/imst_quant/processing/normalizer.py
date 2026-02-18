"""Text normalization."""

import re
from typing import List, NamedTuple


class NormalizedText(NamedTuple):
    raw: str
    cleaned: str
    urls_removed: int
    emojis: List[str]


class TextNormalizer:
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    CASHTAG_PATTERN = re.compile(r"\$([a-zA-Z]{1,5})\b")

    def normalize(self, text: str) -> NormalizedText:
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
        try:
            import emoji
            return [c for c in text if c in emoji.EMOJI_DATA]
        except (ImportError, AttributeError):
            return []
