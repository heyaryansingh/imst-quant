"""Language detection (langdetect - fasttext has numpy 2.0 compat issues)."""

from typing import Tuple


class LanguageDetector:
    """Detect if text is English using langdetect."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def is_english(self, text: str) -> Tuple[bool, float]:
        """Return (is_english, confidence)."""
        if not text or not text.strip():
            return False, 0.0
        cleaned = text.replace("\n", " ").strip()
        if len(cleaned) < 2:
            return False, 0.0
        try:
            import langdetect
            langs = langdetect.detect_langs(cleaned)
            for L in langs:
                if L.lang == "en":
                    return (L.prob >= self.threshold), float(L.prob)
            return False, 0.0
        except Exception:
            return False, 0.0
