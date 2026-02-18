"""Cashtag extraction per PRD FR-ENT-01."""

import re
from typing import List

CASHTAG_PATTERN = re.compile(r"\$([A-Za-z]{1,5})\b")


def extract_cashtags(text: str) -> List[str]:
    """Extract $TICKER patterns from text, returning uppercase ticker symbols."""
    if not text or not isinstance(text, str):
        return []
    matches = CASHTAG_PATTERN.findall(text)
    return [m.upper() for m in matches]
