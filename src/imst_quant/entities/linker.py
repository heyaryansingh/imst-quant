"""Entity linker pipeline: extraction + aliases + disambiguation."""

import re
from typing import List

from .aliases import load_aliases
from .cashtag import extract_cashtags
from .disambiguator import EntityDisambiguator, EntityLink


class EntityLinker:
    """Combine cashtag extraction, alias resolution, and disambiguation."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self._disambiguator = EntityDisambiguator(
            confidence_threshold=confidence_threshold
        )
        self._ticker_dict, self._company_aliases, self._crypto_aliases = (
            load_aliases()
        )
        self._word_pattern = re.compile(r"[a-zA-Z0-9]+")

    def _resolve_aliases(self, text: str) -> List[str]:
        """Find candidate tickers from company/crypto aliases in text."""
        text_lower = text.lower()
        words = self._word_pattern.findall(text_lower)
        candidates = set()
        for w in words:
            if w in self._company_aliases:
                candidates.update(self._company_aliases[w])
            if w in self._crypto_aliases:
                candidates.update(self._crypto_aliases[w])
        return list(candidates)

    def link_entities(self, text: str, subreddit: str = "") -> List[EntityLink]:
        """
        Extract and link entities from text.

        1. Extract cashtags
        2. Resolve aliases from normalized text
        3. Merge and dedupe candidates
        4. Disambiguate via embeddings
        5. Dedupe by asset_id, keep highest confidence
        """
        if not text or not isinstance(text, str):
            return []
        cashtags = extract_cashtags(text)
        alias_candidates = self._resolve_aliases(text)
        candidates = list(set(cashtags) | set(alias_candidates))
        candidates = [c for c in candidates if c in self._ticker_dict]
        if not candidates:
            return []
        links = self._disambiguator.disambiguate(
            text, candidates, subreddit, self.confidence_threshold
        )
        cashtag_set = {t.upper() for t in cashtags}
        for ct in cashtag_set:
            if ct in self._ticker_dict and not any(
                l.asset_id == ct for l in links
            ):
                links.append(
                    EntityLink(
                        asset_id=ct,
                        confidence=1.0,
                        method="cashtag",
                        matched_span=ct,
                    )
                )
        by_asset = {}
        for link in links:
            if link.asset_id not in by_asset or link.confidence > by_asset[link.asset_id].confidence:
                by_asset[link.asset_id] = link
        return list(by_asset.values())
