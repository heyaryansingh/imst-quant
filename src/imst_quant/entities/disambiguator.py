"""Contextual entity disambiguation using embeddings per PRD FR-ENT-02."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .aliases import load_aliases

CRYPTO_SUBS = {"cryptocurrency", "bitcoin", "ethereum", "cryptomarkets"}


@dataclass
class EntityLink:
    """Single entity link with confidence and metadata."""

    asset_id: str
    confidence: float
    method: str  # "cashtag" | "alias" | "embedding"
    matched_span: Optional[str] = None


class EntityDisambiguator:
    """Disambiguate candidates using SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.7,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._asset_embeddings: dict = {}
        self._ticker_dict: dict = {}
        self._crypto_tickers: set = set()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        ticker_dict, _, crypto_aliases = load_aliases()
        self._ticker_dict = ticker_dict
        self._crypto_tickers = {
            t for aliases in crypto_aliases.values() for t in aliases
        }
        descs = []
        tickers = []
        for ticker, info in ticker_dict.items():
            name = info.get("name", "")
            sector = info.get("sector", "")
            descs.append(f"{ticker} {name} {sector}".strip())
            tickers.append(ticker)
        if descs:
            embs = self._model.encode(descs)
            for t, e in zip(tickers, embs):
                self._asset_embeddings[t] = e

    def _subreddit_matches(self, subreddit: str, ticker: str) -> bool:
        """Return True if ticker is appropriate for subreddit context."""
        sub_lower = subreddit.lower() if subreddit else ""
        if sub_lower in CRYPTO_SUBS:
            return ticker in self._crypto_tickers
        return True

    def disambiguate(
        self,
        text: str,
        candidates: List[str],
        subreddit: str = "",
        confidence_threshold: Optional[float] = None,
    ) -> List[EntityLink]:
        """Disambiguate candidates against text using cosine similarity."""
        if not text or not candidates:
            return []
        thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        self._ensure_loaded()
        text_emb = self._model.encode([text])[0]
        links = []
        for cand in candidates:
            cand_upper = cand.upper()
            if cand_upper not in self._asset_embeddings:
                continue
            if not self._subreddit_matches(subreddit, cand_upper):
                continue
            emb = self._asset_embeddings[cand_upper]
            sim = float(np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb)))
            if sim >= thresh:
                links.append(
                    EntityLink(
                        asset_id=cand_upper,
                        confidence=sim,
                        method="embedding",
                        matched_span=cand,
                    )
                )
        return links
