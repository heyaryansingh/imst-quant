"""MinHash-based near-duplicate detection."""

from typing import Optional, Tuple

from datasketch import MinHash, MinHashLSH


class Deduplicator:
    """Detect near-duplicate text via MinHash LSH."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def get_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for word in text.lower().split():
            m.update(word.encode("utf-8"))
        return m

    def is_duplicate(self, post_id: str, text: str) -> Tuple[bool, Optional[str]]:
        mh = self.get_minhash(text)
        result = self.lsh.query(mh)
        if result:
            return True, result[0]
        self.lsh.insert(post_id, mh)
        return False, None
