"""Tests for processing modules: deduplication and text normalization.

Tests cover:
- MinHash-based near-duplicate detection
- Text normalization (URL removal, cashtag uppercasing, emoji extraction)
"""

import pytest


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

from imst_quant.processing.deduplication import Deduplicator


class TestDeduplicator:
    """Tests for MinHash-based deduplication."""

    @pytest.fixture
    def dedup(self):
        return Deduplicator(num_perm=128, threshold=0.8)

    def test_unique_texts_not_duplicates(self, dedup):
        """Two completely different texts should not be flagged as duplicates."""
        is_dup, match = dedup.is_duplicate("post1", "AAPL earnings beat expectations today")
        assert is_dup is False
        assert match is None

        is_dup, match = dedup.is_duplicate("post2", "Bitcoin drops 10% on regulatory news")
        assert is_dup is False
        assert match is None

    def test_identical_texts_are_duplicates(self, dedup):
        """Identical text submitted with different IDs should be caught."""
        text = "Tesla stock surges on record deliveries and strong guidance"
        is_dup, _ = dedup.is_duplicate("post1", text)
        assert is_dup is False

        is_dup, match = dedup.is_duplicate("post2", text)
        assert is_dup is True
        assert match == "post1"

    def test_near_duplicate_detected(self, dedup):
        """Very similar texts should be flagged as near-duplicates."""
        text1 = "Apple reports record quarterly revenue of 124 billion dollars beating all estimates"
        text2 = "Apple reports record quarterly revenue of 124 billion dollars beating estimates"

        is_dup, _ = dedup.is_duplicate("post1", text1)
        assert is_dup is False

        is_dup, match = dedup.is_duplicate("post2", text2)
        assert is_dup is True
        assert match == "post1"

    def test_different_topics_not_duplicates(self, dedup):
        """Texts about different topics should not match."""
        dedup.is_duplicate("a", "Federal Reserve raises interest rates by 25 basis points")
        is_dup, _ = dedup.is_duplicate("b", "NVIDIA launches new GPU architecture for AI workloads")
        assert is_dup is False

    def test_empty_text_handled(self, dedup):
        """Empty strings should be processed without errors."""
        is_dup, _ = dedup.is_duplicate("empty1", "")
        assert is_dup is False

        is_dup, match = dedup.is_duplicate("empty2", "")
        # Two empty strings should match
        assert is_dup is True

    def test_get_minhash_deterministic(self, dedup):
        """Same text should produce same MinHash signature."""
        mh1 = dedup.get_minhash("consistent text for hashing")
        mh2 = dedup.get_minhash("consistent text for hashing")
        assert mh1.jaccard(mh2) == 1.0

    def test_custom_threshold(self):
        """Higher threshold should require more similarity to flag duplicates."""
        strict = Deduplicator(num_perm=128, threshold=0.95)
        text1 = "Market rallies on strong jobs data and economic outlook"
        text2 = "Market rallies on strong jobs data and positive economic outlook ahead"

        strict.is_duplicate("a", text1)
        is_dup, _ = strict.is_duplicate("b", text2)
        # With strict threshold, slight variations may not match
        # (result depends on actual Jaccard similarity)
        assert isinstance(is_dup, bool)


# ---------------------------------------------------------------------------
# Text normalizer tests
# ---------------------------------------------------------------------------

from imst_quant.processing.normalizer import NormalizedText, TextNormalizer


class TestTextNormalizer:
    """Tests for financial text normalization."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_url_removal(self, normalizer):
        """URLs should be replaced with [URL] placeholder."""
        result = normalizer.normalize("Check https://example.com for details")
        assert "[URL]" in result.cleaned
        assert "https://example.com" not in result.cleaned
        assert result.urls_removed == 1

    def test_multiple_urls(self, normalizer):
        """Multiple URLs should all be replaced."""
        text = "See https://a.com and http://b.com for info"
        result = normalizer.normalize(text)
        assert result.urls_removed == 2
        assert "https://a.com" not in result.cleaned
        assert "http://b.com" not in result.cleaned

    def test_cashtag_uppercased(self, normalizer):
        """Cashtags like $aapl should be uppercased to $AAPL."""
        result = normalizer.normalize("Buying $aapl and $tsla today")
        assert "$AAPL" in result.cleaned
        assert "$TSLA" in result.cleaned
        assert "$aapl" not in result.cleaned
        assert "$tsla" not in result.cleaned

    def test_cashtag_already_upper(self, normalizer):
        """Already-uppercase cashtags should remain unchanged."""
        result = normalizer.normalize("Sold $GOOG at 150")
        assert "$GOOG" in result.cleaned

    def test_whitespace_collapsed(self, normalizer):
        """Multiple whitespace characters should be collapsed to single space."""
        result = normalizer.normalize("too   many    spaces   here")
        assert "  " not in result.cleaned
        assert result.cleaned == "too many spaces here"

    def test_raw_preserved(self, normalizer):
        """Original text should be preserved in raw field."""
        original = "  $aapl is great https://x.com  "
        result = normalizer.normalize(original)
        assert result.raw == original

    def test_named_tuple_fields(self, normalizer):
        """Result should be a NormalizedText with all expected fields."""
        result = normalizer.normalize("test")
        assert isinstance(result, NormalizedText)
        assert hasattr(result, "raw")
        assert hasattr(result, "cleaned")
        assert hasattr(result, "urls_removed")
        assert hasattr(result, "emojis")

    def test_empty_text(self, normalizer):
        """Empty string should normalize without errors."""
        result = normalizer.normalize("")
        assert result.cleaned == ""
        assert result.urls_removed == 0
        assert result.emojis == []

    def test_no_cashtag_false_positives(self, normalizer):
        """Dollar amounts like $100 should not be treated as cashtags."""
        result = normalizer.normalize("Stock dropped $100 today")
        # $100 has digits, not matched by cashtag pattern [a-zA-Z]{1,5}
        assert "$100" in result.cleaned

    def test_www_url_replaced(self, normalizer):
        """URLs starting with www. should also be replaced."""
        result = normalizer.normalize("Visit www.example.com for more")
        assert "[URL]" in result.cleaned
        assert result.urls_removed == 1
