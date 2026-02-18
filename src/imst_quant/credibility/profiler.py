"""Author profiles and bot probability (CRED-01, CRED-02)."""

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import polars as pl

CASHTAG_PATTERN = re.compile(r"\$[A-Za-z]{1,5}\b")


@dataclass
class AuthorProfile:
    """Author credibility profile."""

    author_id: str
    total_posts: int
    avg_posts_per_day: float
    posting_hours_entropy: float
    link_ratio: float
    cashtag_density: float
    subreddit_entropy: float
    bot_probability: float
    credibility_score: float


def _entropy(counts) -> float:
    import math
    s = sum(counts)
    if s <= 0:
        return 0.0
    p = [c / s for c in counts if c > 0]
    return -sum(x * math.log2(x) for x in p)


class AuthorProfiler:
    """Build author profiles from posts DataFrame."""

    def build_profile(self, author_id: str, df: pl.DataFrame) -> AuthorProfile:
        """Build profile for author from their posts."""
        author_posts = df.filter(pl.col("author_id").cast(str) == str(author_id))
        if len(author_posts) == 0:
            return self._default_profile(author_id)

        total = len(author_posts)

        if "created_utc" in author_posts.columns:
            hours = [
                datetime.fromtimestamp(t).hour
                for t in author_posts["created_utc"].to_list()
            ]
        else:
            hours = [12] * total

        hour_counts = [0] * 24
        for h in hours:
            if 0 <= h < 24:
                hour_counts[h] += 1
        posting_hours_entropy = _entropy(hour_counts)

        texts = []
        for c in ["cleaned_text", "selftext", "title"]:
            if c in author_posts.columns:
                texts = author_posts[c].fill_null("").to_list()
                break
        if not texts:
            texts = [""] * total

        link_posts = 0
        if "url" in author_posts.columns:
            urls = author_posts["url"].fill_null("").to_list()
            link_posts = sum(1 for u in urls if u and "reddit.com" not in str(u))
        link_ratio = link_posts / total if total else 0

        cashtags = sum(len(CASHTAG_PATTERN.findall(t)) for t in texts)
        cashtag_density = cashtags / total if total else 0

        subreddits = []
        if "subreddit" in author_posts.columns:
            subreddits = author_posts["subreddit"].to_list()
        sub_counts = list(Counter(subreddits).values()) if subreddits else [1]
        subreddit_entropy = _entropy(sub_counts)

        avg_len = sum(len(t) for t in texts) / total if total else 100

        bot_prob = self._bot_probability(
            posting_hours_entropy=posting_hours_entropy,
            link_ratio=link_ratio,
            avg_length=avg_len,
            cashtag_density=cashtag_density,
        )
        credibility = max(0.0, 1.0 - bot_prob)

        lookback = 90
        avg_posts_per_day = total / lookback

        return AuthorProfile(
            author_id=author_id,
            total_posts=total,
            avg_posts_per_day=avg_posts_per_day,
            posting_hours_entropy=posting_hours_entropy,
            link_ratio=link_ratio,
            cashtag_density=cashtag_density,
            subreddit_entropy=subreddit_entropy,
            bot_probability=bot_prob,
            credibility_score=credibility,
        )

    def _bot_probability(
        self,
        posting_hours_entropy: float,
        link_ratio: float,
        avg_length: float,
        cashtag_density: float,
    ) -> float:
        score = 0.0
        if posting_hours_entropy < 2.0:
            score += 0.3
        if link_ratio > 0.7:
            score += 0.3
        if avg_length < 50:
            score += 0.2
        if cashtag_density > 3:
            score += 0.2
        return min(score, 1.0)

    def _default_profile(self, author_id: str) -> AuthorProfile:
        return AuthorProfile(
            author_id=author_id,
            total_posts=0,
            avg_posts_per_day=0,
            posting_hours_entropy=3.0,
            link_ratio=0.1,
            cashtag_density=0.5,
            subreddit_entropy=2.0,
            bot_probability=0.3,
            credibility_score=0.5,
        )
