"""Reddit ingestion via PRAW with checkpoint-based incremental crawling.

This module provides utilities for ingesting Reddit posts from configured
subreddits using PRAW (Python Reddit API Wrapper). It supports:
- Incremental crawling with checkpoint persistence
- Automatic rate limiting via PRAW
- Retry logic for transient network failures
- Structured logging with structlog

Example:
    CLI usage::

        python -m imst_quant.ingestion.reddit --limit 500

    Programmatic usage::

        from imst_quant.ingestion.reddit import ingest_subreddit
        from imst_quant.config.settings import Settings

        settings = Settings()
        reddit = create_reddit_client(settings.reddit)
        count = ingest_subreddit(reddit, "wallstreetbets", output_dir, checkpoint_mgr)
"""

from pathlib import Path

import praw
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from imst_quant.config.settings import RedditSettings
from imst_quant.storage.raw import store_reddit_post
from imst_quant.utils.checkpoint import CheckpointManager

logger = structlog.get_logger()


def create_reddit_client(settings: RedditSettings) -> praw.Reddit:
    """Create authenticated Reddit client with built-in rate limiting.

    Args:
        settings: Reddit API credentials and configuration.

    Returns:
        Authenticated praw.Reddit instance with rate limiting enabled.

    Raises:
        praw.exceptions.ResponseException: If authentication fails.
    """
    return praw.Reddit(
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        user_agent=settings.user_agent,
        ratelimit_seconds=300,
    )


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def ingest_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    output_dir: Path,
    checkpoint_mgr: CheckpointManager,
    limit: int = 1000,
) -> int:
    """Ingest posts from a subreddit with checkpoint-based incremental crawling.

    Fetches new posts from the specified subreddit, storing them as JSON files.
    Uses checkpointing to track progress and avoid re-fetching posts on restart.
    PRAW handles Reddit API rate limiting automatically.

    Args:
        reddit: Authenticated praw.Reddit client instance.
        subreddit_name: Name of the subreddit (without r/ prefix).
        output_dir: Directory to store raw JSON post files.
        checkpoint_mgr: Manager for tracking last ingested timestamp.
        limit: Maximum number of posts to fetch (default: 1000).

    Returns:
        Number of new posts ingested.

    Raises:
        ConnectionError: If network connection fails (retried up to 3 times).
        TimeoutError: If request times out (retried up to 3 times).
    """
    subreddit = reddit.subreddit(subreddit_name)
    last_ts = checkpoint_mgr.get_last_timestamp(subreddit_name)

    logger.info(
        "Starting ingestion",
        subreddit=subreddit_name,
        limit=limit,
        last_timestamp=last_ts,
    )

    count = 0
    newest_ts = last_ts

    for submission in subreddit.new(limit=limit):
        if submission.created_utc <= last_ts:
            continue

        store_reddit_post(submission, output_dir)
        newest_ts = max(newest_ts, submission.created_utc)
        count += 1

        if count % 100 == 0:
            checkpoint_mgr.update(subreddit_name, newest_ts)
            checkpoint_mgr.save()
            logger.info("Progress", subreddit=subreddit_name, count=count)

    if count > 0:
        checkpoint_mgr.update(subreddit_name, newest_ts)
        checkpoint_mgr.save()

    logger.info("Ingestion complete", subreddit=subreddit_name, count=count)
    return count


def main() -> None:
    """CLI entry point for Reddit ingestion.

    Reads subreddit configuration from config/subreddits.yaml and ingests
    posts from all configured subreddits. Requires REDDIT_CLIENT_ID and
    REDDIT_CLIENT_SECRET environment variables.

    Raises:
        SystemExit: If credentials are missing or config file not found.
    """
    import argparse
    import sys
    import yaml

    from imst_quant.config.settings import Settings

    parser = argparse.ArgumentParser(
        description="Ingest Reddit posts from configured subreddits"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max posts per subreddit (default: 100)",
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.reddit.client_id or not settings.reddit.client_secret.get_secret_value():
        print(
            "Error: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env\n"
            "Create a Reddit app at https://www.reddit.com/prefs/apps"
        )
        sys.exit(1)

    config_path = Path("config/subreddits.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    subreddits = (
        config.get("equity_subreddits", [])
        + config.get("crypto_subreddits", [])
    )

    raw_reddit_dir = Path(settings.data.raw_dir) / "reddit"
    checkpoint_file = raw_reddit_dir / ".checkpoints.json"
    checkpoint_mgr = CheckpointManager(checkpoint_file)

    reddit = create_reddit_client(settings.reddit)

    total = 0
    for sub in subreddits:
        try:
            n = ingest_subreddit(
                reddit, sub, raw_reddit_dir, checkpoint_mgr, limit=args.limit
            )
            total += n
        except Exception as e:
            logger.exception("Ingestion failed", subreddit=sub, error=str(e))
            print(f"Warning: Failed to ingest r/{sub}: {e}")

    print(f"Done. Total posts ingested: {total}")
