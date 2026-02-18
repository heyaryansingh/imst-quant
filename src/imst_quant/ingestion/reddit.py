"""Reddit ingestion via PRAW with checkpoint-based incremental crawling."""

from pathlib import Path

import praw
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from imst_quant.config.settings import RedditSettings
from imst_quant.storage.raw import store_reddit_post
from imst_quant.utils.checkpoint import CheckpointManager

logger = structlog.get_logger()


def create_reddit_client(settings: RedditSettings) -> praw.Reddit:
    """Create authenticated Reddit client with built-in rate limiting."""
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
    """Ingest posts from subreddit with checkpointing.

    PRAW handles rate limiting automatically. Checkpoint updated every 100 posts.
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
    """CLI entry point for Reddit ingestion."""
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
