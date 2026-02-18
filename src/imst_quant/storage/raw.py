"""Raw storage with causality-preserving timestamps."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def store_reddit_post(submission, output_dir: Path) -> None:
    """Store Reddit submission with causality timestamps.

    Per RESEARCH Pattern 1 - causality-preserving raw storage.
    Every record has both created_utc (content time) and retrieved_at (ingestion time).

    Args:
        submission: PRAW Submission object
        output_dir: Base directory for reddit data (will write to output_dir/subreddit/YYYY-MM-DD/)
    """
    author_str = str(submission.author) if submission.author else "[deleted]"
    author_id = hashlib.sha256(author_str.encode()).hexdigest()

    record = {
        "id": submission.id,
        "author_id": author_id,
        "created_utc": submission.created_utc,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "subreddit": submission.subreddit.display_name,
        "author": author_str,
        "title": getattr(submission, "title", None) or "",
        "selftext": getattr(submission, "selftext", None) or "",
        "score": submission.score,
        "num_comments": getattr(submission, "num_comments", 0) or 0,
        "url": getattr(submission, "url", None) or "",
        "permalink": getattr(submission, "permalink", None) or "",
    }

    date_str = datetime.fromtimestamp(
        submission.created_utc, timezone.utc
    ).strftime("%Y-%m-%d")
    subreddit_name = submission.subreddit.display_name
    output_path = output_dir / subreddit_name / date_str / f"{submission.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
