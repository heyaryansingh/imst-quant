"""Checkpoint manager for incremental crawling."""

import json
from datetime import datetime, timezone
from pathlib import Path


class CheckpointManager:
    """Manage crawling checkpoints for incremental ingestion.

    Per RESEARCH Pattern 2 - checkpoint-based incremental crawling.
    """

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = Path(checkpoint_file)
        self._checkpoints = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(self._checkpoints, f, indent=2)

    def get_last_timestamp(self, subreddit: str) -> float:
        """Get last processed timestamp for subreddit."""
        return self._checkpoints.get(subreddit, {}).get("last_created_utc", 0)

    def update(self, subreddit: str, created_utc: float) -> None:
        """Update checkpoint for subreddit."""
        if subreddit not in self._checkpoints:
            self._checkpoints[subreddit] = {}
        self._checkpoints[subreddit]["last_created_utc"] = created_utc
        self._checkpoints[subreddit]["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
