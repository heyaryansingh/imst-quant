"""Checkpoint manager for incremental crawling.

This module provides functionality to track crawling progress across sessions,
enabling incremental data ingestion without re-processing already fetched items.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manage crawling checkpoints for incremental ingestion.

    Per RESEARCH Pattern 2 - checkpoint-based incremental crawling.

    Attributes:
        checkpoint_file: Path to the JSON file storing checkpoint data.

    Example:
        >>> manager = CheckpointManager(Path("checkpoints.json"))
        >>> last_ts = manager.get_last_timestamp("wallstreetbets")
        >>> # ... crawl new posts after last_ts ...
        >>> manager.update("wallstreetbets", new_timestamp)
        >>> manager.save()
    """

    def __init__(self, checkpoint_file: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_file: Path to the JSON checkpoint file.
        """
        self.checkpoint_file = Path(checkpoint_file)
        self._checkpoints: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Persist checkpoints to disk.

        Creates parent directories if they don't exist.
        """
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
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
