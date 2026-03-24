"""
Reasoning Manager for tracking and saving model reasoning (TextPart) content

This module tracks TextPart content as it streams from the model and saves it
to .data/reasoning/{hash}.txt files. The hash can be referenced in Message records
or diff JSON files for later retrieval.

Architecture:
- Uses ContextVar for async-safe isolation per request/execution context.
- Each streaming agent run gets its own ReasoningManager instance.
- At the end of a stream, finalize_and_save() hashes the content and persists it.
"""

import hashlib
import os
import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

# Base directory for reasoning files (relative to the backend working directory)
REASONING_DATA_DIR = os.path.join(".data", "reasoning")

# Context variable for reasoning manager - provides isolation per async execution context
_reasoning_manager_ctx: ContextVar[Optional["ReasoningManager"]] = ContextVar(
    "_reasoning_manager_ctx", default=None
)


class ReasoningManager:
    """Manages reasoning content (TextPart) for a single agent session/stream."""

    def __init__(self):
        self.content: str = ""
        self.reasoning_hash: Optional[str] = None
        logger.debug("ReasoningManager: Created new instance")

    def append_content(self, text: str) -> None:
        """Append text content (from TextPart / TextPartDelta) to the reasoning buffer."""
        if text:
            self.content += text

    def finalize_and_save(self) -> Optional[str]:
        """
        Finalize the reasoning content, generate a SHA-256 hash, and save to file.

        Returns:
            The reasoning hash if content was saved, None if no content was accumulated.
        """
        if not self.content:
            logger.debug("ReasoningManager: No content to save")
            return None

        # Generate hash from content
        content_bytes = self.content.encode("utf-8")
        self.reasoning_hash = hashlib.sha256(content_bytes).hexdigest()

        # Save to .data/reasoning/{hash}.txt
        try:
            os.makedirs(REASONING_DATA_DIR, exist_ok=True)

            filepath = os.path.join(REASONING_DATA_DIR, f"{self.reasoning_hash}.txt")

            # Skip writing if the same hash already exists (content is identical)
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.content)

            logger.info(
                "ReasoningManager: Saved reasoning content to %s "
                "(hash: %s, size: %d chars)",
                filepath,
                self.reasoning_hash,
                len(self.content),
            )
            return self.reasoning_hash
        except Exception as e:
            logger.error("ReasoningManager: Failed to save reasoning content: %s", e)
            return None

    def get_reasoning_hash(self) -> Optional[str]:
        """Get the current reasoning hash (None if not finalized)."""
        return self.reasoning_hash

    def get_content(self) -> str:
        """Get the accumulated reasoning content."""
        return self.content


# ---------------------------------------------------------------------------
# Module-level helpers for ContextVar-based usage
# ---------------------------------------------------------------------------

def get_reasoning_manager() -> ReasoningManager:
    """Get the current reasoning manager for this async execution context,
    creating a new one if needed."""
    manager = _reasoning_manager_ctx.get()
    if manager is None:
        logger.debug(
            "ReasoningManager: Creating new manager instance for this execution context"
        )
        manager = ReasoningManager()
        _reasoning_manager_ctx.set(manager)
    return manager


def reset_reasoning_manager() -> None:
    """Reset the reasoning manager for this async execution context.
    Call this at the START of a new streaming request to ensure a clean slate."""
    old_manager = _reasoning_manager_ctx.get()
    if old_manager:
        old_hash = old_manager.reasoning_hash
        old_size = len(old_manager.content)
        logger.debug(
            "ReasoningManager: Resetting manager (old hash: %s, old size: %d chars)",
            old_hash,
            old_size,
        )
    new_manager = ReasoningManager()
    _reasoning_manager_ctx.set(new_manager)
    logger.debug("ReasoningManager: Reset complete")


def finalize_reasoning() -> Optional[str]:
    """Convenience function: finalize the current context's reasoning manager and return the hash."""
    manager = _reasoning_manager_ctx.get()
    if manager is None:
        return None
    return manager.finalize_and_save()


def load_reasoning_content(reasoning_hash: str) -> Optional[str]:
    """Load reasoning content from disk by its hash.

    Returns:
        The reasoning text content, or None if the file does not exist.
    """
    filepath = os.path.join(REASONING_DATA_DIR, f"{reasoning_hash}.txt")
    if not os.path.exists(filepath):
        logger.warning("ReasoningManager: Reasoning file not found: %s", filepath)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("ReasoningManager: Failed to read reasoning file %s: %s", filepath, e)
        return None
