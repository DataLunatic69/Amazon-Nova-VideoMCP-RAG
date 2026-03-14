"""
Conversation memory backed by Pixeltable.

Each agent instance owns one Memory object, which persists the conversation
history in a local Pixeltable table.  On `reset_memory()` the table is
dropped and recreated, giving a clean slate for a new session.

Why Pixeltable for memory?
──────────────────────────
We already depend on Pixeltable for the video indexes.  Using it for memory
too avoids adding a second persistence dependency (e.g. SQLite or Redis) and
gives us query/filter capabilities for free if we want to retrieve specific
turns later.
"""

import uuid
from datetime import datetime
from typing import List

import pixeltable as pxt
from loguru import logger

from video_rag.agent.models import MemoryRecord

logger = logger.bind(name="Memory")


class Memory:
    """Pixeltable-backed sliding-window conversation memory."""

    _TABLE_SCHEMA = {
        "message_id": pxt.String,
        "role":       pxt.String,
        "content":    pxt.String,
        "timestamp":  pxt.Timestamp,
    }

    def __init__(self, session_id: str) -> None:
        """
        Args:
            session_id: Unique identifier for this conversation session.
                        Used as the Pixeltable directory name so multiple
                        concurrent sessions don't collide.
        """
        self.session_id = session_id
        self._dir = f"mem_{session_id}"
        self._table_name = f"{self._dir}.history"

        pxt.create_dir(self._dir, if_exists="replace_force")
        self._table = pxt.create_table(
            self._table_name,
            self._TABLE_SCHEMA,
            if_exists="ignore",
        )
        logger.debug(f"Memory initialised for session '{session_id}'.")

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """Append one message to the history."""
        self._table.insert([{
            "message_id": uuid.uuid4().hex,
            "role":       role,
            "content":    content,
            "timestamp":  datetime.utcnow(),
        }])

    def add_pair(self, user_message: str, assistant_message: str) -> None:
        """Append a user + assistant turn in one batch insert."""
        now = datetime.utcnow()
        self._table.insert([
            {
                "message_id": uuid.uuid4().hex,
                "role":       "user",
                "content":    user_message,
                "timestamp":  now,
            },
            {
                "message_id": uuid.uuid4().hex,
                "role":       "assistant",
                "content":    assistant_message,
                "timestamp":  now,
            },
        ])

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_all(self) -> List[MemoryRecord]:
        rows = self._table.order_by(self._table.timestamp, asc=True).collect()
        return [MemoryRecord(message_id=r["message_id"], role=r["role"], content=r["content"]) for r in rows]

    def get_latest(self, n: int) -> List[MemoryRecord]:
        """Return the *n* most recent messages, oldest-first."""
        all_records = self.get_all()
        return all_records[-n:] if len(all_records) > n else all_records

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop and recreate the memory table, clearing all history."""
        pxt.drop_dir(self._dir, if_not_exists="ignore", force=True)
        pxt.create_dir(self._dir, if_exists="replace_force")
        self._table = pxt.create_table(
            self._table_name,
            self._TABLE_SCHEMA,
            if_exists="ignore",
        )
        logger.info(f"Memory reset for session '{self.session_id}'.")