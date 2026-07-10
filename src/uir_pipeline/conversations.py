"""conversations -- persistent chat threads for the console's Chats panel.

Backs the MonadLabs console's Chats panel. Each user owns a list of
conversation threads; each thread is an append-only log of messages. A
message the user types is a ``user`` message; when it starts with the
``gemini:`` trigger the console sends the remainder to the grounded chat
model and stores the reply as an ``assistant`` message (with the citations
it was given). Plain messages are just notes in the thread.

Storage mirrors :mod:`uir_pipeline.auth`: the same SQLite database file, a
fresh connection per call (console scale is a handful of users), WAL mode,
and ownership carried by ``user_id`` on every row. Every read and write is
scoped to the calling user, so one account can never see or mutate
another's threads -- the route layer relies on that, not on its own checks.

This module stores and retrieves; it never calls a model. The web layer
owns the ``gemini:`` routing and the retrieval/answer step, so this file
has no dependency on chat, embeddings, or Weaviate and stays cheap to
import and to test.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)


#: Longest title / message we store. Titles are truncated for the list view;
#: messages are capped so a single paste cannot bloat the database.
MAX_TITLE_LEN: Final[int] = 80
MAX_MESSAGE_LEN: Final[int] = 8000

_DEFAULT_TITLE: Final[str] = "New conversation"

_VALID_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant"})


_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    title      TEXT NOT NULL DEFAULT 'New conversation',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT,
    grounded        INTEGER,
    created_at      REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON conversation_messages (conversation_id, id);
"""


def _preview(text: str) -> str:
    """A single-line, length-capped snippet for the conversation list."""
    flat = " ".join((text or "").split())
    return flat[:120]


class ConversationStore:
    """SQLite-backed conversation + message store, one connection per call.

    Shares the console database file with :class:`~uir_pipeline.auth.UserStore`;
    each store creates only its own tables (``IF NOT EXISTS``), so construction
    order does not matter.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # ON DELETE CASCADE only fires when foreign keys are enabled, and the
        # pragma is per-connection. Without it, deleting a conversation would
        # orphan its messages.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- conversations --------------------------------------------------

    def create(self, user_id: int, title: str = _DEFAULT_TITLE) -> dict[str, Any]:
        now = time.time()
        title = (title or "").strip()[:MAX_TITLE_LEN] or _DEFAULT_TITLE
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(user_id), title, now, now),
            )
            cid = int(cur.lastrowid)  # type: ignore[arg-type]
        return {
            "id": cid,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "preview": "",
            "message_count": 0,
        }

    def list_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Conversations for ``user_id``, newest activity first, each with a
        preview drawn from its most recent message."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT content FROM conversation_messages m
                         WHERE m.conversation_id = c.id
                         ORDER BY m.id DESC LIMIT 1) AS last_content,
                       (SELECT role FROM conversation_messages m
                         WHERE m.conversation_id = c.id
                         ORDER BY m.id DESC LIMIT 1) AS last_role,
                       (SELECT COUNT(*) FROM conversation_messages m
                         WHERE m.conversation_id = c.id) AS message_count
                  FROM conversations c
                 WHERE c.user_id = ?
                 ORDER BY c.updated_at DESC, c.id DESC
                """,
                (int(user_id),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": int(r["id"]),
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "preview": _preview(r["last_content"] or ""),
                "last_role": r["last_role"],
                "message_count": int(r["message_count"]),
            })
        return out

    def get(self, user_id: int, conversation_id: int) -> dict[str, Any] | None:
        """Return the conversation iff it belongs to ``user_id``, else None.

        Ownership is enforced in the query, so a caller can never fetch a
        thread they do not own by guessing its id.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE id = ? AND user_id = ?",
                (int(conversation_id), int(user_id)),
            ).fetchone()
        return dict(row) if row else None

    def rename(self, user_id: int, conversation_id: int, title: str) -> bool:
        title = (title or "").strip()[:MAX_TITLE_LEN] or _DEFAULT_TITLE
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                (title, int(conversation_id), int(user_id)),
            )
            return cur.rowcount > 0

    def delete(self, user_id: int, conversation_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (int(conversation_id), int(user_id)),
            )
            return cur.rowcount > 0

    # -- messages -------------------------------------------------------

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, citations, grounded, created_at "
                "FROM conversation_messages WHERE conversation_id = ? ORDER BY id",
                (int(conversation_id),),
            ).fetchall()
        return [self._message_row(r) for r in rows]

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        grounded: bool | None = None,
    ) -> dict[str, Any]:
        """Append a message and bump the conversation's ``updated_at``.

        Raises ``ValueError`` on an unknown role. ``content`` is capped at
        :data:`MAX_MESSAGE_LEN`; ``citations`` is JSON-encoded.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")
        content = (content or "")[:MAX_MESSAGE_LEN]
        now = time.time()
        cites_json = json.dumps(citations) if citations else None
        grounded_int = None if grounded is None else int(bool(grounded))
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversation_messages "
                "(conversation_id, role, content, citations, grounded, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (int(conversation_id), role, content, cites_json, grounded_int, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, int(conversation_id)),
            )
            mid = int(cur.lastrowid)  # type: ignore[arg-type]
        return {
            "id": mid,
            "role": role,
            "content": content,
            "citations": citations or [],
            "grounded": grounded,
            "created_at": now,
        }

    def autotitle_if_default(self, conversation_id: int, text: str) -> None:
        """Set the title from ``text`` only while it is still the default.

        Lets a fresh thread name itself from its first real message without
        ever clobbering a title the user set deliberately.
        """
        candidate = " ".join((text or "").split())[:MAX_TITLE_LEN].strip()
        if not candidate:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND title = ?",
                (candidate, int(conversation_id), _DEFAULT_TITLE),
            )

    @staticmethod
    def _message_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(r["id"]),
            "role": r["role"],
            "content": r["content"],
            "citations": json.loads(r["citations"]) if r["citations"] else [],
            "grounded": None if r["grounded"] is None else bool(r["grounded"]),
            "created_at": r["created_at"],
        }


__all__ = [
    "ConversationStore",
    "MAX_MESSAGE_LEN",
    "MAX_TITLE_LEN",
]
