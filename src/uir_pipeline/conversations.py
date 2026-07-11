"""conversations -- multi-user chat threads for the console's Chats panel.

A conversation is a thread between two people, each identified by email. You
start one by entering someone's address (they need not have an account yet --
they see the thread the moment they sign up with that email). Both members
see the same messages.

Within a thread, a message that starts with ``@fireworks`` is a command: the
console answers its remainder from the *sender's* own converted documents and
posts the reply into the shared thread, so both members see the question and
the grounded answer. Anything else is an ordinary message between the two
people.

Access is defined by membership, not ownership: every read and write is
scoped to conversations whose member list contains the caller's email, so a
user can only ever see or post to threads they are part of. Storage mirrors
:mod:`uir_pipeline.auth` -- the same SQLite file, a fresh connection per call,
WAL, and ``ON DELETE CASCADE`` so leaving the last seat drops the thread.

This module stores and retrieves; it never calls a model. The web layer owns
the ``@fireworks`` routing and the retrieval/answer step.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

from uir_pipeline.auth import normalize_email

logger = logging.getLogger(__name__)


#: Longest message we store; a single paste cannot bloat the database.
MAX_MESSAGE_LEN: Final[int] = 8000

_VALID_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant"})


class ConversationError(Exception):
    """User-facing conversation failure; the message is safe to show a client."""


_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_members (
    conversation_id INTEGER NOT NULL,
    email           TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (conversation_id, email),
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_members_email
    ON conversation_members (email);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    sender_email    TEXT COLLATE NOCASE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT,
    grounded        INTEGER,
    tool_steps      TEXT,
    created_at      REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON conversation_messages (conversation_id, id);
"""


def _preview(text: str) -> str:
    return " ".join((text or "").split())[:120]


class ConversationStore:
    """SQLite-backed conversation store, one connection per call.

    Shares the console database file with :class:`~uir_pipeline.auth.UserStore`;
    each store creates only its own tables (``IF NOT EXISTS``).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._migrate_legacy(conn)
            conn.executescript(_SCHEMA)
            self._migrate_add_tool_steps(conn)

    @staticmethod
    def _migrate_add_tool_steps(conn: sqlite3.Connection) -> None:
        """Add the ``tool_steps`` column to pre-existing conversation_messages.

        Fresh databases get the column from ``_SCHEMA``; databases created
        before the agent tool-calling feature need an ``ALTER TABLE``. The
        column is nullable, so old rows simply read as ``tool_steps = []``.
        """
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(conversation_messages)")}
        if "tool_steps" not in cols:
            conn.execute("ALTER TABLE conversation_messages ADD COLUMN tool_steps TEXT")

    @staticmethod
    def _migrate_legacy(conn: sqlite3.Connection) -> None:
        """Drop the first-generation chat tables if present.

        An earlier version modelled conversations as single-owner notes: a
        ``conversations`` table with a ``user_id NOT NULL`` column and no
        ``conversation_members``. ``CREATE TABLE IF NOT EXISTS`` would leave
        that incompatible table in place, so every insert here would fail the
        old NOT NULL. The two models don't map onto each other (notes-to-self
        vs a thread between two people), so there is nothing to migrate --
        drop the legacy chat tables and let the new schema recreate them. The
        users table is untouched.
        """
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "user_id" in cols:
            logger.warning(
                "dropping legacy single-owner conversation tables; the chat "
                "model changed to multi-user membership and the two are not "
                "convertible"
            )
            conn.execute("DROP TABLE IF EXISTS conversation_messages")
            conn.execute("DROP TABLE IF EXISTS conversation_members")
            conn.execute("DROP TABLE IF EXISTS conversations")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # CASCADE only fires with foreign keys on, and the pragma is
        # per-connection: without it, deleting a conversation orphans its
        # members and messages.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- conversations --------------------------------------------------

    def create_with(self, creator_email: str, peer_email: str) -> tuple[dict[str, Any], bool]:
        """Return the 1:1 thread between these two, creating it if new.

        Returns ``(conversation, created)``. Reuses an existing thread with
        exactly these two members rather than piling up duplicates. Raises
        :class:`ConversationError` on a missing/invalid peer or self-chat.
        """
        creator = normalize_email(creator_email)
        peer = normalize_email(peer_email)
        if "@" not in peer or len(peer) < 3:
            raise ConversationError("Enter a valid email address.")
        if peer == creator:
            raise ConversationError("You can't start a conversation with yourself.")

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT m1.conversation_id AS cid
                  FROM conversation_members m1
                  JOIN conversation_members m2
                    ON m1.conversation_id = m2.conversation_id
                 WHERE m1.email = ? AND m2.email = ?
                   AND (SELECT COUNT(*) FROM conversation_members m3
                         WHERE m3.conversation_id = m1.conversation_id) = 2
                 LIMIT 1
                """,
                (creator, peer),
            ).fetchone()
            if existing is not None:
                cid = int(existing["cid"])
                return self._conversation_dict(conn, cid, creator), False

            now = time.time()
            cur = conn.execute(
                "INSERT INTO conversations (created_at, updated_at) VALUES (?, ?)",
                (now, now),
            )
            cid = int(cur.lastrowid)  # type: ignore[arg-type]
            conn.executemany(
                "INSERT INTO conversation_members (conversation_id, email) VALUES (?, ?)",
                [(cid, creator), (cid, peer)],
            )
            return self._conversation_dict(conn, cid, creator), True

    def list_for_email(self, email: str) -> list[dict[str, Any]]:
        """Threads ``email`` is a member of, newest activity first."""
        email = normalize_email(email)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.created_at, c.updated_at,
                       (SELECT email FROM conversation_members m
                         WHERE m.conversation_id = c.id AND m.email <> ?
                         LIMIT 1) AS peer_email,
                       (SELECT content FROM conversation_messages msg
                         WHERE msg.conversation_id = c.id
                         ORDER BY msg.id DESC LIMIT 1) AS last_content,
                       (SELECT role FROM conversation_messages msg
                         WHERE msg.conversation_id = c.id
                         ORDER BY msg.id DESC LIMIT 1) AS last_role,
                       (SELECT COUNT(*) FROM conversation_messages msg
                         WHERE msg.conversation_id = c.id) AS message_count
                  FROM conversations c
                  JOIN conversation_members me
                    ON me.conversation_id = c.id AND me.email = ?
                 ORDER BY c.updated_at DESC, c.id DESC
                """,
                (email, email),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": int(r["id"]),
                "peer_email": r["peer_email"] or "",
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "preview": _preview(r["last_content"] or ""),
                "last_role": r["last_role"],
                "message_count": int(r["message_count"]),
            })
        return out

    def get_for_email(self, email: str, conversation_id: int) -> dict[str, Any] | None:
        """Return the thread iff ``email`` is a member, else ``None``.

        Membership is the access check, so a non-member can never fetch a
        thread by guessing its id.
        """
        email = normalize_email(email)
        with self._connect() as conn:
            member = conn.execute(
                "SELECT 1 FROM conversation_members WHERE conversation_id = ? AND email = ?",
                (int(conversation_id), email),
            ).fetchone()
            if member is None:
                return None
            return self._conversation_dict(conn, int(conversation_id), email)

    def leave(self, email: str, conversation_id: int) -> bool:
        """Remove ``email`` from the thread; drop the thread if now empty.

        Returns ``True`` if the caller was a member. Leaving does not delete
        the other person's copy until the last member is gone.
        """
        email = normalize_email(email)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversation_members WHERE conversation_id = ? AND email = ?",
                (int(conversation_id), email),
            )
            if cur.rowcount == 0:
                return False
            remaining = conn.execute(
                "SELECT COUNT(*) AS n FROM conversation_members WHERE conversation_id = ?",
                (int(conversation_id),),
            ).fetchone()["n"]
            if remaining == 0:
                conn.execute("DELETE FROM conversations WHERE id = ?", (int(conversation_id),))
            return True

    # -- messages -------------------------------------------------------

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, sender_email, role, content, citations, grounded, tool_steps, created_at "
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
        sender_email: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        grounded: bool | None = None,
        tool_steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append a message and bump the thread's ``updated_at``.

        ``sender_email`` is the author for a ``user`` message and ``None`` for
        an ``assistant`` (fireworks) reply. ``tool_steps`` records the agent's
        tool calls for a fireworks reply (``None`` for ordinary notes). Raises
        ``ValueError`` on a bad role or a ``user`` message with no sender.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")
        if role == "user" and not sender_email:
            raise ValueError("a 'user' message requires a sender_email")
        sender = normalize_email(sender_email) if sender_email else None
        content = (content or "")[:MAX_MESSAGE_LEN]
        now = time.time()
        cites_json = json.dumps(citations) if citations else None
        steps_json = json.dumps(tool_steps) if tool_steps else None
        grounded_int = None if grounded is None else int(bool(grounded))
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversation_messages "
                "(conversation_id, sender_email, role, content, citations, grounded, tool_steps, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (int(conversation_id), sender, role, content, cites_json, grounded_int, steps_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, int(conversation_id)),
            )
            mid = int(cur.lastrowid)  # type: ignore[arg-type]
        return {
            "id": mid,
            "sender_email": sender,
            "role": role,
            "content": content,
            "citations": citations or [],
            "grounded": grounded,
            "tool_steps": tool_steps or [],
            "created_at": now,
        }

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _conversation_dict(conn: sqlite3.Connection, cid: int, viewer: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT id, created_at, updated_at FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        members = [
            r["email"] for r in conn.execute(
                "SELECT email FROM conversation_members WHERE conversation_id = ? ORDER BY email",
                (cid,),
            ).fetchall()
        ]
        peers = [m for m in members if m != normalize_email(viewer)]
        return {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "members": members,
            "peer_email": peers[0] if peers else "",
        }

    @staticmethod
    def _message_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(r["id"]),
            "sender_email": r["sender_email"],
            "role": r["role"],
            "content": r["content"],
            "citations": json.loads(r["citations"]) if r["citations"] else [],
            "grounded": None if r["grounded"] is None else bool(r["grounded"]),
            "tool_steps": json.loads(r["tool_steps"]) if r["tool_steps"] else [],
            "created_at": r["created_at"],
        }


__all__ = [
    "ConversationError",
    "ConversationStore",
    "MAX_MESSAGE_LEN",
]
