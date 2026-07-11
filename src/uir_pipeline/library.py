"""library -- persistent folders + jobs for the console's file browser.

The console's Upload tab is a Google-Drive-like workspace: documents live in
folders, and both folders and the job records (filename, status, output paths,
intent, folder membership) must survive a server restart. Until now jobs lived
only in an in-memory dict (see :mod:`uir_pipeline.web`), so a restart wiped the
whole library even though the uploaded files and UIR/UMR artefacts were still
on disk.

This module mirrors :mod:`uir_pipeline.conversations`: the same SQLite file as
the user store, a fresh connection per call, WAL journaling, ``foreign_keys=
ON``, and ``CREATE TABLE IF NOT EXISTS`` so several stores can each own their
own tables in one database. It stores and retrieves only; the web layer owns
the runner thread and the in-memory ``Job`` objects (write-through keeps the two
in sync -- see ``web._runner``).

Folders are intentionally **flat** (no ``parent_id`` nesting). They group a
handful of uploaded documents, not a filesystem tree; nesting would add
recursive CTEs and tree-drag UX for no real win. A ``folder_id`` of ``NULL`` on
a job means "lives at the All-files root". Deleting a folder ``SET NULL``s its
jobs rather than cascading, so a tidy-up never silently destroys documents.

This module never imports :mod:`uir_pipeline.web` (that would be a cycle:
``web`` imports this one inside ``create_app``). ``upsert_job`` duck-types the
passed object's attributes; the row-returning accessors hand plain dicts back
to the web layer, which rebuilds ``Job`` dataclasses from them.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)


#: Folders and jobs table set. ``jobs`` mirrors the persistent fields of the
#: ``Job`` dataclass in ``web.py``; dict-typed fields (``result``,
#: ``stage_meta``, ``intent_summary``) are stored as JSON TEXT.
_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS folders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    name       TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_folders_user ON folders (user_id);

CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    user_id          INTEGER NOT NULL,
    filename         TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'queued',
    progress_stage   TEXT NOT NULL DEFAULT 'queued',
    progress_percent INTEGER NOT NULL DEFAULT 0,
    submitted_at     REAL NOT NULL,
    finished_at      REAL,
    upload_path      TEXT,
    uir_path         TEXT,
    umr_path         TEXT,
    intent           TEXT,
    intent_uir_path  TEXT,
    intent_umr_path  TEXT,
    intent_summary   TEXT,
    result           TEXT,
    stage_meta       TEXT,
    error            TEXT,
    folder_id        INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (folder_id) REFERENCES folders (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_folder ON jobs (folder_id);
"""


#: Path-bearing fields on the ``Job`` dataclass that round-trip through TEXT.
_PATH_FIELDS: Final[tuple[str, ...]] = (
    "upload_path",
    "uir_path",
    "umr_path",
    "intent_uir_path",
    "intent_umr_path",
)

#: dict-typed fields stored as JSON TEXT.
_JSON_FIELDS: Final[tuple[str, ...]] = (
    "intent_summary",
    "result",
    "stage_meta",
)


def _name_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _path_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


class LibraryStore:
    """SQLite-backed folders + jobs store, one connection per call.

    Shares the console database file with :class:`~uir_pipeline.auth.UserStore`
    and :class:`~uir_pipeline.conversations.ConversationStore`; each store
    creates only its own tables (``IF NOT EXISTS``).
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
        # CASCADE / SET NULL only fire with foreign keys on, and the pragma is
        # per-connection: without it, deleting a folder orphans its jobs and
        # deleting a user orphans their folders.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- folders --------------------------------------------------------

    def create_folder(self, user_id: int, name: str) -> dict[str, Any]:
        """Insert a folder owned by ``user_id`` and return it (file_count 0)."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Folder name is required.")
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO folders (user_id, name, created_at) VALUES (?, ?, ?)",
                (int(user_id), name, now),
            )
            fid = int(cur.lastrowid)  # type: ignore[arg-type]
        return {"id": fid, "name": name, "created_at": now, "file_count": 0}

    def list_folders(self, user_id: int) -> list[dict[str, Any]]:
        """The user's folders, newest first, each annotated with file_count."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.name, f.created_at,
                       (SELECT COUNT(*) FROM jobs j
                         WHERE j.folder_id = f.id) AS file_count
                  FROM folders f
                 WHERE f.user_id = ?
                 ORDER BY f.created_at DESC, f.id DESC
                """,
                (int(user_id),),
            ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "name": r["name"],
                "created_at": r["created_at"],
                "file_count": int(r["file_count"]),
            }
            for r in rows
        ]

    def _owns_folder(self, conn: sqlite3.Connection, folder_id: int, user_id: int) -> bool:
        row = conn.execute(
            "SELECT 1 FROM folders WHERE id = ? AND user_id = ?",
            (int(folder_id), int(user_id)),
        ).fetchone()
        return row is not None

    def rename_folder(self, folder_id: int, user_id: int, name: str) -> bool:
        """Rename a folder the caller owns. ``False`` if not found / not owned."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Folder name is required.")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE folders SET name = ? WHERE id = ? AND user_id = ?",
                (name, int(folder_id), int(user_id)),
            )
            return cur.rowcount > 0

    def delete_folder(self, folder_id: int, user_id: int) -> bool:
        """Delete a folder the caller owns; its jobs fall back to the root.

        ``ON DELETE SET NULL`` does the job reassignment, so the jobs
        themselves are never touched here. Returns ``False`` if the folder
        was not found or not owned.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM folders WHERE id = ? AND user_id = ?",
                (int(folder_id), int(user_id)),
            )
            return cur.rowcount > 0

    # -- jobs -----------------------------------------------------------

    def upsert_job(self, job: Any) -> None:
        """Persist every durable field of a ``Job`` (INSERT OR REPLACE).

        ``job`` is duck-typed: the web layer passes its in-memory ``Job``
        dataclass. Path and dict fields are serialized to TEXT/JSON. The
        hot read path stays the in-memory dict; this is the write-through
        mirror that makes jobs survive a restart.
        """
        cols = (
            "job_id, user_id, filename, status, progress_stage, "
            "progress_percent, submitted_at, finished_at, "
            "upload_path, uir_path, umr_path, intent, "
            "intent_uir_path, intent_umr_path, "
            "intent_summary, result, stage_meta, error, folder_id"
        )
        placeholders = ", ".join("?" for _ in cols.split(", "))
        values = (
            job.job_id,
            job.user_id,
            job.filename,
            job.status,
            job.progress_stage,
            job.progress_percent,
            job.submitted_at,
            job.finished_at,
            _path_or_none(job.upload_path),
            _path_or_none(job.uir_path),
            _path_or_none(job.umr_path),
            job.intent,
            _path_or_none(job.intent_uir_path),
            _path_or_none(job.intent_umr_path),
            json.dumps(job.intent_summary) if job.intent_summary is not None else None,
            json.dumps(job.result) if job.result is not None else None,
            json.dumps(job.stage_meta) if job.stage_meta else None,
            job.error,
            job.folder_id,
        )
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO jobs ({cols}) VALUES ({placeholders})",
                values,
            )

    def _job_from_row(self, r: sqlite3.Row) -> dict[str, Any]:
        """Rebuild a plain dict the web layer turns into a ``Job`` dataclass."""
        out: dict[str, Any] = {
            "job_id": r["job_id"],
            "user_id": r["user_id"],
            "filename": r["filename"],
            "status": r["status"],
            "progress_stage": r["progress_stage"],
            "progress_percent": r["progress_percent"],
            "submitted_at": r["submitted_at"],
            "finished_at": r["finished_at"],
            "intent": r["intent"],
            "error": r["error"],
            "folder_id": r["folder_id"],
        }
        for f in _PATH_FIELDS:
            v = r[f]
            out[f] = Path(v) if v else None
        for f in _JSON_FIELDS:
            v = r[f]
            out[f] = json.loads(v) if v else ({} if f == "stage_meta" else None)
        return out

    def get_job_row(self, job_id: str, user_id: int) -> dict[str, Any] | None:
        """Return the job row iff ``user_id`` owns it, else ``None``."""
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND user_id = ?",
                (job_id, int(user_id)),
            ).fetchone()
        return self._job_from_row(r) if r is not None else None

    def list_job_rows(
        self, user_id: int, *, folder_id: int | None = None, only_running: bool = False
    ) -> list[dict[str, Any]]:
        """Job rows for ``user_id``, oldest-submitted first (matches /api/jobs).

        ``folder_id`` (non-null) restricts to one folder; ``None`` returns jobs
        across all folders and the root. ``only_running`` further restricts to
        ``queued``/``running`` jobs -- used on startup to find the orphans left
        behind by a restart (their runner thread is gone).
        """
        sql = "SELECT * FROM jobs WHERE user_id = ?"
        params: list[Any] = [int(user_id)]
        if only_running:
            sql += " AND status IN ('queued', 'running')"
        if folder_id is not None:
            sql += " AND folder_id = ?"
            params.append(int(folder_id))
        sql += " ORDER BY submitted_at ASC, job_id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._job_from_row(r) for r in rows]

    def list_all_job_rows(self, *, only_running: bool = False) -> list[dict[str, Any]]:
        """Every persisted job row across all users (for startup rehydration).

        ``only_running`` restricts to ``queued``/``running`` -- the orphans a
        restart leaves behind. Order is unspecified; the caller loads them
        into a dict keyed by ``job_id``.
        """
        sql = "SELECT * FROM jobs"
        if only_running:
            sql += " WHERE status IN ('queued', 'running')"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._job_from_row(r) for r in rows]

    def delete_job(self, job_id: str, user_id: int) -> bool:
        """Delete a job the caller owns. ``False`` if not found / not owned."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM jobs WHERE job_id = ? AND user_id = ?",
                (job_id, int(user_id)),
            )
            return cur.rowcount > 0

    def set_folder(self, job_id: str, user_id: int, folder_id: int | None) -> bool:
        """Move a job into a folder (``None`` = root). ``False`` if not owned.

        If ``folder_id`` is non-null it must belong to ``user_id``; otherwise
        the update is refused (returns ``False``) so a caller can't park a
        document in another user's folder.
        """
        with self._connect() as conn:
            if folder_id is not None and not self._owns_folder(conn, folder_id, user_id):
                return False
            cur = conn.execute(
                "UPDATE jobs SET folder_id = ? WHERE job_id = ? AND user_id = ?",
                (folder_id, job_id, int(user_id)),
            )
            return cur.rowcount > 0


__all__ = ["LibraryStore"]
