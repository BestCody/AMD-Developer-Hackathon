"""auth -- user accounts, password hashing, and login throttling.

Backs the MonadLabs console's login/signup screens. Deliberately small:
a single SQLite table, ``werkzeug.security`` for hashing, and an
in-process throttle on failed logins.

Security posture (read this before deploying anywhere real):
    * Passwords are hashed with Werkzeug's default (scrypt, n=32768).
      Never stored or logged in plaintext.
    * ``verify_user`` runs a hash comparison even when the email is
      unknown, so response timing does not reveal whether an account
      exists. Callers must also return an identical error message for
      "no such user" and "wrong password".
    * The failed-login throttle is **per-process and in-memory**. It is
      reset by a restart and is not shared across workers. It raises the
      cost of online guessing; it is not a substitute for a real rate
      limiter at the edge.
    * There is no password-reset flow. Resetting a password requires an
      email transport this project does not have. See ``README``.

Transport caveat: ``web.py`` binds ``0.0.0.0`` over plain HTTP by
default. Passwords and session cookies cross the LAN in cleartext.
Auth here protects against casual access by other users of the box, not
against anyone who can sniff the network. Put it behind TLS before it
leaves a trusted network.
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Policy constants
# ----------------------------------------------------------------------------

#: NIST SP 800-63B floor. We do not impose composition rules (no "must
#: contain a symbol") because they demonstrably push users toward weaker,
#: more predictable passwords.
MIN_PASSWORD_LEN: Final[int] = 8
MAX_PASSWORD_LEN: Final[int] = 1024  # guard against scrypt DoS on huge inputs

#: Failed-login throttle: N failures inside WINDOW locks for LOCKOUT.
_MAX_FAILURES: Final[int] = 5
_WINDOW_SECONDS: Final[float] = 900.0   # 15 min
_LOCKOUT_SECONDS: Final[float] = 900.0  # 15 min


class AuthError(Exception):
    """User-facing auth failure. The message is safe to show a client."""


class RateLimited(AuthError):
    """Too many failed attempts for this (email, ip) pair."""


# ----------------------------------------------------------------------------
# Timing-equalisation dummy hash
# ----------------------------------------------------------------------------
# Computed once at import against a throwaway secret. ``verify_user``
# checks the submitted password against this when the email is unknown so
# the unknown-email path costs the same wall-clock as the wrong-password
# path. Without it, a fast 401 leaks "this address has no account".
_DUMMY_HASH: Final[str] = generate_password_hash(secrets.token_hex(32))


# ----------------------------------------------------------------------------
# Secret key
# ----------------------------------------------------------------------------

def resolve_secret_key(data_dir: Path) -> bytes:
    """Return the Flask session-signing key.

    Order of preference:
        1. ``SECRET_KEY`` env var (what you should use in production).
        2. ``<data_dir>/.secret_key``, generated on first run.

    A generated key is persisted so sessions survive a restart. If it
    were regenerated per boot, every user would be silently logged out
    on deploy. The file is created 0600; on Windows the mode is
    advisory, so keep ``data/`` off network shares.
    """
    env = (os.environ.get("SECRET_KEY") or "").strip()
    if env:
        return env.encode("utf-8")

    key_path = data_dir / ".secret_key"
    if key_path.is_file():
        raw = key_path.read_text(encoding="utf-8").strip()
        if raw:
            return raw.encode("utf-8")

    generated = secrets.token_hex(32)
    data_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_text(generated, encoding="utf-8")
    try:
        key_path.chmod(0o600)
    except OSError:  # pragma: no cover -- Windows / exotic filesystems
        pass
    logger.warning(
        "SECRET_KEY not set; generated one at %s. Set SECRET_KEY in the "
        "environment for any real deployment.",
        key_path,
    )
    return generated.encode("utf-8")


# ----------------------------------------------------------------------------
# Login throttle
# ----------------------------------------------------------------------------

class _Throttle:
    """In-memory failed-login counter keyed by (email, client-ip)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> (failure_count, first_failure_ts, locked_until_ts)
        self._state: dict[tuple[str, str], tuple[int, float, float]] = {}

    def check(self, email: str, ip: str) -> None:
        """Raise :class:`RateLimited` if this pair is currently locked."""
        key = (email, ip)
        now = time.time()
        with self._lock:
            entry = self._state.get(key)
            if entry is None:
                return
            _count, _first, locked_until = entry
            if now < locked_until:
                raise RateLimited(
                    "Too many failed attempts. Try again in "
                    f"{int((locked_until - now) / 60) + 1} minutes."
                )

    def record_failure(self, email: str, ip: str) -> None:
        key = (email, ip)
        now = time.time()
        with self._lock:
            count, first, locked_until = self._state.get(key, (0, now, 0.0))
            # Roll the window if the last burst has aged out.
            if now - first > _WINDOW_SECONDS:
                count, first = 0, now
            count += 1
            if count >= _MAX_FAILURES:
                locked_until = now + _LOCKOUT_SECONDS
                logger.warning(
                    "login throttle engaged for %s from %s (%d failures)",
                    email, ip, count,
                )
            self._state[key] = (count, first, locked_until)

    def record_success(self, email: str, ip: str) -> None:
        with self._lock:
            self._state.pop((email, ip), None)


throttle = _Throttle()


# ----------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);
"""


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


class UserStore:
    """SQLite-backed user table.

    One connection per call. At console scale (a handful of users, a
    login every few minutes) the connect cost is irrelevant and it
    sidesteps every ``check_same_thread`` hazard that comes with sharing
    a connection across Flask's worker threads.
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
        return conn

    # -- writes ---------------------------------------------------------

    def create_user(self, email: str, password: str, name: str = "") -> dict[str, Any]:
        """Insert a user. Raises :class:`AuthError` on bad input / duplicate."""
        email = normalize_email(email)
        if "@" not in email or len(email) < 3:
            raise AuthError("Enter a valid email address.")
        if len(password) < MIN_PASSWORD_LEN:
            raise AuthError(
                f"Password must be at least {MIN_PASSWORD_LEN} characters."
            )
        if len(password) > MAX_PASSWORD_LEN:
            raise AuthError("Password is too long.")

        pw_hash = generate_password_hash(password)
        now = time.time()
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (email, name, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (email, (name or "").strip()[:120], pw_hash, now),
                )
                user_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            # UNIQUE violation. This *does* disclose that the address is
            # taken -- unavoidable for a self-serve signup form, and the
            # standard tradeoff. Login stays non-disclosing.
            raise AuthError("An account with that email already exists.") from exc

        logger.info("created user id=%d email=%s", user_id, email)
        return {"id": user_id, "email": email, "name": name.strip()}

    def set_password(self, email: str, password: str) -> None:
        """Replace a user's password. Raises :class:`AuthError` if unknown.

        There is no self-serve reset flow: that needs a mail transport to
        prove control of the address, and we have none. Emailing nothing and
        calling it a reset would be worse than admitting the gap. This is the
        operator path -- see ``python -m uir_pipeline.auth``.

        Unlike :meth:`verify_user`, this *does* disclose whether the address
        exists. That is fine: the caller already has filesystem access to the
        password database.
        """
        email = normalize_email(email)
        if len(password) < MIN_PASSWORD_LEN:
            raise AuthError(
                f"Password must be at least {MIN_PASSWORD_LEN} characters."
            )
        if len(password) > MAX_PASSWORD_LEN:
            raise AuthError("Password is too long.")

        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (generate_password_hash(password), email),
            )
            if cur.rowcount == 0:
                raise AuthError(f"No account with email {email!r}.")
        logger.info("password reset for email=%s", email)

    # -- reads ----------------------------------------------------------

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, email, name, created_at FROM users ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, name FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def verify_user(self, email: str, password: str, *, ip: str = "-") -> dict[str, Any]:
        """Return the user dict, or raise :class:`AuthError`.

        The same message is raised for "no such account" and "wrong
        password", and a hash comparison runs in both branches, so
        neither the response body nor its latency distinguishes them.
        """
        email = normalize_email(email)
        throttle.check(email, ip)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, name, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()

        if row is None:
            # Burn equivalent CPU so the timing matches the found-user path.
            check_password_hash(_DUMMY_HASH, password)
            throttle.record_failure(email, ip)
            raise AuthError("Incorrect email or password.")

        if not check_password_hash(row["password_hash"], password):
            throttle.record_failure(email, ip)
            raise AuthError("Incorrect email or password.")

        throttle.record_success(email, ip)
        return {"id": int(row["id"]), "email": row["email"], "name": row["name"]}


#: Where `web.create_app` puts the user database when `data_dir` is default.
DEFAULT_DB_PATH = Path("data") / "monadlabs.db"


def _main(argv: list[str] | None = None) -> int:
    """Operator CLI: list accounts and reset a forgotten password.

    The console has no self-serve password reset, because a reset link has to
    be delivered to a mailbox we can't reach -- there is no mail transport
    configured. Rather than pretend, the recovery path is explicitly manual
    and requires filesystem access to the database::

        python -m uir_pipeline.auth list
        python -m uir_pipeline.auth reset alice@example.com

    The password is read from a no-echo prompt, or from stdin when piped. It
    is never taken from argv: arguments land in shell history and are visible
    to every other process via `ps`.
    """
    import argparse
    import getpass

    parser = argparse.ArgumentParser(prog="python -m uir_pipeline.auth")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help=f"path to the user database (default: {DEFAULT_DB_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list accounts")
    reset = sub.add_parser("reset", help="set a new password for an account")
    reset.add_argument("email")

    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"no user database at {args.db}", file=sys.stderr)
        return 1
    store = UserStore(args.db)

    if args.command == "list":
        users = store.list_users()
        if not users:
            print("no accounts")
        for u in users:
            print(f"{u['id']:>4}  {u['email']}")
        return 0

    if sys.stdin.isatty():
        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Repeat: "):
            print("passwords did not match", file=sys.stderr)
            return 1
    else:
        password = sys.stdin.readline().rstrip("\n")

    try:
        store.set_password(args.email, password)
    except AuthError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"password updated for {normalize_email(args.email)}")
    print("Any existing session cookie remains valid; rotate SECRET_KEY to "
          "invalidate all sessions.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "AuthError",
    "MIN_PASSWORD_LEN",
    "RateLimited",
    "UserStore",
    "normalize_email",
    "resolve_secret_key",
    "throttle",
]
