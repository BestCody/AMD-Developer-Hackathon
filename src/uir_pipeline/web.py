"""web -- HTTP front-end for the pipeline (the MonadLabs console).

Routes:
    GET  /                            -- the console SPA
    GET  /api/health                  -- readiness probe (public)

    POST /api/auth/signup             -- create an account, starts a session
    POST /api/auth/login              -- start a session
    POST /api/auth/logout             -- end the session
    GET  /api/auth/me                 -- current user, or 401

    POST /api/run                     -- accepts ``file=<doc>``, returns ``{job_id}``
    GET  /api/jobs                    -- the caller's own jobs
    GET  /api/status/<job_id>         -- returns ``{status, progress, ...}``
    GET  /api/result/<job_id>         -- the UIR JSON, inline
    GET  /api/umr/<job_id>            -- the UMR markdown companion
    GET  /api/download/<job_id>       -- streams the produced ``.uir.json``
    POST /api/chat                    -- grounded Q&A over the caller's documents

Architecture:
    * ``create_app(...)`` returns a Flask application. Tests use this
      factory with the production module's ``pipeline.run`` monkey-patched
      to a stub so the test suite never pulls in BGE / spaCy / pdfplumber.
    * Each upload is written to a per-app ``upload_dir`` and a job-id in a
      thread-safe dict tracks ``queued | running | done | error`` state with
      a progress percentage and the final ``PipelineResult``.
    * The runner thread catches all exceptions and records them in the job
      dict so the front-end can surface them without crashing the worker.

On auth (this file used to say "no auth; keep it that way"):
    Accounts were added deliberately, not by drift. Every job now carries
    the ``user_id`` that created it, and every job route checks ownership
    before serving bytes -- without that, a session cookie would gate the
    *UI* while any authenticated user could still read any other user's
    document by guessing a job id. A mismatch returns 404, not 403, so the
    job-id space cannot be probed for existence.

    Jobs remain in an in-memory dict: they do not survive a restart, and
    they are not shared across workers. Run this single-process. Users, by
    contrast, live in SQLite (see :mod:`uir_pipeline.auth`).
"""
from __future__ import annotations

import atexit
import functools
import importlib
import json
import logging
import multiprocessing
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)
# Routes return `(body, status)` tuples as well as bare Responses; that
# union is exactly what Flask calls a ResponseReturnValue.
from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean from the environment ('1', 'true', 'yes' are true)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------
# Pipeline execution: process isolation
# ----------------------------------------------------------------------------
#
# The orchestrator loads native code (Docling / pypdfium2 / torch). Under
# memory pressure Docling's C++ layer raises ``std::bad_alloc`` and can abort
# the process outright. A ``threading.Thread`` cannot survive that -- SIGSEGV
# is not a Python exception, so the runner's ``except Exception`` never fires
# and the whole Flask server dies with the job. Uploading a mid-sized PDF was
# enough to take the console down.
#
# So the orchestrator runs in a child process. A crash there costs one job:
# the parent sees a non-zero ``exitcode``, marks the job ``error``, and keeps
# serving. This is the default. ``execution="thread"`` restores the old
# in-process behaviour and exists for tests, which monkeypatch
# ``pipeline.run`` in the parent -- a patch a spawned child cannot inherit.

#: Import path of the orchestrator module. Overridable so a child process can
#: be pointed at a lightweight stub (tests) without importing the real,
#: heavyweight pipeline.
PIPELINE_MODULE_ENV = "UIR_PIPELINE_MODULE"
_DEFAULT_PIPELINE_MODULE = "uir_pipeline.pipeline"

#: Seconds to wait for a child that has stopped reporting before giving up.
_CHILD_JOIN_TIMEOUT = 15.0


class PipelineWorkerCrash(RuntimeError):
    """The pipeline child process died without reporting a result.

    Almost always a native crash (SIGSEGV / abort) inside Docling or a
    downstream C extension, or an OOM kill. Carries the exit code because
    that is the only forensic signal the parent gets.
    """


class PipelineWorkerError(RuntimeError):
    """The child raised a Python exception; its text is already formatted.

    ``str(exc)`` is the child's ``"TypeName: message"``. ``_runner`` re-raises
    nothing and copies this verbatim, so job errors read
    ``DoclingPartialConversion: ...`` rather than
    ``RuntimeError: DoclingPartialConversion: ...``.
    """


def _resolve_pipeline_module() -> Any:
    return importlib.import_module(
        os.environ.get(PIPELINE_MODULE_ENV) or _DEFAULT_PIPELINE_MODULE
    )


def _result_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "out_path": str(getattr(result, "out_path", "")),
        "uir_id": getattr(result, "uir_id", None),
        "chunk_count": int(getattr(result, "chunk_count", 0)),
        "entity_count": int(getattr(result, "entity_count", 0)),
        "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
    }
    # Omit rather than blank: the parent's ``AttributeError`` fallback
    # derives a sibling path, and Path("") silently means Path(".").
    umr = getattr(result, "umr_path", None)
    if umr:
        payload["umr_path"] = str(umr)
    return payload


def _pipeline_worker_loop(job_q: Any, progress_q: Any, result_q: Any) -> None:
    """Child-process entry point. Must stay importable at module scope.

    Windows uses the ``spawn`` start method, which re-imports this module in
    the child and unpickles the target by qualified name. A closure or a
    nested function would not survive that.

    The orchestrator is imported **once**, before the loop -- that import
    pulls in torch and Docling and costs ~8s. Jobs then arrive over
    ``job_q``, so only the first upload after a (re)spawn pays for it.

    Every message carries the job id. The parent discards this whole worker
    on a crash, so ids can't collide across workers; within one worker they
    guard against a late message from an abandoned job being read as the
    current one's result.
    """
    try:
        mod = _resolve_pipeline_module()
    except BaseException as exc:  # noqa: BLE001 -- a bad import must be reported, not hang
        result_q.put((None, "error", f"{type(exc).__name__}: {exc}"))
        return

    while True:
        message = job_q.get()
        if message[0] == "stop":
            return
        _, job_id, upload_path, output_dir, intent = message

        def _on_progress(stage: str, pct: int, _jid: str = job_id, **meta: Any) -> None:
            try:
                progress_q.put((_jid, str(stage), int(pct), dict(meta)))
            except Exception:  # noqa: BLE001 -- progress is advisory, never fatal
                pass

        try:
            result = mod.run(
                Path(upload_path),
                fast_path="docling",
                output_dir=Path(output_dir),
                skip_weaviate=True,
                with_embeddings=True,
                on_progress=_on_progress,
                intent=intent,
            )
            result_q.put((job_id, "ok", _result_payload(result)))
        except BaseException as exc:  # noqa: BLE001 -- one bad job must not end the worker
            result_q.put((job_id, "error", f"{type(exc).__name__}: {exc}"))


#: Environment the *child* reads at import/convert time. A spawned process
#: inherits the parent's environment once, at spawn; changing one of these in
#: a running server would otherwise have no effect until the worker happened
#: to die, which is a confusing thing to debug. We fingerprint them and
#: respawn on change.
_WORKER_ENV_PREFIXES: tuple[str, ...] = ("DOCLING_", "UIR_PIPELINE_MODULE")


def _worker_env_fingerprint() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if any(k.startswith(p) for p in _WORKER_ENV_PREFIXES)
    }


class _WarmWorker:
    """A single long-lived pipeline child process, respawned when it dies.

    Spawning per job cost ~8s of `import torch`/`import docling` before any
    work began -- more than the conversion itself on a small PDF. One
    persistent child pays that once per server lifetime.

    Crash isolation is unchanged: a native SIGSEGV inside Docling still kills
    only the child. The parent notices, fails that one job, discards the
    worker, and the next job spawns a replacement. That path is now rare
    (the backend that caused the crashes is fixed), which is precisely why
    it is safe to keep a worker alive across jobs.

    Jobs are **serialised** by ``_lock``. The previous code could run several
    conversions at once; each holds a Docling model set resident, and two
    concurrent conversions of a real PDF do not fit in memory on a typical
    dev box. Serialising is the honest behaviour, not a regression.

    Started lazily on the first job: an idle server should not hold ~1.5 GB
    of torch and Docling weights resident for uploads that never come.
    """

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._proc: Any = None
        self._job_q: Any = None
        self._progress_q: Any = None
        self._result_q: Any = None
        self._env: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------
    def _spawn(self) -> None:
        self._env = _worker_env_fingerprint()
        self._job_q = self._ctx.Queue()
        self._progress_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_pipeline_worker_loop,
            args=(self._job_q, self._progress_q, self._result_q),
            daemon=True,
        )
        self._proc.start()
        logger.info("pipeline worker started (pid %s)", self._proc.pid)

    def _discard(self) -> int | None:
        """Tear the worker down and return its exit code."""
        code = None
        if self._proc is not None:
            if self._proc.is_alive():
                # It is parked on job_q.get() and will never exit on its own;
                # without this the join below always burns its full timeout.
                try:
                    self._job_q.put(("stop",))
                except Exception:  # noqa: BLE001
                    pass
            self._proc.join(timeout=_CHILD_JOIN_TIMEOUT)
            if self._proc.is_alive():  # pragma: no cover -- child ignoring termination
                self._proc.kill()
                self._proc.join(timeout=5)
            code = self._proc.exitcode
        # The queues may hold half-written messages from the dead child; a
        # fresh worker gets fresh queues rather than inheriting that debris.
        # `close()` alone leaves the background feeder thread holding the
        # pipe write-end open; `join_thread()` is what actually releases it.
        # Without the join, every crash/respawn cycle leaked ~2 FDs per queue
        # until the parent itself hit EMFILE under multi-upload.
        for q in (self._job_q, self._progress_q, self._result_q):
            try:
                if q is not None:
                    q.close()
                    q.join_thread(timeout=2.0)
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                pass
        self._proc = self._job_q = self._progress_q = self._result_q = None
        return code

    def shutdown(self) -> None:
        with self._lock:
            self._discard()  # sends ("stop",) if the child is still alive

    # -- job submission ----------------------------------------------------
    def run(
        self,
        *,
        upload_path: Path,
        output_dir: Path,
        intent: str | None,
        on_progress: Any,
    ) -> SimpleNamespace:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                current = _worker_env_fingerprint()
                if current != self._env:
                    changed = sorted(
                        set(current) ^ set(self._env)
                        | {k for k in current.keys() & self._env.keys()
                           if current[k] != self._env[k]}
                    )
                    logger.info(
                        "worker environment changed (%s); respawning so the "
                        "child picks it up", ", ".join(changed),
                    )
                    self._discard()

            if self._proc is None or not self._proc.is_alive():
                if self._proc is not None:
                    self._discard()
                self._spawn()

            job_id = uuid.uuid4().hex
            self._job_q.put(
                ("job", job_id, str(upload_path), str(output_dir), intent)
            )
            return self._collect(job_id, on_progress)

    def _drain_progress(self, job_id: str, on_progress: Any) -> None:
        while True:
            try:
                jid, stage, pct, meta = self._progress_q.get_nowait()
            except (queue.Empty, OSError, EOFError, ValueError):
                return
            if jid == job_id:
                on_progress(stage, pct, **meta)

    def _collect(self, job_id: str, on_progress: Any) -> SimpleNamespace:
        outcome: tuple[str, Any] | None = None
        while True:
            self._drain_progress(job_id, on_progress)
            try:
                jid, kind, payload = self._result_q.get(timeout=0.1)
                # `jid is None` is the worker reporting a failed import.
                if jid in (job_id, None):
                    outcome = (kind, payload)
                    break
            except queue.Empty:
                pass
            except (OSError, EOFError, ValueError):
                # The queue died with the child; fall through to the liveness
                # check, which raises PipelineWorkerCrash with the exit code.
                break
            if not self._proc.is_alive():
                # The child exited. Give the queue a moment to surface a
                # result written just before exit, then decide.
                try:
                    jid, kind, payload = self._result_q.get(timeout=1.0)
                    if jid in (job_id, None):
                        outcome = (kind, payload)
                except (queue.Empty, OSError, EOFError, ValueError):
                    outcome = None
                break

        self._drain_progress(job_id, on_progress)

        if outcome is None:
            code = self._discard()
            raise PipelineWorkerCrash(
                f"pipeline worker died without reporting a result (exit code {code}). "
                "This is typically a native crash or an out-of-memory kill inside "
                "Docling. The server itself is unaffected."
            )

        kind, payload = outcome
        if kind == "error":
            if not self._proc.is_alive():
                self._discard()  # a failed import leaves no usable worker
            raise PipelineWorkerError(payload)
        return SimpleNamespace(**payload)


# ----------------------------------------------------------------------------
# Job state
# ----------------------------------------------------------------------------

# Status constants — the sequence the front-end animates through.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"


@dataclass
class Job:
    job_id: str
    #: The account that submitted this job. Every job route checks this
    #: against the session before serving anything. ``None`` only for
    #: jobs created by tests that bypass the auth layer.
    user_id: int | None = None
    #: Original filename as the browser reported it, sanitised to a bare
    #: basename. Shown in the console's document folder -- the on-disk
    #: name is ``<job_id><ext>`` and is not user-meaningful.
    filename: str = ""
    status: str = JOB_QUEUED
    progress_stage: str = "queued"
    progress_percent: int = 0
    submitted_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    upload_path: Path | None = None
    uir_path: Path | None = None
    result: dict[str, Any] | None = None
    stage_meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Intent-filter fields: when the user submits an optional ``intent``
    # form field on /api/run, the runner thread post-processes the full
    # UIR JSON down to matching chunks and stashes both paths + a
    # summary here. ``/api/result`` serves the filtered form so the
    # front-end shows only the chunks the agent needs to read.
    intent: str | None = None
    intent_uir_path: Path | None = None
    intent_summary: dict[str, Any] | None = None
    # Phase 17: companion UMR Markdown paths. Always populated after a
    # successful run. ``umr_path`` is the full-document view;
    # ``intent_umr_path`` is the intent-filtered subset when an intent
    # was submitted. The web UI's `/api/umr/<job_id>` endpoint serves
    # whichever of these two the user-facing view should display.
    umr_path: Path | None = None
    intent_umr_path: Path | None = None
    #: Folder membership for the file-browser's left tree. ``None`` means the
    #: document lives at the "All files" root. Set at upload time from the
    #: optional ``folder`` form field and mutated by ``PATCH /api/jobs/<id>``.
    #: Persisted alongside the rest of the job (see :mod:`uir_pipeline.library`)
    #: so the library survives a restart.
    folder_id: int | None = None

    def to_public(self) -> dict[str, Any]:
        """Shape we return to the front-end (excludes raw filesystem paths).

        ``stage_meta`` is filtered to a small allowlist before serializing:
        the orchestrator may push keys like ``error`` that contain Python
        exception strings (paths, library versions), which would leak to
        LAN viewers on a multi-user deploy. The full text stays in the
        local stderr log via ``pipeline.py``. We surface only scalar
        counters + booleans.
        """
        # Allowlist: stage_meta values we hand the front-end. Anything not
        # in this set is hidden before JSON serialization. Add a key here
        # only after a deliberate UX decision.
        _META_PUBLIC_KEYS = (
            "caption_records_total",
            "caption_records_with_text",
            "caption_records_empty",
            "dropped_entities",
            "dropped_relations",
            "entity_count",
            "relationship_count",
            "transcription_length",
            "language_detected",
            "duration_seconds",
            "speaker_count",
            "frame_count",
            "frame_descriptions",
        )
        public_meta = {
            k: v for k, v in (self.stage_meta or {}).items()
            if k in _META_PUBLIC_KEYS
        }
        # Intent-filter summary: surfaced only when an intent was provided.
        # Counters + keywords are safe to expose; we do not surface the
        # keyword-match breakdown or chunk-level diff (LAN UI doesn't need it).
        public_intent = None
        if self.intent_summary is not None:
            public_intent = {
                "query": self.intent,
                "matched_chunks": int(self.intent_summary.get("matched_chunks", 0)),
                "total_chunks": int(self.intent_summary.get("total_chunks", 0)),
                "keywords": self.intent_summary.get("keywords", []),
                "no_match_fallback": bool(self.intent_summary.get("no_match", False)),
            }
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.progress_stage,
            "percent": self.progress_percent,
            "submitted_at": self.submitted_at,
            "finished_at": self.finished_at,
            "stage_meta": public_meta,
            "intent": public_intent,
            "result": self.result,
            "error": self.error,
            "folder_id": self.folder_id,
        }


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------

def create_app(
    *,
    upload_dir: Path | None = None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    template_folder: Path | None = None,
    static_folder: Path | None = None,
    max_upload_mb: int = 64,
    execution: str | None = None,
) -> Flask:
    """Build a Flask application with isolated per-instance state.

    ``execution`` selects how the orchestrator is invoked:

    ``"process"`` (default)
        Run it in a child process so a native crash (Docling's
        ``std::bad_alloc`` -> SIGSEGV) kills one job, not the server.
    ``"thread"``
        Run it in-process, the historical behaviour. Tests use this because
        they monkeypatch ``pipeline.run`` in the parent, which a spawned
        child cannot inherit.

    Overridable via ``UIR_WEB_EXECUTION``.
    """
    # Raise the file-descriptor soft limit to the hard limit. The pipeline
    # child loads torch + Docling + BGE + PyMuPDF -- hundreds of FDs -- and
    # macOS's default soft cap (256) is low enough that a real conversion
    # exhausts it, crashing the child; the crash/respawn cycle then leaks
    # pipe FDs in this parent until werkzeug itself EMFILEs (500s on static
    # + JSX -> Babel can't load -> whitescreen, and polls fail). A `spawn`
    # child inherits these limits, so raising once here covers both. No-op
    # where the soft limit is already at the hard cap or `resource` is absent
    # (Windows). See error.tmp for the original EMFILE traceback.
    try:
        import resource as _resource
        _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        if _soft < _hard and _hard != _resource.RLIM_INFINITY:
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (_hard, _hard))
    except (ImportError, ValueError, OSError):  # pragma: no cover -- non-Unix / denied
        pass

    from uir_pipeline.auth import UserStore, resolve_secret_key
    from uir_pipeline.conversations import ConversationStore

    upload_dir = (Path(upload_dir) if upload_dir else Path("/tmp/uir_web_uploads")).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir = (Path(output_dir) if output_dir else Path("/tmp/uir_web_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Default templates/static fold under the package.
    _pkg_root = Path(__file__).resolve().parent
    _repo_root = _pkg_root.parent.parent
    data_dir = (Path(data_dir) if data_dir else (_repo_root / "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Distinct names: the parameters are `Path | None`, Flask wants `str`.
    template_dir = str(template_folder or (_repo_root / "templates"))
    static_dir = str(static_folder or (_repo_root / "static"))

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["UPLOAD_DIR"] = upload_dir
    app.config["OUTPUT_DIR"] = output_dir

    execution = (execution or os.environ.get("UIR_WEB_EXECUTION") or "process").lower()
    if execution not in ("process", "thread"):
        raise ValueError(f"execution must be 'process' or 'thread', got {execution!r}")
    app.config["EXECUTION"] = execution
    if execution == "thread":
        logger.warning(
            "pipeline runs in-process (execution='thread'): a native crash in "
            "Docling will take the whole server down with it."
        )
    # Per-app, not a module global: tests build several apps in one process and
    # must not share a worker (nor its UIR_PIPELINE_MODULE, read at spawn time).
    # Lazy -- no child exists until the first upload.
    worker = _WarmWorker() if execution == "process" else None
    app.config["PIPELINE_WORKER"] = worker
    # NB: no atexit hook here. The factory is called once per test, and an
    # atexit handler per app would (a) accumulate unboundedly and (b) fire at
    # interpreter shutdown, after pytest has closed the streams the worker's
    # logger writes to. The child is a daemon, so it dies with the parent
    # regardless; the server entrypoint registers a clean stop explicitly.

    # -------- session / auth --------------------------------------------
    app.secret_key = resolve_secret_key(data_dir)
    app.config.update(
        # Signed session cookie. HttpOnly keeps it away from any XSS that
        # slips through; SameSite=Lax is our primary CSRF defence, since a
        # cross-site POST will not carry the cookie. Secure must be on for
        # any HTTPS deploy -- it is off by default only because web.py's
        # default bind is plain HTTP on the LAN.
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_env_flag("SESSION_COOKIE_SECURE", default=False),
        PERMANENT_SESSION_LIFETIME=timedelta(days=14),
    )
    users = UserStore(data_dir / "monadlabs.db")
    app.config["USER_STORE"] = users
    # Chats-panel threads live in the same SQLite file, one table set each.
    conversations = ConversationStore(data_dir / "monadlabs.db")
    app.config["CONVERSATION_STORE"] = conversations
    # Folders + durable job records for the file-browser library. The jobs
    # table mirrors the persistent fields of the ``Job`` dataclass so the
    # in-memory registry can be rehydrated after a restart (see
    # :func:`_rehydrate_jobs`). Write-through keeps the two in sync.
    from uir_pipeline.library import LibraryStore
    library = LibraryStore(data_dir / "monadlabs.db")
    app.config["LIBRARY_STORE"] = library

    # Per-app job registry -- thread-safe by virtue of the GIL guarding dict
    # assignment + the lock below for read-modify-write on the Job itself.
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    # Exposed for tests that need to seed a completed job without driving a
    # full upload. It is the same dict the routes close over, not a copy.
    app.config["JOBS"] = jobs

    # -------- auth helpers ----------------------------------------------

    def _current_user() -> dict[str, Any] | None:
        uid = session.get("user_id")
        if uid is None:
            return None
        user = users.get_by_id(int(uid))
        if user is None:
            # Account deleted out from under a live cookie.
            session.clear()
            return None
        return user

    def login_required(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            user = _current_user()
            if user is None:
                abort(401, description="Sign in to continue.")
            request.user = user  # type: ignore[attr-defined]
            return fn(*args, **kwargs)
        return wrapper

    def _owned_job(job_id: str) -> Job:
        """Fetch a job the caller owns, or 404.

        Deliberately 404 (not 403) when the job exists but belongs to
        someone else: a 403 would confirm the id is real and turn the
        job-id space into an oracle.
        """
        uid = request.user["id"]  # type: ignore[attr-defined]
        job = jobs.get(job_id)
        if job is None or job.user_id != uid:
            abort(404, description="job not found")
        return job

    @app.before_request
    def _reject_cross_origin_writes() -> Response | None:
        """Defence-in-depth CSRF check on state-changing requests.

        ``SameSite=Lax`` already stops a cross-site form POST from
        carrying the session cookie. This adds a same-origin assertion so
        the protection does not rest on one browser behaviour alone.
        Requests with no Origin header (curl, same-origin GET) pass.
        """
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None
        origin = request.headers.get("Origin")
        if not origin:
            return None
        if urlparse(origin).netloc != request.host:
            logger.warning(
                "rejected cross-origin %s %s from Origin=%s",
                request.method, request.path, origin,
            )
            abort(403, description="cross-origin request rejected")
        return None

    def _persist(job: Job) -> None:
        """Write-through mirror of the in-memory Job into SQLite.

        Best-effort: the in-memory dict is the hot read path, so a DB hiccup
        must never kill the runner. The exception is logged and swallowed.
        """
        try:
            library.upsert_job(job)
        except Exception as exc:  # noqa: BLE001 -- persistence is best-effort
            logger.warning("job persist failed for %s: %s", job.job_id, exc)

    def _runner(job: Job) -> None:
        """Background thread body -- runs the pipeline and updates the Job."""
        try:
            _advance(job, JOB_RUNNING, "ingest", 5, lock=jobs_lock)
            _persist(job)

            def _on_progress(stage: str, pct: int, **meta: Any) -> None:
                """Forward orchestrator progress + optional stage meta to the Job.

                The orchestrator's :func:`uir_pipeline.pipeline.run` now passes
                metadata (caption-empty counters, dropped-entity counts, etc.)
                via ``**meta``. We accept and dispatch the same way the
                on_progress callback signature evolved in pipeline.py. The
                kwargs are absorbed at the job level so the front-end can
                pick them up in subsequent ``/api/status/<job_id>`` polls.
                """
                _advance(job, JOB_RUNNING, stage, pct, lock=jobs_lock)
                if meta:
                    with jobs_lock:
                        job.stage_meta = dict(meta)
                _persist(job)

            if job.upload_path is None:  # pragma: no cover -- set before dispatch
                raise RuntimeError(f"job {job.job_id} has no upload path")

            # Branch on the worker, not on `execution`: the worker is None
            # exactly when execution == "thread", and this way the two can
            # never disagree.
            #
            # The two branches return different types -- a SimpleNamespace
            # rebuilt from the child's payload, or a real PipelineResult. Only
            # the attributes read below are common to both.
            result: Any
            if worker is not None:
                # Isolated: a SIGSEGV inside Docling costs this job only.
                result = worker.run(
                    upload_path=job.upload_path,
                    output_dir=app.config["OUTPUT_DIR"],
                    intent=job.intent,
                    on_progress=_on_progress,
                )
            else:
                # Imported here, not at factory time: in "process" mode the
                # parent never needs the heavy orchestrator, and tests patch
                # ``pipeline.run`` on the module object before this resolves it.
                from uir_pipeline import pipeline as _pipeline_mod

                result = _pipeline_mod.run(
                    job.upload_path,
                    fast_path="docling",  # web pins Docling (single backend; docling failures propagate)
                    output_dir=app.config["OUTPUT_DIR"],
                    skip_weaviate=True,   # web UX: skip Weaviate by default
                    with_embeddings=True,
                    on_progress=_on_progress,
                    intent=job.intent,
                )
            _advance(job, JOB_RUNNING, "done", 100, lock=jobs_lock)
            with jobs_lock:
                job.uir_path = Path(result.out_path)
                # Companion UMR markdown is the agent-friendly view the
                # web UI surfaces by default (Phase 17). Population is
                # best-effort when dry_run=True (path only, no file).
                try:
                    job.umr_path = Path(result.umr_path)
                except AttributeError:
                    # older PipelineResult test stub without umr_path --
                    # derive from out_path sibling so the UI doesn't break.
                    job.umr_path = job.uir_path.with_suffix(".umr.md") \
                        if job.uir_path.suffix == ".json" \
                        else Path(str(job.uir_path) + ".umr.md")
                # Build the public dict by hand rather than ``dataclasses.asdict``
                # so we stay robust if :func:`pipeline.run` ever returns a
                # non-dataclass object (e.g. a stub in tests).
                job.result = {
                    "uir_id": getattr(result, "uir_id", None),
                    "out_path": str(getattr(result, "out_path", "")),
                    "chunk_count": int(getattr(result, "chunk_count", 0)),
                    "entity_count": int(getattr(result, "entity_count", 0)),
                    "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
                }
                # PLAN §17 §Multi-format follow-up: surface the resolved
                # source format + extraction route so the front-end can
                # render a "PDF via Docling", "PPTX via Native Walker",
                # etc. pill in the result bar without a second fetch.
                # The UIR JSON's top-level ``source`` field is the
                # source of truth (see :class:`uir_pipeline.uir_schema
                # .Source`); we read it here so the format label is
                # available the moment ``/api/status`` flips to ``done``.
                # Best-effort: a partial / malformed UIR must not
                # poison the job (the front-end will simply omit the
                # pill).
                try:
                    uir_doc = json.loads(job.uir_path.read_text(encoding="utf-8"))
                    source_meta = (uir_doc.get("source") or {}) \
                        if isinstance(uir_doc, dict) else {}
                    job.result["source_format"] = str(
                        source_meta.get("format") or "UNKNOWN"
                    )
                    job.result["source_route"] = str(
                        source_meta.get("route") or "unknown"
                    )
                except Exception as exc:  # noqa: BLE001 -- best-effort
                    logger.warning(
                        "could not read UIR source for job %s: %s",
                        job.job_id, exc,
                    )
                    job.result.setdefault("source_format", "UNKNOWN")
                    job.result.setdefault("source_route", "unknown")
                # Tier 1.5 + web: if an intent was supplied, post-process
                # the freshly-written UIR JSON so the front-end can fetch a
                # narrowed ``/api/result`` payload. ``intent_filter`` is a
                # pure function over the JSON on disk -- safe to call from
                # this thread (no shared state with the orchestrator).
                if job.intent:
                    try:
                        from uir_pipeline.intent_filter import (
                            filter_uirstream_by_intent,
                        )
                        summary = filter_uirstream_by_intent(
                            job.uir_path, job.intent,
                        )
                        job.intent_uir_path = Path(summary["out_path"])
                        job.intent_summary = summary
                        # Build the intent-filtered UMR companion file so
                        # the web UI can render the narrowed markdown
                        # view alongside the narrowed JSON view. Falls
                        # back to the full UMR if the build fails so the
                        # front-end never breaks on this post-process.
                        try:
                            from uir_pipeline.umr import build_umr as _build_umr
                            intent_umr_path = job.intent_uir_path.with_suffix(
                                ".md"
                            ) if job.intent_uir_path.suffix == ".json" \
                                else Path(str(job.intent_uir_path) + ".umr.md")
                            # Use ``intent_filter``-aware form so the
                            # rendered markdown reflects the matched
                            # subset rather than the full document.
                            intent_filter_arg = {
                                "intent": summary.get("intent"),
                                "keywords": summary.get("keywords", []),
                                "matches": summary.get("matches", []),
                            }
                            intent_umr_path.write_text(
                                _build_umr(
                                    json.loads(
                                        job.intent_uir_path.read_text(encoding="utf-8")
                                    ),
                                    intent_filter=intent_filter_arg,
                                ),
                                encoding="utf-8",
                            )
                            job.intent_umr_path = intent_umr_path
                        except Exception as exc:  # noqa: BLE001 -- best-effort
                            logger.warning(
                                "intent-umr build failed for job %s: %s",
                                job.job_id, exc,
                            )
                        # Augment result so the UI can show
                        # ``"X of Y chunks (intent: ...)"`` from a single poll.
                        job.result["intent_matched_chunks"] = int(
                            summary["matched_chunks"]
                        )
                        job.result["intent_total_chunks"] = int(
                            summary["total_chunks"]
                        )
                        job.result["intent_keywords"] = list(
                            summary["keywords"]
                        )
                        job.result["intent_no_match_fallback"] = bool(
                            summary["no_match"]
                        )
                        logger.info(
                            "intent-filter: %d/%d chunks matched keywords=%s",
                            summary["matched_chunks"], summary["total_chunks"],
                            summary["keywords"],
                        )
                    except Exception as exc:  # noqa: BLE001 -- post-process is best-effort
                        logger.warning(
                            "intent-filter failed for job %s: %s",
                            job.job_id, exc,
                        )
                job.status = JOB_DONE
                job.finished_at = time.time()
            _persist(job)
        except Exception as exc:  # noqa: BLE001 -- top-level worker guard
            logger.exception("web job %s failed", job.job_id)
            with jobs_lock:
                job.status = JOB_ERROR
                # A PipelineWorkerError already carries the child's
                # "TypeName: message"; re-prefixing would double it up.
                job.error = (
                    str(exc) if isinstance(exc, PipelineWorkerError)
                    else f"{type(exc).__name__}: {exc}"
                )
                job.finished_at = time.time()
            _persist(job)

    # -------- routes ----------------------------------------------------

    @app.get("/")
    def index() -> ResponseReturnValue:
        return render_template("console.html", max_upload_mb=max_upload_mb)

    @app.get("/api/health")
    def health() -> ResponseReturnValue:
        return jsonify({"ok": True, "upload_dir": str(upload_dir)})

    # -------- auth ------------------------------------------------------

    def _json_body() -> dict[str, Any]:
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else {}

    @app.post("/api/auth/signup")
    def api_signup() -> ResponseReturnValue:
        from uir_pipeline.auth import AuthError

        body = _json_body()
        try:
            user = users.create_user(
                body.get("email") or "",
                body.get("password") or "",
                body.get("name") or "",
            )
        except AuthError as exc:
            return jsonify({"error": str(exc)}), 400
        session.clear()
        session["user_id"] = user["id"]
        session.permanent = True
        return jsonify({"user": user})

    @app.post("/api/auth/login")
    def api_login() -> ResponseReturnValue:
        from uir_pipeline.auth import AuthError, RateLimited

        body = _json_body()
        ip = request.remote_addr or "-"
        try:
            user = users.verify_user(
                body.get("email") or "", body.get("password") or "", ip=ip,
            )
        except RateLimited as exc:
            return jsonify({"error": str(exc)}), 429
        except AuthError as exc:
            return jsonify({"error": str(exc)}), 401
        # Rotate the session id on privilege change to blunt session fixation.
        session.clear()
        session["user_id"] = user["id"]
        session.permanent = True
        return jsonify({"user": user})

    @app.post("/api/auth/logout")
    def api_logout() -> ResponseReturnValue:
        session.clear()
        return jsonify({"ok": True})

    @app.get("/api/auth/me")
    def api_me() -> ResponseReturnValue:
        user = _current_user()
        if user is None:
            return jsonify({"error": "not authenticated"}), 401
        return jsonify({"user": user})

    @app.get("/api/users/search")
    @login_required
    def api_users_search() -> ResponseReturnValue:
        """Autocomplete: registered users whose email starts with ``q``."""
        q = (request.args.get("q") or "").strip().lower()
        if len(q) < 2:
            return jsonify({"users": []})
        return jsonify({"users": users.search_by_email(q, limit=10)})

    # -------- jobs ------------------------------------------------------

    @app.get("/api/jobs")
    @login_required
    def api_jobs() -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        # Optional ``?folder_id=`` narrows to one folder. The literal ``""``
        # means "root only" (folder_id IS NULL); absent means "all of the
        # caller's jobs regardless of folder". The file-browser's left tree
        # uses this to populate the currently-selected folder's grid.
        folder_param = request.args.get("folder_id")
        with jobs_lock:
            mine_jobs = [j for j in jobs.values() if j.user_id == uid]
            if folder_param is not None:
                fid = int(folder_param) if folder_param != "" else None
                mine_jobs = [j for j in mine_jobs if j.folder_id == fid]
            mine = [j.to_public() for j in mine_jobs]
        mine.sort(key=lambda j: j["submitted_at"])
        return jsonify({"jobs": mine})

    # -------- folders / library ----------------------------------------

    @app.get("/api/folders")
    @login_required
    def api_folders_list() -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        return jsonify({"folders": library.list_folders(uid)})

    @app.post("/api/folders")
    @login_required
    def api_folders_create() -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        body = _json_body()
        name = (body.get("name") or "").strip()
        if not name:
            abort(400, description="folder name is required")
        try:
            folder = library.create_folder(uid, name)
        except ValueError as exc:
            abort(400, description=str(exc))
        return jsonify({"folder": folder}), 201

    @app.patch("/api/folders/<int:folder_id>")
    @login_required
    def api_folders_rename(folder_id: int) -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        body = _json_body()
        name = (body.get("name") or "").strip()
        if not name:
            abort(400, description="folder name is required")
        try:
            ok = library.rename_folder(folder_id, uid, name)
        except ValueError as exc:
            abort(400, description=str(exc))
        if not ok:
            abort(404, description="folder not found")
        return jsonify({"ok": True})

    @app.delete("/api/folders/<int:folder_id>")
    @login_required
    def api_folders_delete(folder_id: int) -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        if not library.delete_folder(folder_id, uid):
            abort(404, description="folder not found")
        # ``ON DELETE SET NULL`` already moved the folder's jobs to the root
        # in SQLite; mirror that in the in-memory registry so the next
        # ``/api/jobs`` reflects it without a restart.
        with jobs_lock:
            for j in jobs.values():
                if j.user_id == uid and j.folder_id == folder_id:
                    j.folder_id = None
        return jsonify({"ok": True})

    @app.patch("/api/jobs/<job_id>")
    @login_required
    def api_jobs_update(job_id: str) -> ResponseReturnValue:
        """Move a job into a folder (``folder_id``) or to the root (``null``)."""
        uid = request.user["id"]  # type: ignore[attr-defined]
        body = _json_body()
        raw = body.get("folder_id")
        # Accept null/missing -> root; otherwise coerce to int.
        folder_id: int | None
        if raw is None:
            folder_id = None
        else:
            try:
                folder_id = int(raw)
            except (TypeError, ValueError):
                abort(400, description="folder_id must be an integer or null")
        if not library.set_folder(job_id, uid, folder_id):
            # Either the job isn't the caller's, or the target folder isn't.
            # 404 (not 403) to avoid leaking which id is real.
            abort(404, description="job or folder not found")
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None and job.user_id == uid:
                job.folder_id = folder_id
        return jsonify({"ok": True})

    @app.delete("/api/jobs/<job_id>")
    @login_required
    def api_jobs_delete(job_id: str) -> ResponseReturnValue:
        """Permanently delete a job: on-disk artefacts + DB row + memory entry."""
        uid = request.user["id"]  # type: ignore[attr-defined]
        with jobs_lock:
            job = _owned_job(job_id)  # 404 if missing or not owner
            # Remove every artefact the pipeline wrote for this job.
            for attr in ("upload_path", "uir_path", "umr_path",
                         "intent_uir_path", "intent_umr_path"):
                p = getattr(job, attr, None)
                if p is not None:
                    try:
                        Path(p).unlink()
                    except OSError:
                        pass
            library.delete_job(job_id, uid)
            jobs.pop(job_id, None)
        return jsonify({"ok": True})

    @app.get("/api/thumb/<job_id>")
    @login_required
    def api_thumb(job_id: str) -> ResponseReturnValue:
        """A non-text preview tile for a finished job's source file.

        Images are served verbatim (the original is the preview). PDFs render
        page 1 to PNG via PyMuPDF. Video files extract a frame via ffmpeg.
        Everything else 404s so the front-end falls back to a filetype icon.
        Cached for an hour -- the source file is immutable once the job is done.
        """
        with jobs_lock:
            job = _owned_job(job_id)
            if job.status != JOB_DONE:
                abort(409, description="job not done yet")
            upload_path = job.upload_path
        if upload_path is None or not Path(upload_path).is_file():
            abort(404, description="source file not found")
        ext = Path(upload_path).suffix.lower()
        _IMG_MIME = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
            ".tiff": "image/tiff", ".tif": "image/tiff",
            ".avif": "image/avif", ".heic": "image/heic", ".heif": "image/heif",
        }
        if ext in _IMG_MIME:
            return send_file(upload_path, mimetype=_IMG_MIME[ext])
        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF -- already a dependency (see caption.py)
            except ImportError:
                abort(404, description="thumbnail renderer unavailable")
            try:
                doc = fitz.open(str(upload_path))
                if doc.page_count < 1:
                    doc.close()
                    abort(404, description="empty pdf")
                pix = doc[0].get_pixmap(dpi=144)
                img = pix.tobytes("png")
                doc.close()
            except Exception as exc:  # noqa: BLE001 -- render is best-effort
                logger.warning("thumb render failed for job %s: %s", job_id, exc)
                abort(404, description="thumbnail render failed")
            return Response(img, mimetype="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})
        _VIDEO_EXT = {
            ".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv", ".m4v",
        }
        if ext in _VIDEO_EXT:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    frame_path = Path(tmpdir) / "frame.png"
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(upload_path),
                            "-ss", "00:00:01", "-vframes", "1",
                            "-q:v", "2", str(frame_path),
                        ],
                        capture_output=True, timeout=30,
                    )
                    if result.returncode != 0 or not frame_path.is_file():
                        abort(404, description="video frame extraction failed")
                    img = frame_path.read_bytes()
                    return Response(img, mimetype="image/png",
                                    headers={"Cache-Control": "public, max-age=3600"})
            except Exception as exc:  # noqa: BLE001
                logger.warning("video thumb failed for job %s: %s", job_id, exc)
                abort(404, description="video thumbnail unavailable")
        abort(404, description="no thumbnail for this file type")

    @app.get("/api/original/<job_id>")
    @login_required
    def api_original(job_id: str) -> ResponseReturnValue:
        """Serve the original uploaded file for media playback.

        Video and audio files need their original bytes for <video>/<audio>
        playback in the browser. Returns the source file with a detected MIME type.
        """
        with jobs_lock:
            job = _owned_job(job_id)
            if job.status != JOB_DONE:
                abort(409, description="job not done yet")
            upload_path = job.upload_path
        if upload_path is None or not Path(upload_path).is_file():
            abort(404, description="source file not found")
        ext = Path(upload_path).suffix.lower()
        _MIME = {
            ".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mov": "video/quicktime",
            ".webm": "video/webm", ".mkv": "video/x-matroska", ".flv": "video/x-flv",
            ".wmv": "video/x-ms-wmv", ".m4v": "video/mp4",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
            ".wma": "audio/x-ms-wma",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
            ".tiff": "image/tiff", ".tif": "image/tiff",
            ".avif": "image/avif", ".heic": "image/heic", ".heif": "image/heif",
        }
        mimetype = _MIME.get(ext) or "application/octet-stream"
        return send_file(
            upload_path,
            mimetype=mimetype,
            as_attachment=False,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.post("/api/run")
    @login_required
    def api_run() -> ResponseReturnValue:
        if "file" not in request.files:
            abort(400, description="missing 'file' field in multipart upload")
        upload = request.files["file"]
        if not upload.filename:
            abort(400, description="empty filename")
        # PLAN §17 §Multi-format follow-up: the pipeline now routes a
        # dozen+ formats through PDF / DOCLING / PPTX_NATIVE / TEXT /
        # IMAGE -- the upload form should accept everything the
        # orchestrator can ingest, and nothing it cannot.
        #
        # CONVERTIBLE_EXTENSIONS, not SUPPORTED_EXTENSIONS: the latter also
        # names the legacy binary Office formats (.doc/.ppt/.xls), which the
        # router recognises but classifies SKIP. Gating on it accepted the
        # upload and then failed the job seconds later inside `ingest_any`.
        from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS
        # Path(...).name strips any directory components a malicious
        # client might have inserted (the FileStorage.filename is taken
        # from the multipart Content-Disposition verbatim). ``suffix``
        # then yields the literal lowercased extension or ``""`` if the
        # upload has none.
        safe_name = Path(upload.filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in CONVERTIBLE_EXTENSIONS:
            abort(400, description=(
                f"unsupported file type {ext!r}; supported: "
                f"{', '.join(sorted(CONVERTIBLE_EXTENSIONS))}"
            ))

        job_id = uuid.uuid4().hex
        # Preserve the original extension on disk so :func:`format_router
        # .detect_format`'s extension-fallback can still resolve a label
        # when magic bytes are ambiguous (e.g. text / markdown / code
        # files share the same "no magic" signature). The
        # ``format_router`` is robust to a missing extension because
        # the orchestrator re-derives the format from magic bytes, but
        # keeping the suffix is cheap and gives operators a clear hint
        # in the upload directory listing.
        saved = upload_dir / f"{job_id}{ext}"
        upload.save(saved)

        # Optional intent: a free-text reader-mode query.  Blank / missing
        # means "send me the full document"; the front-end maps this to
        # an opt-in text input alongside the file picker.
        intent_str = (request.form.get("intent") or "").strip() or None
        # Optional folder: the file-browser uploads into the currently-open
        # folder. ``None``/missing lands the document at the "All files" root.
        folder_raw = (request.form.get("folder") or "").strip() or None
        folder_id: int | None = int(folder_raw) if folder_raw is not None else None

        job = Job(
            job_id=job_id,
            user_id=request.user["id"],  # type: ignore[attr-defined]
            filename=safe_name,
            upload_path=saved,
            intent=intent_str,
            folder_id=folder_id,
        )
        with jobs_lock:
            jobs[job_id] = job
            # Write-through: persist the queued job so it survives a restart
            # even if the runner hasn't moved it yet.
            library.upsert_job(job)

        th = threading.Thread(target=_runner, args=(job,), daemon=True, name=f"web-{job_id[:8]}")
        th.start()
        return jsonify({"job_id": job_id, "status_url": f"/api/status/{job_id}"})

    @app.get("/api/status/<job_id>")
    @login_required
    def api_status(job_id: str) -> ResponseReturnValue:
        with jobs_lock:
            job = _owned_job(job_id)
            return jsonify(job.to_public())

    @app.get("/api/download/<job_id>")
    @login_required
    def api_download(job_id: str) -> ResponseReturnValue:
        with jobs_lock:
            job = _owned_job(job_id)
            if job.status != JOB_DONE or job.uir_path is None:
                abort(409, description=f"job not done (status={job.status})")
            path = job.uir_path
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/result/<job_id>")
    @login_required
    def api_result(job_id: str) -> ResponseReturnValue:
        """Serve the (intent-filtered, if set) UIR document JSON inline.

        When the user submitted an optional ``intent`` form field on
        ``/api/run``, this endpoint serves the narrowed result so the
        front-end receives only the chunks that match the reader query
        (reduces tokens sent to the calling LLM down to a small handful
        instead of the full document). ``/api/download/<job_id>`` still
        streams the *full* file so users can save the complete archive.
        """
        with jobs_lock:
            job = _owned_job(job_id)
            if job.status != JOB_DONE or job.uir_path is None:
                abort(409, description=f"job not done (status={job.status})")
            path = job.intent_uir_path or job.uir_path
        return send_file(
            path,
            mimetype="application/json",
            as_attachment=False,
            download_name=path.name,
        )

    @app.get("/api/umr/<job_id>")
    @login_required
    def api_umr(job_id: str) -> ResponseReturnValue:
        """Serve the UMR Markdown companion file (Phase 17).

        When the user submitted an ``intent`` field, the narrowed
        filtered markdown (``intent_umr_path``) is served so the
        front-end shows only the matched sections/chunks. Otherwise the
        full-document UMR is streamed.

        ``/api/umr/<job_id>`` is the agent-facing view that the web UI
        displays by default -- replaces the verbose UIR JSON ``<pre>``
        that used to bloat the response with entities/relationships.
        The downloadable UIR JSON still lives at ``/api/download/<job_id>``
        for power users / corpus debugging.
        """
        with jobs_lock:
            job = _owned_job(job_id)
            if job.status != JOB_DONE or job.umr_path is None:
                abort(409, description=f"job not done (status={job.status})")
            # Prefer the intent-filtered UMR when intent was supplied;
            # else the full-document UMR. Both are markdown with the
            # same mime type / same front-end renderer.
            candidate = job.intent_umr_path or job.umr_path
            if not candidate.is_file():
                # Fall through to a synthetic placeholder so a partial
                # crash in the umr builder doesn't 404 the front-end.
                return Response(
                    (
                        "# UMR not yet generated\n\n"
                        "_Pipeline run completed but the UMR file is missing._"
                    ),
                    mimetype="text/markdown",
                )
        return send_file(
            candidate,
            mimetype="text/markdown",
            as_attachment=False,
            download_name=candidate.name,
        )

    # -------- chat ------------------------------------------------------

    def _grounded_answer(
        uid: int, message: str, history: list[Any], *,
        job_ids: set[str] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Answer ``message`` from the caller's own documents.

        Returns ``(payload, status)``. Retrieval spans every ``done`` job the
        caller owns, filtered to ``job_ids`` when given. Jobs belonging to
        anyone else are excluded before retrieval, so the model never sees a
        passage from a document the caller cannot already read. Shared by
        ``/api/chat`` and the Chats panel's ``@fireworks`` command so the two
        cannot drift.
        """
        from uir_pipeline import chat as _chat

        with jobs_lock:
            docs = [
                {
                    "job_id": job.job_id,
                    "uir_path": job.uir_path,
                    "filename": job.filename,
                }
                for job in jobs.values()
                if job.user_id == uid
                and job.status == JOB_DONE
                and job.uir_path is not None
                and (job_ids is None or job.job_id in job_ids)
            ]
            paths = [Path(d["uir_path"]) for d in docs]

        if not docs:
            # Same key set as a successful answer so no caller special-cases it.
            return {
                "answer": (
                    "You haven't converted any documents yet. Upload one and "
                    "I'll be able to answer questions about it."
                ),
                "citations": [],
                "cited": [],
                "invalid_citations": [],
                "grounded": False,
                "model": None,
                "tool_steps": [],
            }, 200

        # Autonomous mode: don't pre-fetch passages. The agent must call
        # ``search`` / ``get_more_sources`` to find relevant passages itself.
        # Single-shot mode (no docs) still retrieves up-front for backward
        # compatibility with tests that don't supply a ``docs`` list.
        if docs:
            contexts = []
        else:
            contexts = _chat.retrieve(paths, message)
        result = _chat.answer(message, contexts, history=history, docs=docs, job_ids=job_ids)

        if not result["success"]:
            # The model call failed (missing key, upstream 5xx). Surface it
            # rather than inventing a reply.
            return {"error": result["error"]}, 502

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            # Which passages the answer actually cites, and any numbers the
            # model invented (already stripped from `answer`). The UI can show
            # only the cited sources instead of all six retrieved ones.
            "cited": result.get("cited", []),
            "invalid_citations": result.get("invalid_citations", []),
            "grounded": result.get("grounded", False),
            "model": result.get("model"),
            "tool_steps": result.get("tool_steps", []),
        }, 200

    @app.post("/api/search")
    @login_required
    def api_search() -> ResponseReturnValue:
        """Semantic + title-priority passage search over the caller's documents.

        Ranks passages by BGE cosine to the query, with a boost for documents
        whose title matches (title priority). Used by the global search bar;
        the agent's ``search``/``get_more_sources`` tools call the same
        :func:`uir_pipeline.search.search` directly.
        """
        body = _json_body()
        q = (body.get("query") or "").strip()
        if not q:
            return jsonify({"results": []})
        uid = request.user["id"]  # type: ignore[attr-defined]
        job_ids: set[str] | None = None
        if isinstance(body.get("job_ids"), list):
            job_ids = {str(j) for j in body["job_ids"]}
        with jobs_lock:
            docs = [
                {"job_id": job.job_id, "uir_path": job.uir_path, "filename": job.filename}
                for job in jobs.values()
                if job.user_id == uid and job.status == JOB_DONE and job.uir_path
                and (job_ids is None or job.job_id in job_ids)
            ]
        from uir_pipeline.search import search as _doc_search
        top_k = int(body.get("top_k") or 8)
        return jsonify({"results": _doc_search(docs, q, top_k=top_k)})

    @app.post("/api/chat")
    @login_required
    def api_chat() -> ResponseReturnValue:
        """Answer a question grounded in the caller's converted documents.

        Retrieval spans every ``done`` job the caller owns, unless the
        body narrows it with ``job_ids``. The agent may call ``search`` /
        ``get_more_sources`` to gather more passages before answering.
        """
        body = _json_body()
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400

        uid = request.user["id"]  # type: ignore[attr-defined]
        job_ids: set[str] | None = None
        if isinstance(body.get("job_ids"), list):
            job_ids = {str(j) for j in body["job_ids"]}

        # Parse @filename mentions in the message and scope retrieval to them.
        mention_ids, message = _parse_file_mentions(message, uid, jobs, jobs_lock)
        if mention_ids:
            job_ids = (job_ids or set()) | mention_ids

        history_raw = body.get("history")
        history: list[Any] = history_raw if isinstance(history_raw, list) else []

        payload, status = _grounded_answer(uid, message, history, job_ids=job_ids)
        return jsonify(payload), status

    # -------- conversations (Chats panel) -------------------------------

    @app.get("/api/conversations")
    @login_required
    def api_conversations_list() -> ResponseReturnValue:
        email = request.user["email"]  # type: ignore[attr-defined]
        cs = conversations.list_for_email(email)
        for c in cs:
            c["peer_registered"] = bool(users.get_by_email(c["peer_email"]))
        return jsonify({"conversations": cs})

    @app.post("/api/conversations")
    @login_required
    def api_conversations_create() -> ResponseReturnValue:
        """Start (or reopen) a 1:1 thread with the person at ``peer_email``.

        The peer need not have an account: membership is by email, so they
        see the thread when they sign up with that address. We do not check
        whether the address is registered, both to keep the flow simple and
        to avoid turning this into an account-enumeration oracle.
        """
        from uir_pipeline.conversations import ConversationError

        email = request.user["email"]  # type: ignore[attr-defined]
        body = _json_body()
        try:
            convo, created = conversations.create_with(email, body.get("peer_email") or "")
        except ConversationError as exc:
            return jsonify({"error": str(exc)}), 400
        convo["peer_registered"] = bool(users.get_by_email(convo.get("peer_email", "")))
        return jsonify({"conversation": convo}), (201 if created else 200)

    @app.delete("/api/conversations/<int:cid>")
    @login_required
    def api_conversations_delete(cid: int) -> ResponseReturnValue:
        email = request.user["email"]  # type: ignore[attr-defined]
        # 404 (not 403) for a non-member id: never confirm it exists.
        if not conversations.leave(email, cid):
            abort(404, description="conversation not found")
        return jsonify({"ok": True})

    @app.get("/api/conversations/<int:cid>/messages")
    @login_required
    def api_conversation_messages(cid: int) -> ResponseReturnValue:
        email = request.user["email"]  # type: ignore[attr-defined]
        convo = conversations.get_for_email(email, cid)
        if convo is None:
            abort(404, description="conversation not found")
        convo["peer_registered"] = bool(users.get_by_email(convo.get("peer_email", "")))
        return jsonify({
            "conversation": convo,
            "messages": conversations.list_messages(cid),
        })

    @app.post("/api/conversations/<int:cid>/messages")
    @login_required
    def api_conversation_send(cid: int) -> ResponseReturnValue:
        """Post a message to a thread the caller is a member of.

        A plain message is stored as the caller's message (the other member
        sees it). A message beginning with the ``@fireworks`` trigger is stored
        verbatim, then its remainder is answered from the *sender's* own
        documents and the reply is stored as a shared ``assistant`` message
        both members see. The user message is persisted before the model
        runs, so a model outage never loses what the user typed.
        """
        email = request.user["email"]  # type: ignore[attr-defined]
        uid = request.user["id"]  # type: ignore[attr-defined]
        convo = conversations.get_for_email(email, cid)
        if convo is None:
            abort(404, description="conversation not found")

        body = _json_body()
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        question = _fireworks_question(text)
        if question is not None and not question:
            return jsonify({"error": "Add a question after '@fireworks'."}), 400

        user_msg = conversations.add_message(cid, "user", text, sender_email=email)

        if question is None:
            # Ordinary message between the two people -- nothing to answer.
            return jsonify({"user_message": user_msg, "reply": None})

        # Parse @filename mentions in the question and scope retrieval.
        mention_ids, question = _parse_file_mentions(question, uid, jobs, jobs_lock)
        job_ids = mention_ids if mention_ids else None

        # Model history: this thread's prior fireworks exchanges only -- the
        # stripped questions and the assistant answers. Ordinary person-to-
        # person messages are not part of the model dialogue.
        history: list[dict[str, Any]] = []
        for m in conversations.list_messages(cid):
            if m["id"] == user_msg["id"]:
                continue
            if m["role"] == "assistant":
                history.append({"role": "assistant", "content": m["content"]})
            else:
                q = _fireworks_question(m["content"])
                if q:
                    history.append({"role": "user", "content": q})

        payload, status = _grounded_answer(uid, question, history, job_ids=job_ids)
        if status != 200:
            # Model failed. The user message is already saved; surface the
            # error without fabricating an assistant reply.
            return jsonify({"user_message": user_msg, "error": payload.get("error")}), 502

        reply = conversations.add_message(
            cid, "assistant", payload["answer"],
            citations=payload.get("cited") or payload.get("citations") or [],
            grounded=bool(payload.get("grounded")),
            tool_steps=payload.get("tool_steps") or [],
        )
        return jsonify({"user_message": user_msg, "reply": reply})

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    @app.errorhandler(429)
    def _handle_http_err(exc: Any) -> ResponseReturnValue:
        return jsonify({"error": exc.description if hasattr(exc, "description") else str(exc)}), exc.code

    # Belt-and-suspenders cache-busting on /static/<...>. The console ships
    # Babel-in-the-browser and serves its own ``.jsx`` files over GET
    # ``/static/...``; if a browser caches them, a stale (or now-removed)
    # file keeps being parsed on every reload and surfaces as a ghostly
    # Babel SyntaxError long after the source went away.
    # ``Cache-Control: no-store`` tells the browser to never cache at all
    # -- the only fully-bulletproof answer for this category of "Babel
    # parses a file that no longer exists" failure. ``max-age=0`` was
    # considered but weak: revalidation against a stable ETag can still
    # return 304 and re-serve the stale body.
    # Dev-grade behaviour: the design intent (templates/console.html's own
    # comment) is to drop the whole Babel pass before any LAN demo grows up
    # into a real deploy, at which point this hook goes away with it.
    #
    # Endpoint check rather than path-prefix -- so a future blueprint that
    # registers its own static folder (which would name its endpoint
    # ``blueprintname.static``, not ``"static"``) keeps standard caching.
    @app.after_request
    def _no_store_on_static(response: Response) -> Response:
        if request.endpoint == "static":
            response.headers["Cache-Control"] = "no-store"
        return response

    # Repopulate the in-memory job registry from SQLite so the file-browser
    # library (folders + documents) survives a server restart. Orphaned
    # in-flight jobs are marked errored here -- their runner thread is gone.
    _rehydrate_jobs(app)

    return app


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _advance(
    job: Job,
    status: str,
    stage: str,
    percent: int,
    *,
    lock: threading.Lock,
) -> None:
    """Atomically update job state from the runner thread."""
    with lock:
        job.status = status
        job.progress_stage = stage
        job.progress_percent = max(0, min(100, int(percent)))


def _rehydrate_jobs(app: Flask) -> None:
    """Repopulate the in-memory job registry from SQLite after a restart.

    Jobs that were still ``queued``/``running`` when the server died are
    orphaned: their daemon thread is gone and will never advance them again.
    Flip them to ``error`` so the UI shows a recoverable state instead of a
    spinner that never resolves, and persist that flip. ``done``/``error``
    jobs come back verbatim, so the file-browser library reappears exactly
    as the user left it. Safe to call on a fresh database (no rows -> no-op).
    """
    store = app.config.get("LIBRARY_STORE")
    jobs = app.config.get("JOBS")
    if store is None or jobs is None:
        return
    for row in store.list_all_job_rows():
        job = Job(**row)
        if job.status in (JOB_QUEUED, JOB_RUNNING):
            job.status = JOB_ERROR
            job.error = "Server restarted while this job was running."
            job.finished_at = time.time()
            try:
                store.upsert_job(job)
            except Exception as exc:  # noqa: BLE001 -- best-effort
                logger.warning("rehydrate persist failed for %s: %s", job.job_id, exc)
        jobs[job.job_id] = job


#: The Chats panel's inline-assistant trigger. A message whose first
#: non-space characters are ``@fireworks`` (any case) is a command: the text
#: after the trigger is answered from the caller's documents. The keyword is
#: literally "fireworks" by product request, even though the backend model is
#: Fireworks -- it is the user-facing invocation word, not a model id.
_FIREWORKS_PREFIX = re.compile(r"^\s*@fireworks\b\s*", re.IGNORECASE)


def _fireworks_question(text: str) -> str | None:
    """Return the question after an ``@fireworks`` trigger, else ``None``.

    ``None`` means "not a command, store as a plain note". An empty string
    means "was a command but nothing followed the trigger" -- the caller
    rejects that rather than sending the model an empty prompt.
    """
    m = _FIREWORKS_PREFIX.match(text or "")
    if not m:
        return None
    return text[m.end():].strip()


#: Matches ``@filename`` file mentions anywhere in the text.  The filename is
#: captured as group 1 and may include spaces or dots (e.g. ``@My Report.pdf``).
_FILE_MENTION_RE: re.Pattern[str] = re.compile(r"@([^\s@]+(?:\.\S+)?)")


def _parse_file_mentions(
    text: str,
    uid: int,
    jobs: dict[str, Any],
    jobs_lock: threading.Lock,
) -> tuple[set[str] | None, str]:
    """Extract ``@filename`` mentions from ``text`` and resolve them to job IDs.

    Returns ``(job_ids, cleaned_text)``.  ``job_ids`` is ``None`` when no
    mentions were found, otherwise a set of ``job_id`` strings belonging to
    ``uid``'s DONE jobs whose filename matches a mention.  The matching
    mentions are removed from the returned text so the model never sees them.
    """
    if not text or not text.strip():
        return None, text

    mentions: set[str] = set()
    with jobs_lock:
        job_map = {
            job.filename: job.job_id
            for job in jobs.values()
            if job.user_id == uid and job.status == JOB_DONE and job.filename
        }
    if not job_map:
        return None, text

    def _replace(match: re.Match[str]) -> str:
        mention = match.group(1)
        # Look for an exact filename match, or a case-insensitive one.
        for filename, job_id in job_map.items():
            if filename.lower() == mention.lower():
                mentions.add(job_id)
                return ""  # strip the mention from the text
        return match.group(0)  # keep unmatched @words

    cleaned = _FILE_MENTION_RE.sub(_replace, text)
    # Normalise whitespace left by stripped mentions.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return (mentions if mentions else None), cleaned


#: Hosts whose traffic never leaves the machine.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_loopback_host(host: str) -> bool:
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def assert_safe_bind(host: str, port: int) -> None:
    """Refuse to serve auth over plain HTTP on a routable interface.

    Passwords are POSTed and the session cookie is replayed on every request.
    Over plain HTTP both cross the network in cleartext, so anyone on the same
    wifi can read them or steal the session outright. Loopback is fine -- the
    traffic never leaves the machine. A routable bind is not, unless a TLS
    terminator sits in front, which is exactly what ``SESSION_COOKIE_SECURE``
    asserts.

    Every entrypoint must call this. ``web.py`` at the repo root is the one
    the README tells people to run, and it binds ``0.0.0.0`` by history.

    Raises:
        SystemExit: routable bind, no TLS, no explicit override.
    """
    if is_loopback_host(host) or _env_flag("SESSION_COOKIE_SECURE", default=False):
        return
    if not _env_flag("UIR_ALLOW_INSECURE_BIND", default=False):
        raise SystemExit(
            f"Refusing to serve on {host}:{port} over plain HTTP.\n"
            "This console has user accounts: passwords and session cookies "
            "would cross the network in cleartext.\n\n"
            "  - Put a TLS terminator in front and set SESSION_COOKIE_SECURE=1, or\n"
            "  - bind loopback with HOST=127.0.0.1, or\n"
            "  - set UIR_ALLOW_INSECURE_BIND=1 if this network is genuinely trusted."
        )
    logger.warning(
        "serving on %s over plain HTTP with UIR_ALLOW_INSECURE_BIND=1: "
        "passwords and session cookies are readable by anyone on this network",
        host,
    )


def register_worker_shutdown(app: Flask) -> None:
    """Ask the pipeline child to stop between jobs at interpreter exit.

    Entrypoints call this; ``create_app`` must not. The factory runs once per
    test, and an atexit handler per app would accumulate unboundedly and fire
    after pytest has closed the streams the worker logs to.
    """
    worker = app.config.get("PIPELINE_WORKER")
    if worker is not None:
        atexit.register(worker.shutdown)


# Allow ``python -m uir_pipeline.web`` for power users.
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _port = int(os.environ.get("PORT", "5000"))
    _host = os.environ.get("HOST", "127.0.0.1")
    assert_safe_bind(_host, _port)

    _app = create_app()
    register_worker_shutdown(_app)
    _app.run(host=_host, port=_port, debug=False, use_reloader=False)


__all__ = [
    "assert_safe_bind",
    "create_app",
    "is_loopback_host",
    "register_worker_shutdown",
]
