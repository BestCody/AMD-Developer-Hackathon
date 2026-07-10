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
        for q in (self._job_q, self._progress_q, self._result_q):
            try:
                if q is not None:
                    q.close()
            except Exception:  # noqa: BLE001
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
    from uir_pipeline.auth import UserStore, resolve_secret_key

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

    # Per-app job registry -- thread-safe by virtue of the GIL guarding dict
    # assignment + the lock below for read-modify-write on the Job itself.
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()

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

    def _runner(job: Job) -> None:
        """Background thread body -- runs the pipeline and updates the Job."""
        try:
            _advance(job, JOB_RUNNING, "ingest", 5, lock=jobs_lock)

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

    # -------- jobs ------------------------------------------------------

    @app.get("/api/jobs")
    @login_required
    def api_jobs() -> ResponseReturnValue:
        uid = request.user["id"]  # type: ignore[attr-defined]
        with jobs_lock:
            mine = [j.to_public() for j in jobs.values() if j.user_id == uid]
        mine.sort(key=lambda j: j["submitted_at"])
        return jsonify({"jobs": mine})

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
        # orchestrator can ingest. We use the format_router's canonical
        # SUPPORTED_EXTENSIONS list as the source of truth so a future
        # format addition is a one-line change.
        from uir_pipeline.format_router import SUPPORTED_EXTENSIONS
        # Path(...).name strips any directory components a malicious
        # client might have inserted (the FileStorage.filename is taken
        # from the multipart Content-Disposition verbatim). ``suffix``
        # then yields the literal lowercased extension or ``""`` if the
        # upload has none.
        safe_name = Path(upload.filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            abort(400, description=(
                f"unsupported file type {ext!r}; supported: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
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

        job = Job(
            job_id=job_id,
            user_id=request.user["id"],  # type: ignore[attr-defined]
            filename=safe_name,
            upload_path=saved,
            intent=intent_str,
        )
        with jobs_lock:
            jobs[job_id] = job

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

    @app.post("/api/chat")
    @login_required
    def api_chat() -> ResponseReturnValue:
        """Answer a question grounded in the caller's converted documents.

        Retrieval spans every ``done`` job the caller owns, unless the
        body narrows it with ``job_ids``. Jobs belonging to anyone else
        are filtered out before retrieval -- the model never sees a
        passage from a document the caller cannot already read.
        """
        from uir_pipeline import chat as _chat

        body = _json_body()
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400

        uid = request.user["id"]  # type: ignore[attr-defined]
        requested: set[str] | None = None
        if isinstance(body.get("job_ids"), list):
            requested = {str(j) for j in body["job_ids"]}

        with jobs_lock:
            paths = [
                job.uir_path
                for job in jobs.values()
                if job.user_id == uid
                and job.status == JOB_DONE
                and job.uir_path is not None
                and (requested is None or job.job_id in requested)
            ]

        if not paths:
            return jsonify({
                "answer": (
                    "You haven't converted any documents yet. Upload one and "
                    "I'll be able to answer questions about it."
                ),
                "citations": [],
                "grounded": False,
            })

        history = body.get("history") if isinstance(body.get("history"), list) else []
        contexts = _chat.retrieve(paths, message)
        result = _chat.answer(message, contexts, history=history)

        if not result["success"]:
            # The model call failed (missing key, upstream 5xx). Surface it
            # rather than inventing a reply.
            return jsonify({"error": result["error"]}), 502

        return jsonify({
            "answer": result["answer"],
            "citations": result["citations"],
            # Which passages the answer actually cites, and any numbers the
            # model invented (already stripped from `answer`). The UI can show
            # only the cited sources instead of all six retrieved ones.
            "cited": result.get("cited", []),
            "invalid_citations": result.get("invalid_citations", []),
            "grounded": result.get("grounded", False),
            "model": result.get("model"),
        })

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    @app.errorhandler(429)
    def _handle_http_err(exc: Any) -> ResponseReturnValue:
        return jsonify({"error": exc.description if hasattr(exc, "description") else str(exc)}), exc.code

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


# Allow ``python -m uir_pipeline.web`` for power users.
if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")

    # Passwords are POSTed and the session cookie is replayed on every request.
    # Over plain HTTP both cross the network in cleartext, so anyone on the same
    # wifi can read them or steal the session. Loopback is fine (the traffic
    # never leaves the machine); a routable bind is not, unless a TLS terminator
    # sits in front -- which is exactly what SESSION_COOKIE_SECURE asserts.
    _loopback = host in ("127.0.0.1", "::1", "localhost")
    if not _loopback and not _env_flag("SESSION_COOKIE_SECURE", default=False):
        if not _env_flag("UIR_ALLOW_INSECURE_BIND", default=False):
            raise SystemExit(
                f"Refusing to serve on {host}:{port} over plain HTTP.\n"
                "Passwords and session cookies would cross the network in "
                "cleartext.\n\n"
                "  - Put a TLS terminator in front and set SESSION_COOKIE_SECURE=1, or\n"
                "  - keep HOST=127.0.0.1 (the default), or\n"
                "  - set UIR_ALLOW_INSECURE_BIND=1 if this network is genuinely trusted."
            )
        logger.warning(
            "serving on %s over plain HTTP with UIR_ALLOW_INSECURE_BIND=1: "
            "passwords and session cookies are readable by anyone on this network",
            host,
        )

    _app = create_app()
    _worker = _app.config.get("PIPELINE_WORKER")
    if _worker is not None:
        # Ask the child to stop between jobs rather than be killed mid-write.
        atexit.register(_worker.shutdown)
    _app.run(host=host, port=port, debug=False, use_reloader=False)


__all__ = ["create_app"]
