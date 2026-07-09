"""web -- Phase N minimal HTTP front-end for testing the pipeline.

Routes:
    GET  /                            -- single-page upload form
    POST /api/run                     -- accepts ``file=<pdf>``, returns ``{job_id}``
    GET  /api/status/<job_id>         -- returns ``{status, progress, ...}``
    GET  /api/download/<job_id>       -- streams the produced ``.uir.json`` file
    GET  /api/health                   -- readiness probe

Architecture:
    * ``create_app(...)`` returns a Flask application. Tests use this
      factory with the production module's ``pipeline.run`` monkey-patched
      to a stub so the test suite never pulls in BGE / spaCy / pdfplumber.
    * Each upload is written to a per-app ``upload_dir`` and a job-id in a
      thread-safe dict tracks ``queued | running | done | error`` state with
      a progress percentage and the final ``PipelineResult``.
    * The runner thread catches all exceptions and records them in the job
      dict so the front-end can surface them without crashing the worker.

The web app is intentionally *simple* -- no auth, no queue, no Redis, no
rate-limiting.  Keep it that way until we know what real users need.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

logger = logging.getLogger(__name__)


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
    template_folder: Path | None = None,
    static_folder: Path | None = None,
    max_upload_mb: int = 64,
) -> Flask:
    """Build a Flask application with isolated per-instance state.

    Tests use this factory with monkey-patched ``pipeline.run`` so they
    don't pay the heavy-dep startup cost.
    """
    upload_dir = (Path(upload_dir) if upload_dir else Path("/tmp/uir_web_uploads")).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir = (Path(output_dir) if output_dir else Path("/tmp/uir_web_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Default templates/static fold under the package.
    _pkg_root = Path(__file__).resolve().parent
    template_folder = str(template_folder or (_pkg_root.parent.parent / "templates"))
    static_folder = str(static_folder or (_pkg_root.parent.parent / "static"))

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
    )
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["UPLOAD_DIR"] = upload_dir
    app.config["OUTPUT_DIR"] = output_dir

    # Per-app job registry -- thread-safe by virtue of the GIL guarding dict
    # assignment + the lock below for read-modify-write on the Job itself.
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()

    # Import the orchestrator lazily so tests with a stubbed module attribute
    # can patch it before the runner thread is started.
    from uir_pipeline import pipeline as _pipeline_mod

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

            result = _pipeline_mod.run(                  job.upload_path,
                  fast_path="docling",  # web pins Docling (single backend; docling failures propagate)
                  output_dir=app.config["OUTPUT_DIR"],
                skip_weaviate=True,        # web UX: skip Weaviate by default
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
                    uir_doc = json.loads(job.uir_path.read_text())
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
                                    json.loads(job.intent_uir_path.read_text()),
                                    intent_filter=intent_filter_arg,
                                )
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
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()

    # -------- routes ----------------------------------------------------

    @app.get("/")
    def index() -> Response:
        return render_template("index.html", max_upload_mb=max_upload_mb)

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"ok": True, "upload_dir": str(upload_dir)})

    @app.post("/api/run")
    def api_run() -> Response:
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

        job = Job(job_id=job_id, upload_path=saved, intent=intent_str)
        with jobs_lock:
            jobs[job_id] = job

        th = threading.Thread(target=_runner, args=(job,), daemon=True, name=f"web-{job_id[:8]}")
        th.start()
        return jsonify({"job_id": job_id, "status_url": f"/api/status/{job_id}"})

    @app.get("/api/status/<job_id>")
    def api_status(job_id: str) -> Response:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404, description="job not found")
            return jsonify(job.to_public())

    @app.get("/api/download/<job_id>")
    def api_download(job_id: str) -> Response:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404, description="job not found")
            if job.status != JOB_DONE or job.uir_path is None:
                abort(409, description=f"job not done (status={job.status})")
            path = job.uir_path
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/result/<job_id>")
    def api_result(job_id: str) -> Response:
        """Serve the (intent-filtered, if set) UIR document JSON inline.

        When the user submitted an optional ``intent`` form field on
        ``/api/run``, this endpoint serves the narrowed result so the
        front-end receives only the chunks that match the reader query
        (reduces tokens sent to the calling LLM down to a small handful
        instead of the full document). ``/api/download/<job_id>`` still
        streams the *full* file so users can save the complete archive.
        """
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404, description="job not found")
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
    def api_umr(job_id: str) -> Response:
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
            job = jobs.get(job_id)
            if job is None:
                abort(404, description="job not found")
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

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    def _handle_http_err(exc: Any) -> Response:
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
    create_app().run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


__all__ = ["create_app"]
