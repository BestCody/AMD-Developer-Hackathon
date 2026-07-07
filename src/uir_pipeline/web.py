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
    error: str | None = None

    def to_public(self) -> dict[str, Any]:
        """Shape we return to the front-end (excludes paths)."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.progress_stage,
            "percent": self.progress_percent,
            "submitted_at": self.submitted_at,
            "finished_at": self.finished_at,
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

            def _on_progress(stage: str, pct: int) -> None:
                _advance(job, JOB_RUNNING, stage, pct, lock=jobs_lock)

            result = _pipeline_mod.run(
                job.upload_path,
                output_dir=app.config["OUTPUT_DIR"],
                skip_weaviate=True,        # web UX: skip Weaviate by default
                with_embeddings=True,
                on_progress=_on_progress,
            )
            _advance(job, JOB_RUNNING, "done", 100, lock=jobs_lock)
            with jobs_lock:
                job.uir_path = Path(result.out_path)
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
        if not upload.filename.lower().endswith(".pdf"):
            abort(400, description="only .pdf files are accepted")

        job_id = uuid.uuid4().hex
        saved = upload_dir / f"{job_id}.pdf"
        upload.save(saved)

        job = Job(job_id=job_id, upload_path=saved)
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
        """Serve the full UIR document JSON inline (no attachment header).

        The front-end fetches this on job completion so the user sees the
        full ``UIRV1`` document (with its ``structure``/``semantics``/
        ``provenance`` blocks) instead of just the metadata shim.  Use
        ``/api/download/<job_id>`` to save the file.
        """
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404, description="job not found")
            if job.status != JOB_DONE or job.uir_path is None:
                abort(409, description=f"job not done (status={job.status})")
            path = job.uir_path
        return send_file(
            path,
            mimetype="application/json",
            as_attachment=False,
            download_name=path.name,
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
