"""The pipeline runs in a child process, so a native crash costs one job.

Background: Docling's C++ layer raises ``std::bad_alloc`` under memory
pressure and can abort the process. When the orchestrator ran in a
``threading.Thread``, that killed the whole Flask server -- uploading a
mid-sized PDF was a one-request denial of service, and ``_runner``'s
``except Exception`` never fired because SIGSEGV is not an exception.

These tests drive the real ``spawn`` machinery. They cannot use
``monkeypatch.setattr(pipeline_mod, "run", ...)`` -- that patch lives in
the parent and a spawned child re-imports everything fresh. Instead the
child is pointed at ``tests.stub_pipelines`` via ``UIR_PIPELINE_MODULE``
(an env var, which *does* cross the boundary), and its behaviour is
selected with ``STUB_PIPELINE_MODE``.
"""
from __future__ import annotations

import io
import time

import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A real app in the production execution mode: process isolation."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    monkeypatch.setenv("UIR_PIPELINE_MODULE", "tests.stub_pipelines")

    from uir_pipeline.web import create_app
    application = create_app(
        upload_dir=tmp_path / "up",
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        execution="process",
    )
    application.config.update(TESTING=True)
    return application


def _client(app):
    c = app.test_client()
    r = c.post("/api/auth/signup", json={"email": "iso@example.com", "password": "iso-password-12"})
    assert r.status_code == 200, r.get_data(as_text=True)
    return c


def _upload(client, name="doc.pdf", mode=None):
    """Upload one PDF. ``mode`` selects the stub's behaviour for *this job*.

    The worker is long-lived, so ``STUB_PIPELINE_MODE`` set in the parent
    after the child spawned would never reach it. ``intent`` is per job and
    crosses on every call, so mode-switching tests must use it.
    """
    data = {"file": (io.BytesIO(b"%PDF-1.4 fake"), name)}
    if mode is not None:
        data["intent"] = mode
    r = client.post("/api/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["job_id"]


def _wait(client, job_id, timeout_s=90.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never settled")


# ---------------------------------------------------------------------------
# happy path across the process boundary
# ---------------------------------------------------------------------------

def test_process_mode_completes_a_job(app, monkeypatch):
    monkeypatch.setenv("STUB_PIPELINE_MODE", "ok")
    c = _client(app)
    final = _wait(c, _upload(c))
    assert final["status"] == "done", final.get("error")
    assert final["result"]["chunk_count"] == 1


def test_artifacts_written_by_the_child_are_served_by_the_parent(app, monkeypatch):
    monkeypatch.setenv("STUB_PIPELINE_MODE", "ok")
    c = _client(app)
    job = _upload(c)
    assert _wait(c, job)["status"] == "done"
    assert "Stub Doc" in c.get(f"/api/umr/{job}").get_data(as_text=True)
    assert c.get(f"/api/result/{job}").get_json()["id"] == "doc_stub"
    assert c.get(f"/api/download/{job}").status_code == 200


def test_progress_crosses_the_process_boundary(app, monkeypatch):
    """Stage updates from the child must reach the parent's Job."""
    monkeypatch.setenv("STUB_PIPELINE_MODE", "ok")
    c = _client(app)
    job = _upload(c)
    seen = set()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        s = c.get(f"/api/status/{job}").get_json()
        seen.add(s["stage"])
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    # "done" is set by the parent; the others can only have come from the child.
    assert seen & {"ingest", "chunk", "embed"}, f"no child progress observed: {seen}"


# ---------------------------------------------------------------------------
# the reason this exists
# ---------------------------------------------------------------------------

def test_native_crash_fails_the_job_instead_of_the_server(app, monkeypatch):
    monkeypatch.setenv("STUB_PIPELINE_MODE", "crash")
    c = _client(app)
    final = _wait(c, _upload(c))

    assert final["status"] == "error"
    assert "worker died" in final["error"], final["error"]
    # The exit code is the only forensic signal a SIGSEGV leaves us.
    assert "exit code" in final["error"]

    # ...and the server is still answering.
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/auth/me").status_code == 200


def test_server_still_converts_after_a_crash(app):
    """A crashed job must not poison the worker pool or the session."""
    c = _client(app)
    assert _wait(c, _upload(c, "boom.pdf", mode="crash"))["status"] == "error"

    good = _wait(c, _upload(c, "fine.pdf", mode="ok"))
    assert good["status"] == "done", good.get("error")
    assert good["filename"] == "fine.pdf"


def test_a_crash_does_not_leak_into_other_jobs_status(app):
    c = _client(app)
    bad = _upload(c, "boom.pdf", mode="crash")
    assert _wait(c, bad)["status"] == "error"

    good = _upload(c, "fine.pdf", mode="ok")
    assert _wait(c, good)["status"] == "done"

    jobs = {j["job_id"]: j for j in c.get("/api/jobs").get_json()["jobs"]}
    assert jobs[bad]["status"] == "error"
    assert jobs[good]["status"] == "done"


def test_child_exception_becomes_a_job_error_with_its_message(app, monkeypatch):
    """An ordinary Python exception in the child is reported, not swallowed."""
    monkeypatch.setenv("STUB_PIPELINE_MODE", "raise")
    c = _client(app)
    final = _wait(c, _upload(c))
    assert final["status"] == "error"
    assert "ValueError" in final["error"]
    assert "stub pipeline exploded" in final["error"]
    assert c.get("/api/health").status_code == 200


def test_child_exception_is_not_double_prefixed(app, monkeypatch):
    """`ValueError: msg`, not `PipelineWorkerError: ValueError: msg`."""
    monkeypatch.setenv("STUB_PIPELINE_MODE", "raise")
    c = _client(app)
    err = _wait(c, _upload(c))["error"]
    assert err.startswith("ValueError:"), err
    assert "PipelineWorkerError" not in err
    assert "RuntimeError" not in err


# ---------------------------------------------------------------------------
# the worker stays warm across jobs
# ---------------------------------------------------------------------------
# Spawning per job cost ~8s of `import torch` / `import docling` before any
# work started. One long-lived child pays that once. These tests pin the
# reuse, and the respawn that keeps crash isolation intact.

def _pid(app):
    proc = app.config["PIPELINE_WORKER"]._proc
    return None if proc is None else proc.pid


def test_worker_is_not_started_until_the_first_job(app):
    """An idle server must not hold torch + docling weights resident."""
    assert app.config["PIPELINE_WORKER"] is not None
    assert _pid(app) is None


def test_worker_process_is_reused_across_jobs(app, monkeypatch):
    monkeypatch.setenv("STUB_PIPELINE_MODE", "ok")
    c = _client(app)
    assert _wait(c, _upload(c, "one.pdf"))["status"] == "done"
    first = _pid(app)
    assert first is not None

    assert _wait(c, _upload(c, "two.pdf"))["status"] == "done"
    assert _pid(app) == first, "second job must not spawn a new interpreter"


def test_worker_is_respawned_after_a_native_crash(app):
    c = _client(app)
    assert _wait(c, _upload(c, "one.pdf", mode="ok"))["status"] == "done"
    before = _pid(app)

    assert _wait(c, _upload(c, "boom.pdf", mode="crash"))["status"] == "error"

    assert _wait(c, _upload(c, "two.pdf", mode="ok"))["status"] == "done"
    after = _pid(app)
    assert after is not None and after != before, "crash must force a fresh worker"


def test_a_python_exception_keeps_the_worker_alive(app):
    """One bad PDF must not cost the next upload an 8s respawn."""
    c = _client(app)
    assert _wait(c, _upload(c, "one.pdf", mode="ok"))["status"] == "done"
    before = _pid(app)

    assert _wait(c, _upload(c, "bad.pdf", mode="raise"))["status"] == "error"
    assert _pid(app) == before, "a caught exception must not kill the worker"

    assert _wait(c, _upload(c, "two.pdf", mode="ok"))["status"] == "done"
    assert _pid(app) == before


def test_worker_respawns_when_a_docling_env_var_changes(app, monkeypatch):
    """The child reads DOCLING_* at spawn. Changing it must take effect."""
    c = _client(app)
    assert _wait(c, _upload(c, "one.pdf", mode="ok"))["status"] == "done"
    before = _pid(app)

    monkeypatch.setenv("DOCLING_OCR", "off")
    assert _wait(c, _upload(c, "two.pdf", mode="ok"))["status"] == "done"
    assert _pid(app) != before, "a changed DOCLING_* must respawn the worker"


def test_worker_is_not_respawned_for_an_unrelated_env_var(app, monkeypatch):
    c = _client(app)
    assert _wait(c, _upload(c, "one.pdf", mode="ok"))["status"] == "done"
    before = _pid(app)

    monkeypatch.setenv("SOME_UNRELATED_VAR", "1")
    assert _wait(c, _upload(c, "two.pdf", mode="ok"))["status"] == "done"
    assert _pid(app) == before, "only watched vars force a respawn"


def test_env_fingerprint_watches_docling_and_pipeline_module(monkeypatch):
    from uir_pipeline.web import _worker_env_fingerprint

    monkeypatch.setenv("DOCLING_OCR", "auto")
    monkeypatch.setenv("UIR_PIPELINE_MODULE", "x")
    monkeypatch.setenv("TOTALLY_UNRELATED", "y")
    fp = _worker_env_fingerprint()
    assert fp["DOCLING_OCR"] == "auto"
    assert fp["UIR_PIPELINE_MODULE"] == "x"
    assert "TOTALLY_UNRELATED" not in fp


def test_shutdown_stops_the_worker(app, monkeypatch):
    monkeypatch.setenv("STUB_PIPELINE_MODE", "ok")
    c = _client(app)
    assert _wait(c, _upload(c))["status"] == "done"
    w = app.config["PIPELINE_WORKER"]
    assert w._proc.is_alive()
    w.shutdown()
    assert w._proc is None


def test_a_bad_pipeline_module_is_reported_not_hung(tmp_path, monkeypatch):
    """An unimportable orchestrator must fail the job, not block forever."""
    monkeypatch.setenv("SECRET_KEY", "x")
    monkeypatch.setenv("UIR_PIPELINE_MODULE", "tests.no_such_pipeline_module")
    from uir_pipeline.web import create_app
    application = create_app(
        upload_dir=tmp_path / "up", output_dir=tmp_path / "out",
        data_dir=tmp_path / "data", execution="process",
    )
    application.config.update(TESTING=True)
    c = _client(application)
    final = _wait(c, _upload(c))
    assert final["status"] == "error"
    assert "ModuleNotFoundError" in final["error"], final["error"]
    assert c.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_thread_mode_has_no_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "x")
    from uir_pipeline.web import create_app
    a = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                   data_dir=tmp_path / "d", execution="thread")
    assert a.config["PIPELINE_WORKER"] is None

def test_execution_mode_defaults_to_process(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "x")
    monkeypatch.delenv("UIR_WEB_EXECUTION", raising=False)
    from uir_pipeline.web import create_app
    a = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o", data_dir=tmp_path / "d")
    assert a.config["EXECUTION"] == "process"


def test_execution_mode_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "x")
    monkeypatch.setenv("UIR_WEB_EXECUTION", "thread")
    from uir_pipeline.web import create_app
    a = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o", data_dir=tmp_path / "d")
    assert a.config["EXECUTION"] == "thread"


def test_invalid_execution_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "x")
    from uir_pipeline.web import create_app
    with pytest.raises(ValueError, match="process.*thread"):
        create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                   data_dir=tmp_path / "d", execution="magic")
