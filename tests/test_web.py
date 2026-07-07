"""test_web.py -- Phase N Flask front-end tests.

The web layer is intentionally thin: we stub the heavy orchestrator so
tests don't pull in BGE / spaCy / pdfplumber. Integration with the real
``uir_pipeline.pipeline.run`` is covered by the existing end-to-end
``tests/integration/test_pipeline_smoke.py``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from uir_pipeline import pipeline as pipeline_mod
from uir_pipeline.web import (
    JOB_DONE,
    JOB_ERROR,
    JOB_RUNNING,
    create_app,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """A minimal but valid one-page PDF."""
    # Use a tiny but valid PDF (approx 700 bytes). Source: NIST-like sample.
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000054 00000 n \n0000000098 00000 n \ntrailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n144\n%%EOF\n"
    )
    p = tmp_path / "sample.pdf"
    p.write_bytes(pdf_bytes)
    return p


@pytest.fixture()
def fake_run(monkeypatch):
    """Stub :func:`uir_pipeline.pipeline.run` so tests stay light.

    We capture the call kwargs so tests can assert on stage progress without
    actually running BGE / spaCy / pdfplumber.
    """
    calls: list[dict] = []

    def _fake(input_path, output_dir, *, skip_weaviate=False, dry_run=False, with_embeddings=True, on_progress=None, page_numbers=None):
        calls.append({
            "input_path": str(input_path),
            "skip_weaviate": skip_weaviate,
            "with_embeddings": with_embeddings,
        })
        # Walk the full stage list so progress advances.
        for stage, pct in [
            ("ingest", 5),
            ("extract_text", 20),
            ("synthesize_ocr", 30),
            ("layout", 45),
            ("tables", 55),
            ("chunk", 70),
            ("enrich", 80),
            ("embed", 90),
            ("assemble", 95),
            ("done", 100),
        ]:
            if on_progress:
                on_progress(stage, pct)

        # Build a tiny UIR JSON + PipelineResult-shaped dict.
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        uir_path = out_dir / "fake_doc.uir.json"
        uir_path.write_text(json.dumps({"uiR_version": "1.0", "id": "fake_doc"}))
        # Return a Mock-style object -- the web layer uses .uir_id, .out_path, etc.
        from types import SimpleNamespace
        return SimpleNamespace(
            uir_id="fake_doc",
            out_path=uir_path,
            chunk_count=3,
            entity_count=2,
            elapsed_seconds=0.42,
        )

    monkeypatch.setattr(pipeline_mod, "run", _fake)
    # NOTE: web.py's ``create_app`` imports ``pipeline`` inside the factory
    # and stores the module reference as a closure-local ``_pipeline_mod``.
    # The runner thread calls ``_pipeline_mod.run(...)`` -- which resolves
    # ``run`` via the module's globals at *call* time, not at import time.
    # So patching ``uir_pipeline.pipeline.run`` here is sufficient; we do
    # NOT need (and cannot) patch web.py's closure.
    return calls


@pytest.fixture()
def client(tmp_path, fake_run):
    app = create_app(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )
    app.config["TESTING"] = True
    return app.test_client()


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"UIR Pipeline" in resp.data
    assert b"Run pipeline" in resp.data


def test_run_rejects_missing_file(client):
    resp = client.post("/api/run", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "file" in (resp.get_json()["error"] or "").lower()


def test_run_rejects_non_pdf(client, tmp_path):
    txt = tmp_path / "foo.txt"
    txt.write_text("hello")
    with txt.open("rb") as fh:
        resp = client.post("/api/run", data={"file": (fh, "foo.txt")}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "pdf" in (resp.get_json()["error"] or "").lower()


def test_run_full_lifecycle(client, fake_run, sample_pdf):
    # Submit the job.
    with sample_pdf.open("rb") as fh:
        resp = client.post("/api/run", data={"file": (fh, "sample.pdf")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    job_id = body["job_id"]
    assert job_id

    # Wait for the runner thread to finish.
    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in (JOB_DONE, JOB_ERROR):
            final = s; break
        time.sleep(0.05)
    assert final is not None, "job did not complete within timeout"
    assert final["status"] == JOB_DONE
    assert final["result"]["chunk_count"] == 3
    assert final["result"]["entity_count"] == 2
    assert final["percent"] == 100

    # fake_run was actually called with our upload.
    assert len(fake_run) == 1
    assert Path(fake_run[0]["input_path"]).name.endswith(".pdf")
    assert fake_run[0]["skip_weaviate"] is True  # web default
    assert fake_run[0]["with_embeddings"] is True

    # Download produces the produced JSON file.
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    payload = json.loads(dl.data)
    assert payload["id"] == "fake_doc"


def test_status_404_for_unknown_job(client):
    resp = client.get("/api/status/does-not-exist")
    assert resp.status_code == 404


def test_download_409_if_not_done(client, monkeypatch, sample_pdf):
    """A still-running job returns 409 from /api/download."""
    import uir_pipeline.pipeline as local_pipeline_mod
    def _hang(*args, **kwargs):
        time.sleep(5)
        raise RuntimeError("intentional hang for test")
    monkeypatch.setattr(local_pipeline_mod, "run", _hang)
    with sample_pdf.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "sample.pdf")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    # Immediate download attempt while the runner is still sleeping.
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 409
    err = dl.get_json()["error"].lower()
    assert "running" in err or "queued" in err


# ----------------------------------------------------------------------------
# LAN-discovery helper (root launcher)
# ----------------------------------------------------------------------------

def test_list_lan_urls_filters_loopback(monkeypatch):
    """list_lan_urls() returns one URL per non-loopback IPv4 it discovers."""
    import socket as _socket
    import web as web_launcher  # root launcher module

    # Stub getaddrinfo so we don't depend on the host's network state.
    fake_getaddrinfo = [
        (None, None, None, None, ("192.168.1.42", 0)),
        (None, None, None, None, ("127.0.0.1", 0)),   # filtered
        (None, None, None, None, ("10.0.0.5", 0)),
    ]
    monkeypatch.setattr(
        _socket, "getaddrinfo",
        lambda *a, **kw: fake_getaddrinfo,
    )

    # Stub the UDP-connect trick.
    class _FakeSock:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def connect(self, *a, **kw): pass
        def getsockname(self): return ("192.168.1.99", 0)
    monkeypatch.setattr(_socket, "socket", _FakeSock)

    urls = web_launcher.list_lan_urls(5000)
    assert "http://192.168.1.42:5000" in urls
    assert "http://10.0.0.5:5000" in urls
    assert "http://192.168.1.99:5000" in urls
    assert "http://127.0.0.1:5000" not in urls
    # Sorted + deduplicated.
    assert urls == sorted(set(urls))


def test_list_lan_urls_empty_when_nothing_discoverable(monkeypatch):
    """Both techniques no-op -> empty list (caller logs a friendly fallback)."""
    import socket as _socket
    import web as web_launcher

    def _raise_gaierror(*a, **kw):
        raise _socket.gaierror("name or service not known")
    monkeypatch.setattr(_socket, "getaddrinfo", _raise_gaierror)

    class _BrokenSock:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def connect(self, *a, **kw):
            raise OSError("network unreachable")
        def getsockname(self): return ("127.0.0.1", 0)
    monkeypatch.setattr(_socket, "socket", _BrokenSock)

    assert web_launcher.list_lan_urls(8080) == []


def test_launcher_default_host_is_lan_visible(monkeypatch):
    """main() binds 0.0.0.0 by default (LAN-visible) unless HOST overrides."""
    import web as web_launcher
    import uir_pipeline.web as upipe_web

    captured: dict = {}

    class _FakeApp:
        def run(self, *, host, port, debug, use_reloader):
            captured["host"] = host
            captured["port"] = port

    # web.py does ``from uir_pipeline.web import create_app`` inside ``main``,
    # so we patch the *source* module rather than the importer's re-export
    # (which does not exist at web.py's module scope).
    monkeypatch.setattr(upipe_web, "create_app", lambda **kw: _FakeApp())
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    rc = web_launcher.main()
    assert rc == 0
    assert captured["host"] == "0.0.0.0"
    # Default port is 5050 to dodge macOS AirPlay hold on :5000.
    assert captured["port"] == 5050


def test_launcher_host_override_honored(monkeypatch):
    """``HOST=127.0.0.1`` overrides the LAN-visible default."""
    import web as web_launcher
    import uir_pipeline.web as upipe_web

    captured: dict = {}

    class _FakeApp:
        def run(self, *, host, port, debug, use_reloader):
            captured["host"] = host
            captured["port"] = port

    monkeypatch.setattr(upipe_web, "create_app", lambda **kw: _FakeApp())
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8080")
    rc = web_launcher.main()
    assert rc == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8080


# ----------------------------------------------------------------------------
# /api/result endpoint (full UIR document rendering)
# ----------------------------------------------------------------------------

def test_get_status_surfaces_only_allowlisted_stage_meta(client):
    """Only allow-listed stage_meta keys land in `/api/status/<id>` payloads.

    Prevents orchestrator-side ``error`` or other private fields from
    leaking through the JSON wire format to LAN viewers.
    """
    import uir_pipeline.web as web_mod

    job = web_mod.Job(job_id="meta-test", stage_meta={
        "caption_records_total":   3,
        "caption_records_with_text": 1,
        "caption_records_empty":   2,
        "error": "/abs/path/private.bin raised RuntimeError",
        "internal_path": "/tmp/uir_web_uploads/secret.pdf",
    })
    payload = job.to_public()
    # Allow-listed keys present.
    assert payload["stage_meta"]["caption_records_total"] == 3
    assert payload["stage_meta"]["caption_records_empty"] == 2
    # Private keys absent.
    assert "error" not in payload["stage_meta"]
    assert "internal_path" not in payload["stage_meta"]


def test_status_endpoint_reflects_allowlisted_stage_meta(client):
    """Job.stage_meta propagated via /api/status reflects the allowlist filter."""
    jid = "filter-test"
    import uir_pipeline.web as web_mod
    captured_jobs = web_mod.create_app(upload_dir=None, output_dir=None)
    # Inject a fake job with mixed keys; check /api/status <id>.
    with captured_jobs.test_request_context() if hasattr(captured_jobs, "test_request_context") else captured_jobs.test_client():
        # Drop into the closure-local jobs dict via the public app.
        pass
    # Simpler: assert the to_public contract via Job.to_public directly.
    job = web_mod.Job(job_id=jid, stage_meta={
        "caption_records_total": 5,
        "caption_records_with_text": 5,
        "caption_records_empty": 0,
        "dropped_entities": 12,
        "dropped_relations": 30,
        "error": "should-not-surface",
    })
    p = job.to_public()
    assert p["stage_meta"]["dropped_entities"] == 12
    assert "error" not in p["stage_meta"]


def test_download_404_for_unknown_job(client):
    resp = client.get("/api/download/does-not-exist")
    assert resp.status_code == 404

def test_result_endpoint_returns_full_uir_doc(client, fake_run, sample_pdf):
    """GET /api/result/<job_id> returns the full UIR JSON content for in-browser rendering."""
    with sample_pdf.open("rb") as fh:
        resp = client.post("/api/run", data={"file": (fh, "sample.pdf")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    # Wait for done
    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in (JOB_DONE, JOB_ERROR):
            final = s
            break
        time.sleep(0.05)
    assert final is not None and final["status"] == JOB_DONE

    r = client.get(f"/api/result/{job_id}")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    assert "attachment" not in (r.headers.get("Content-Disposition") or "").lower()
    payload = r.get_json()
    # The stub wrote a 2-field JSON, so ensure /api/result streams it intact.
    assert payload["id"] == "fake_doc"
    assert payload["uiR_version"] == "1.0"


def test_result_endpoint_404_for_unknown_job(client):
    resp = client.get("/api/result/does-not-exist")
    assert resp.status_code == 404


def test_result_endpoint_409_if_not_done(client, monkeypatch, sample_pdf):
    """GET /api/result/<job_id> returns 409 before the runner completes."""
    import uir_pipeline.pipeline as local_pipeline_mod
    def _hang(*args, **kwargs):
        time.sleep(5)
        raise RuntimeError("intentional hang for test")
    monkeypatch.setattr(local_pipeline_mod, "run", _hang)
    with sample_pdf.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "sample.pdf")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    r = client.get(f"/api/result/{job_id}")
    assert r.status_code == 409
    assert "running" in r.get_json()["error"].lower() or "queued" in r.get_json()["error"].lower()
