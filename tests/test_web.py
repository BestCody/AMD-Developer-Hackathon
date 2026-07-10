"""test_web.py -- Phase N Flask front-end tests.

The web layer is intentionally thin: we stub the heavy orchestrator so
tests don't pull in BGE / spaCy / pdfplumber. Integration with the real
``uir_pipeline.pipeline.run`` is covered by the existing end-to-end
``tests/integration/test_pipeline_smoke.py``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from uir_pipeline import pipeline as pipeline_mod
from uir_pipeline.web import (
    JOB_DONE,
    JOB_ERROR,
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

    def _fake(input_path, output_dir, *, skip_weaviate=False, dry_run=False, with_embeddings=True, on_progress=None, page_numbers=None, fast_path=None, intent=None):
        calls.append({
            "input_path": str(input_path),
            "skip_weaviate": skip_weaviate,
            "with_embeddings": with_embeddings,
            "fast_path": fast_path,
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
        # PLAN §17 §Multi-format follow-up: include the top-level
        # ``source`` field so the web layer's source-format/route
        # surfacing + the front-end format-pill code path are
        # exercised end-to-end. The real orchestrator writes
        # ``source.format`` and ``source.route`` (see
        # :class:`uir_pipeline.uir_schema.Source`); we mirror that
        # here so the test contract is faithful.
        uir_path.write_text(json.dumps({
            "uiR_version": "1.0",
            "id": "fake_doc",
            "source": {
                "format": "PDF",
                "route": "pdf",
                "uri": "test://fake_doc.pdf",
            },
        }))
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
def client(tmp_path, fake_run, monkeypatch):
    """A *signed-in* test client.

    Every job route now requires a session (see ``uir_pipeline.auth``), so
    the fixture registers a throwaway account and returns the client holding
    its cookie. The tests below exercise pipeline behaviour, not the auth
    boundary -- that lives in ``tests/test_web_auth.py``.

    ``data_dir`` is pinned to ``tmp_path`` so the SQLite user table and the
    generated ``.secret_key`` never land in the real repo's ``data/``.
    """
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    app = create_app(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        # in-process: these tests monkeypatch pipeline.run, which a spawned
        # child process cannot inherit. Crash isolation is covered in
        # tests/test_web_isolation.py.
        execution="thread",
    )
    app.config["TESTING"] = True
    c = app.test_client()
    resp = c.post(
        "/api/auth/signup",
        json={"email": "tester@example.com", "password": "test-password-123", "name": "Tester"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return c


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


def test_index_renders(client):
    """``/`` now serves the MonadLabs console SPA, not the old tester page."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"MonadLabs Console" in resp.data
    assert b"console/app.jsx" in resp.data


def test_run_rejects_missing_file(client):
    resp = client.post("/api/run", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "file" in (resp.get_json()["error"] or "").lower()


def test_run_rejects_unsupported_format(client, tmp_path):
    """PLAN §17 §Multi-format follow-up: the upload form now accepts a
    dozen+ formats via :data:`format_router.SUPPORTED_EXTENSIONS` --
    we test rejection of a *truly* unsupported extension so the test
    is a meaningful regression guard against accidentally widening
    the allow-list further than the orchestrator can ingest.
    """
    bin_file = tmp_path / "foo.xyz"
    bin_file.write_bytes(b"\x00\x01\x02\x03")
    with bin_file.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "foo.xyz")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400
    assert "unsupported" in (resp.get_json()["error"] or "").lower()


def test_run_accepts_text_format(client, fake_run, tmp_path):
    """PLAN §17 §Multi-format follow-up: a ``.txt`` upload should be
    accepted (routed through the TEXT lane) and produce a successful
    job -- not the legacy 400 response the PDF-only code returned.
    """
    txt = tmp_path / "notes.txt"
    txt.write_text("hello world\nthis is a test document for the text route.\n")
    with txt.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "notes.txt")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["job_id"]
    # The runner should pick up the .txt extension on disk -- this is
    # the contract the format_router relies on for the extension
    # fallback (text / code / image files share a "no magic" signature).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{body['job_id']}").get_json()
        if s["status"] in (JOB_DONE, JOB_ERROR):
            break
        time.sleep(0.05)
    assert s["status"] == JOB_DONE
    # The runner should have been invoked with the .txt upload, with
    # the original extension preserved on disk so the format_router
    # extension-fallback still works. This is the multi-format
    # contract end-to-end: accepted -> routed -> called.
    assert any(
        call["input_path"].endswith(".txt") for call in fake_run
    ), f"fake_run not called with .txt; calls={fake_run!r}"


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
            final = s
            break
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
    assert fake_run[0]["fast_path"] == "docling"  # web pins Docling regardless of UIR_FAST_PATH env

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


class _FakeApp:
    """Stands in for the Flask app; `config` is read by register_worker_shutdown."""

    def __init__(self):
        self.config: dict = {"PIPELINE_WORKER": None}
        self.captured: dict = {}

    def run(self, *, host, port, debug, use_reloader):
        self.captured["host"] = host
        self.captured["port"] = port


def _patch_launcher(monkeypatch) -> _FakeApp:
    # web.py does ``from uir_pipeline.web import create_app`` inside ``main``,
    # so we patch the *source* module rather than the importer's re-export
    # (which does not exist at web.py's module scope).
    import uir_pipeline.web as upipe_web

    app = _FakeApp()
    monkeypatch.setattr(upipe_web, "create_app", lambda **kw: app)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("SESSION_COOKIE_SECURE", raising=False)
    monkeypatch.delenv("UIR_ALLOW_INSECURE_BIND", raising=False)
    return app


def test_launcher_default_host_is_loopback(monkeypatch):
    """main() binds 127.0.0.1 by default.

    It used to default to 0.0.0.0 while the docstring claimed "MVP: no auth".
    Once accounts landed, a LAN bind over plain HTTP put passwords and session
    cookies on the wire in cleartext.
    """
    import web as web_launcher

    app = _patch_launcher(monkeypatch)
    assert web_launcher.main() == 0
    assert app.captured["host"] == "127.0.0.1"
    # Default port is 5050 to dodge macOS AirPlay hold on :5000.
    assert app.captured["port"] == 5050


def test_launcher_host_and_port_overrides_honored(monkeypatch):
    import web as web_launcher

    app = _patch_launcher(monkeypatch)
    monkeypatch.setenv("HOST", "localhost")
    monkeypatch.setenv("PORT", "8080")
    assert web_launcher.main() == 0
    assert app.captured["host"] == "localhost"
    assert app.captured["port"] == 8080


def test_launcher_refuses_a_lan_bind_over_plain_http(monkeypatch):
    """The guard must run in the launcher, not only in `python -m`."""
    import web as web_launcher

    app = _patch_launcher(monkeypatch)
    monkeypatch.setenv("HOST", "0.0.0.0")
    with pytest.raises(SystemExit, match="cleartext"):
        web_launcher.main()
    assert "host" not in app.captured, "server must not start"


def test_launcher_allows_a_lan_bind_behind_tls(monkeypatch):
    import web as web_launcher

    app = _patch_launcher(monkeypatch)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "1")
    assert web_launcher.main() == 0
    assert app.captured["host"] == "0.0.0.0"


def test_launcher_refuses_before_building_the_app(monkeypatch):
    """A refused bind must not pay for create_app (which spawns nothing, but
    would open the SQLite user store and resolve the secret key)."""
    import uir_pipeline.web as upipe_web
    import web as web_launcher

    built: list[int] = []

    def _boom(**_kw):
        built.append(1)
        raise AssertionError("create_app must not run on a refused bind")

    monkeypatch.setattr(upipe_web, "create_app", _boom)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("SESSION_COOKIE_SECURE", raising=False)
    monkeypatch.delenv("UIR_ALLOW_INSECURE_BIND", raising=False)
    with pytest.raises(SystemExit):
        web_launcher.main()
    assert built == []


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
    # NOTE: this test asserts the ``Job.to_public`` allowlist directly; it
    # never needed an app. Building one here used to be harmless, but
    # ``create_app`` now provisions a SQLite user store and a secret key,
    # so instantiating it with default paths would write into the repo's
    # real ``data/`` directory during a test run.
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


def test_status_payload_handles_missing_source(client, sample_pdf, monkeypatch):
    """Negative test for PLAN §17: a UIR JSON without a ``source``
    field must result in ``source_format='UNKNOWN'`` /
    ``source_route='unknown'`` rather than a 500 or KeyError. Guards
    the ``or 'UNKNOWN'`` / ``or 'unknown'`` fallback in ``_runner``
    against future regressions (e.g. a refactor that switches to
    ``source_meta['format']`` and crashes on missing keys).
    """
    import uir_pipeline.pipeline as pipeline_mod

    def _fake_no_source(input_path, output_dir, *, skip_weaviate=False, dry_run=False, with_embeddings=True, on_progress=None, page_numbers=None, fast_path=None, intent=None):
        from types import SimpleNamespace
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        uir_path = out_dir / "fake_doc.uir.json"
        # Deliberately omit the ``source`` field.
        uir_path.write_text(json.dumps({"uiR_version": "1.0", "id": "fake_doc"}))
        return SimpleNamespace(
            uir_id="fake_doc", out_path=uir_path,
            chunk_count=3, entity_count=2, elapsed_seconds=0.42,
        )

    monkeypatch.setattr(pipeline_mod, "run", _fake_no_source)

    with sample_pdf.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "sample.pdf")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in (JOB_DONE, JOB_ERROR):
            final = s
            break
        time.sleep(0.05)
    assert final is not None and final["status"] == JOB_DONE
    assert final["result"]["source_format"] == "UNKNOWN"
    assert final["result"]["source_route"] == "unknown"


def test_status_payload_surfaces_source_format_and_route(client, fake_run, sample_pdf):
    """PLAN §17 §Multi-format follow-up: the runner populates
    ``result.source_format`` + ``result.source_route`` by reading the
    just-written UIR JSON's ``source`` field so the front-end format
    pill can render without a second fetch. The fake_run fixture
    writes a UIR with ``source.format='PDF'`` / ``source.route='pdf'``;
    we assert the same fields round-trip through ``/api/status``.
    """
    with sample_pdf.open("rb") as fh:
        resp = client.post(
            "/api/run",
            data={"file": (fh, "sample.pdf")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in (JOB_DONE, JOB_ERROR):
            final = s
            break
        time.sleep(0.05)
    assert final is not None and final["status"] == JOB_DONE
    assert final["result"]["source_format"] == "PDF"
    assert final["result"]["source_route"] == "pdf"


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
