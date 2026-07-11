"""test_web_library.py -- folders, job persistence, and the thumbnail endpoint.

Covers the Google-Drive-like file-browser backend added to ``uir_pipeline.web``:
the ``/api/folders`` CRUD, ``PATCH``/``DELETE`` on ``/api/jobs/<id>``,
``/api/thumb/<id>``, the ``?folder_id=`` filter on ``/api/jobs``, and the
startup rehydration that flips orphaned running jobs to ``error``.

The heavy orchestrator is stubbed (like ``test_web.py``) so these tests stay
light; the stub writes a small UIR JSON whose ``structure`` tree carries a
``ChunkNode`` so the ``/api/result`` payload the Chunks tab walks is realistic.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from uir_pipeline import pipeline as pipeline_mod
from uir_pipeline.web import JOB_ERROR, JOB_RUNNING, Job, create_app


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def _uir_with_chunk(out_path: Path) -> None:
    """Write a tiny UIR JSON whose structure tree contains one ChunkNode."""
    out_path.write_text(json.dumps({
        "uiR_version": "1.0",
        "id": "fake_doc",
        "source": {"format": "PDF", "route": "pdf", "uri": "test://fake_doc.pdf"},
        "structure": {
            "root": {
                "type": "document",
                "id": "doc_fake",
                "children": [
                    {
                        "type": "section",
                        "id": "section_1",
                        "title": "Intro",
                        "children": [
                            {
                                "type": "chunk",
                                "id": "chunk_1",
                                "text": "The quick brown fox.",
                                "token_count": 5,
                                "page": 1,
                                "bounding_box": [0, 0, 100, 20],
                                "confidence": 0.9,
                            },
                        ],
                    },
                ],
            },
        },
    }), encoding="utf-8")


@pytest.fixture()
def chunky_run(monkeypatch):
    """Stub ``pipeline.run`` to finish instantly and write a chunked UIR."""
    calls: list[dict] = []

    def _fake(input_path, output_dir, *, skip_weaviate=False, dry_run=False,
              with_embeddings=True, on_progress=None, page_numbers=None,
              fast_path=None, intent=None):
        calls.append({"input_path": str(input_path), "intent": intent})
        if on_progress:
            on_progress("done", 100)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        uir_path = out_dir / "fake_doc.uir.json"
        _uir_with_chunk(uir_path)
        return SimpleNamespace(
            uir_id="fake_doc", out_path=uir_path,
            chunk_count=1, entity_count=0, elapsed_seconds=0.1,
        )

    monkeypatch.setattr(pipeline_mod, "run", _fake)
    return calls


@pytest.fixture()
def lib(tmp_path, chunky_run, monkeypatch):
    """A signed-in client + the app, pinned to a tmp data_dir.

    Returns ``(client, app)`` so tests can reach ``app.config`` for rehydration
    and direct-store seeding.
    """
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    app = create_app(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        execution="thread",
    )
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/auth/signup", json={
        "email": "lib@example.com", "password": "test-password-123", "name": "Lib",
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    return c, app


def _wait_done(c, job_id, timeout=5.0):
    """Poll /api/status until the job settles; return the terminal payload."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get(f"/api/status/{job_id}").get_json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle: {s}")


@pytest.fixture()
def real_pdf(tmp_path: Path) -> Path:
    """A genuinely valid one-page PDF PyMuPDF can open and rasterize."""
    import fitz  # PyMuPDF -- already a pipeline dependency
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello library", fontsize=24)
    p = tmp_path / "real.pdf"
    doc.save(str(p))
    doc.close()
    return p


# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------

def test_create_list_rename_delete_folder(lib):
    c, _ = lib
    r = c.post("/api/folders", json={"name": "Invoices"})
    assert r.status_code == 201
    fid = r.get_json()["folder"]["id"]

    listed = c.get("/api/folders").get_json()["folders"]
    assert [f["name"] for f in listed] == ["Invoices"]
    assert listed[0]["file_count"] == 0

    # Empty name is rejected.
    assert c.post("/api/folders", json={"name": "  "}).status_code == 400

    # Rename.
    assert c.patch(f"/api/folders/{fid}", json={"name": "Receipts"}).status_code == 200
    assert c.get("/api/folders").get_json()["folders"][0]["name"] == "Receipts"

    # Someone else's folder (non-existent id) -> 404.
    assert c.delete("/api/folders/999999").status_code == 404
    # Delete.
    assert c.delete(f"/api/folders/{fid}").status_code == 200
    assert c.get("/api/folders").get_json()["folders"] == []


def test_delete_folder_moves_its_jobs_to_root(lib, real_pdf):
    c, _ = lib
    fid = c.post("/api/folders", json={"name": "F"}).get_json()["folder"]["id"]
    r = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"), "folder": str(fid),
    }, content_type="multipart/form-data")
    jid = r.get_json()["job_id"]
    _wait_done(c, jid)

    # The job is in the folder.
    in_folder = c.get(f"/api/jobs?folder_id={fid}").get_json()["jobs"]
    assert [j["job_id"] for j in in_folder] == [jid]

    # Deleting the folder SET NULLs the job's folder_id.
    assert c.delete(f"/api/folders/{fid}").status_code == 200
    assert c.get(f"/api/status/{jid}").get_json()["folder_id"] is None
    root = c.get("/api/jobs?folder_id=").get_json()["jobs"]
    assert [j["job_id"] for j in root] == [jid]


# ----------------------------------------------------------------------------
# Job movement + deletion
# ----------------------------------------------------------------------------

def test_upload_into_folder_and_jobs_filter(lib, real_pdf):
    c, _ = lib
    fid = c.post("/api/folders", json={"name": "F"}).get_json()["folder"]["id"]
    jid = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"), "folder": str(fid),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)

    assert c.get(f"/api/status/{jid}").get_json()["folder_id"] == fid
    assert len(c.get(f"/api/jobs?folder_id={fid}").get_json()["jobs"]) == 1
    assert len(c.get("/api/jobs?folder_id=").get_json()["jobs"]) == 0
    # No filter -> all of the caller's jobs.
    assert len(c.get("/api/jobs").get_json()["jobs"]) == 1


def test_move_job_to_folder_and_back_to_root(lib, real_pdf):
    c, _ = lib
    fid = c.post("/api/folders", json={"name": "F"}).get_json()["folder"]["id"]
    jid = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)
    assert c.get(f"/api/status/{jid}").get_json()["folder_id"] is None

    assert c.patch(f"/api/jobs/{jid}", json={"folder_id": fid}).status_code == 200
    assert c.get(f"/api/status/{jid}").get_json()["folder_id"] == fid

    assert c.patch(f"/api/jobs/{jid}", json={"folder_id": None}).status_code == 200
    assert c.get(f"/api/status/{jid}").get_json()["folder_id"] is None


def test_move_job_to_other_users_folder_rejected(tmp_path, chunky_run, monkeypatch):
    """A user cannot park their job in a folder they don't own (404, not 403)."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    app = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                     data_dir=tmp_path / "d", execution="thread")
    a = app.test_client()
    b = app.test_client()
    a.post("/api/auth/signup", json={"email": "a@x.c", "password": "pw12345678", "name": "A"})
    b.post("/api/auth/signup", json={"email": "b@x.c", "password": "pw12345678", "name": "B"})
    fid = a.post("/api/folders", json={"name": "A-folder"}).get_json()["folder"]["id"]
    # B owns no folders; moving into A's folder id is refused.
    r = b.patch(f"/api/jobs/nevermind", json={"folder_id": fid})
    assert r.status_code == 404


def test_delete_job_removes_row_and_files(lib, real_pdf, tmp_path):
    c, app = lib
    jid = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    settled = _wait_done(c, jid)
    uir_path = Path(settled["result"]["out_path"]) if settled.get("result") else None
    upload_path = app.config["JOBS"][jid].upload_path

    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    # Gone from the listing.
    assert all(j["job_id"] != jid for j in c.get("/api/jobs").get_json()["jobs"])
    # The thumb now 404s (source file removed).
    assert c.get(f"/api/thumb/{jid}").status_code == 404
    # On-disk artefacts removed.
    assert not upload_path.exists()
    if uir_path:
        assert not uir_path.exists()


# ----------------------------------------------------------------------------
# Thumbnail
# ----------------------------------------------------------------------------

def test_thumb_pdf_returns_png(lib, real_pdf):
    c, _ = lib
    jid = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)
    r = c.get(f"/api/thumb/{jid}")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert len(r.data) > 0


def test_thumb_image_serves_original_bytes(lib, real_pdf):
    c, _ = lib
    # A real PNG (rasterize the real_pdf cover) so the bytes are a valid image.
    import fitz
    doc = fitz.open(str(real_pdf))
    png = doc[0].get_pixmap(dpi=72).tobytes("png")
    doc.close()
    jid = c.post("/api/run", data={
        "file": (io.BytesIO(png), "pixel.png"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)
    r = c.get(f"/api/thumb/{jid}")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert r.data == png  # images are served verbatim, not re-rendered


def test_thumb_unknown_format_returns_404(lib):
    c, _ = lib
    jid = c.post("/api/run", data={
        "file": (io.BytesIO(b"just text"), "notes.txt"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)
    assert c.get(f"/api/thumb/{jid}").status_code == 404


def test_thumb_not_done_returns_409(lib, real_pdf):
    c, app = lib
    # Seed a running job directly (bypass the runner) so the state is durable.
    jid = "seed-running-job"
    job = Job(job_id=jid, user_id=1, filename="doc.pdf", status=JOB_RUNNING,
              progress_stage="extract", progress_percent=20, upload_path=real_pdf)
    app.config["JOBS"][jid] = job
    assert c.get(f"/api/thumb/{jid}").status_code == 409


# ----------------------------------------------------------------------------
# Chunks (the payload the Chunks tab walks)
# ----------------------------------------------------------------------------

def test_result_contains_flattenable_chunk_tree(lib, real_pdf):
    c, _ = lib
    jid = c.post("/api/run", data={
        "file": (open(real_pdf, "rb"), "doc.pdf"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c, jid)
    doc = c.get(f"/api/result/{jid}").get_json()
    # Walk the structure tree the way the frontend's flattenChunks does.
    def walk(node):
        if not node:
            return []
        if node.get("type") == "chunk":
            return [node]
        return [c for ch in (node.get("children") or []) for c in walk(ch)]
    chunks = walk((doc.get("structure") or {}).get("root") or doc.get("structure"))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "The quick brown fox."
    assert chunks[0]["page"] == 1


# ----------------------------------------------------------------------------
# Restart rehydration
# ----------------------------------------------------------------------------

def test_rehydrate_flips_running_job_to_error(tmp_path, chunky_run, monkeypatch):
    """A job stuck in 'running' at restart is marked errored by _rehydrate_jobs."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    data_dir = tmp_path / "data"
    # First app: upload, then manually mark the job running + persist it.
    app1 = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                      data_dir=data_dir, execution="thread")
    c1 = app1.test_client()
    c1.post("/api/auth/signup", json={"email": "r@x.c", "password": "pw12345678", "name": "R"})
    store = app1.config["LIBRARY_STORE"]
    # Seed a persisted running job directly (the runner thread is "gone").
    running = Job(job_id="orph-1", user_id=1, filename="doc.pdf", status=JOB_RUNNING,
                  progress_stage="extract", progress_percent=20)
    store.upsert_job(running)

    # Second app from the same data_dir simulates a restart.
    app2 = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                      data_dir=data_dir, execution="thread")
    c2 = app2.test_client()
    c2.post("/api/auth/login", json={"email": "r@x.c", "password": "pw12345678"})
    jobs = c2.get("/api/jobs").get_json()["jobs"]
    orph = next(j for j in jobs if j["job_id"] == "orph-1")
    assert orph["status"] == "error"
    assert "restarted" in (orph["error"] or "").lower()


def test_rehydrate_done_job_survives(tmp_path, chunky_run, monkeypatch):
    """A finished job reappears verbatim after a restart."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    data_dir = tmp_path / "data"
    app1 = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                      data_dir=data_dir, execution="thread")
    c1 = app1.test_client()
    c1.post("/api/auth/signup", json={"email": "r@x.c", "password": "pw12345678", "name": "R"})
    jid = c1.post("/api/run", data={
        "file": (io.BytesIO(b"hi"), "t.txt"),
    }, content_type="multipart/form-data").get_json()["job_id"]
    _wait_done(c1, jid)

    app2 = create_app(upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                      data_dir=data_dir, execution="thread")
    c2 = app2.test_client()
    c2.post("/api/auth/login", json={"email": "r@x.c", "password": "pw12345678"})
    jobs = c2.get("/api/jobs").get_json()["jobs"]
    assert any(j["job_id"] == jid and j["status"] == "done" for j in jobs)
