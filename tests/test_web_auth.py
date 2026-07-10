"""Auth, job-ownership, and chat-guard tests for the console backend.

These never touch the real pipeline: ``uir_pipeline.pipeline.run`` is
monkeypatched, so the suite does not pull in Docling / BGE / spaCy.

Patch ``run`` on the module object, not ``sys.modules``. ``web.create_app``
resolves the orchestrator with ``from uir_pipeline import pipeline``, which
reads the attribute already bound on the ``uir_pipeline`` package once any
other test has imported it. Swapping the ``sys.modules`` entry therefore has
no effect and the *real* pipeline runs -- a failure that only appears when
this file is run alongside ``test_web.py``.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from uir_pipeline import pipeline as pipeline_mod


@dataclass
class _StubResult:
    out_path: str
    umr_path: str
    uir_id: str = "doc_stub"
    chunk_count: int = 3
    entity_count: int = 1
    elapsed_seconds: float = 0.01


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A real Flask app with a fake orchestrator."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _fake_run(upload_path, *, output_dir, on_progress=None, intent=None, **_kw):
        uir = Path(output_dir) / "doc_stub.uir.json"
        uir.write_text(json.dumps({
            "uiR_version": "1.0",
            "id": "doc_stub",
            "metadata": {"title": "Stub Doc"},
            "source": {"format": "TEXT", "route": "text"},
            "structure": {"root": {"type": "root", "children": [
                {"id": "chunk_001", "type": "chunk", "page": 1,
                 "text": "The quick brown fox jumps over the lazy dog."},
            ]}},
        }), encoding="utf-8")
        umr = Path(output_dir) / "doc_stub.umr.md"
        umr.write_text("# Stub Doc\n\nThe quick brown fox.", encoding="utf-8")
        if on_progress:
            on_progress("chunk", 50)
        return _StubResult(out_path=str(uir), umr_path=str(umr))

    monkeypatch.setattr(pipeline_mod, "run", _fake_run)
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")

    from uir_pipeline.web import create_app
    application = create_app(
        upload_dir=tmp_path / "up",
        output_dir=out_dir,
        data_dir=tmp_path / "data",
        # in-process: the monkeypatched pipeline.run above cannot cross a
        # spawn() boundary. Crash isolation is covered in test_web_isolation.py.
        execution="thread",
    )
    application.config.update(TESTING=True)
    return application


def _signup(client, email="a@example.com", password="hunter2hunter2"):
    return client.post("/api/auth/signup", json={"email": email, "password": password, "name": "A"})


def _upload(client, name="doc.txt", body=b"hello world"):
    return client.post(
        "/api/run",
        data={"file": (io.BytesIO(body), name)},
        content_type="multipart/form-data",
    )


def _wait(client, job_id, tries=200):
    for _ in range(tries):
        r = client.get(f"/api/status/{job_id}")
        if r.get_json()["status"] in ("done", "error"):
            return r.get_json()
    raise AssertionError("job never settled")


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

def test_signup_creates_session(app):
    c = app.test_client()
    r = _signup(c)
    assert r.status_code == 200
    assert r.get_json()["user"]["email"] == "a@example.com"
    assert c.get("/api/auth/me").status_code == 200


def test_short_password_rejected(app):
    c = app.test_client()
    r = c.post("/api/auth/signup", json={"email": "b@example.com", "password": "short"})
    assert r.status_code == 400
    assert "at least 8" in r.get_json()["error"]


def test_duplicate_email_rejected(app):
    c = app.test_client()
    _signup(c)
    r = _signup(c)
    assert r.status_code == 400


def test_login_wrong_password_and_unknown_user_are_indistinguishable(app):
    c = app.test_client()
    _signup(c)
    wrong = c.post("/api/auth/login", json={"email": "a@example.com", "password": "nope-nope-nope"})
    unknown = c.post("/api/auth/login", json={"email": "ghost@example.com", "password": "nope-nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.get_json()["error"] == unknown.get_json()["error"]


def test_logout_clears_session(app):
    c = app.test_client()
    _signup(c)
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/auth/me").status_code == 401


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("get", "/api/jobs"),
    ("post", "/api/run"),
    ("get", "/api/status/x"),
    ("get", "/api/result/x"),
    ("get", "/api/umr/x"),
    ("get", "/api/download/x"),
    ("post", "/api/chat"),
])
def test_routes_require_auth(app, method, path):
    c = app.test_client()
    r = getattr(c, method)(path)
    assert r.status_code == 401, f"{method.upper()} {path} was reachable anonymously"


def test_health_is_public(app):
    assert app.test_client().get("/api/health").status_code == 200


def test_index_renders_console(app):
    r = app.test_client().get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "MonadLabs Console" in body
    assert "console/app.jsx" in body


# ---------------------------------------------------------------------------
# job ownership -- the reason auth is not merely decorative
# ---------------------------------------------------------------------------

def test_upload_uses_the_real_filename(app):
    c = app.test_client()
    _signup(c)
    job_id = _upload(c, name="contract.pdf", body=b"%PDF-1.4 fake").get_json()["job_id"]
    assert c.get(f"/api/status/{job_id}").get_json()["filename"] == "contract.pdf"


def test_other_user_cannot_read_your_job(app):
    alice = app.test_client()
    _signup(alice, "alice@example.com")
    job_id = _upload(alice).get_json()["job_id"]
    _wait(alice, job_id)

    bob = app.test_client()
    _signup(bob, "bob@example.com")
    for path in (f"/api/status/{job_id}", f"/api/result/{job_id}",
                 f"/api/umr/{job_id}", f"/api/download/{job_id}"):
        r = bob.get(path)
        # 404, not 403: a 403 would confirm the job id exists.
        assert r.status_code == 404, f"{path} leaked to a non-owner ({r.status_code})"


def test_jobs_listing_is_scoped_to_owner(app):
    alice = app.test_client()
    _signup(alice, "alice@example.com")
    _upload(alice)

    bob = app.test_client()
    _signup(bob, "bob@example.com")
    assert bob.get("/api/jobs").get_json()["jobs"] == []
    assert len(alice.get("/api/jobs").get_json()["jobs"]) == 1


def test_pipeline_runs_and_serves_artifacts(app):
    c = app.test_client()
    _signup(c)
    job_id = _upload(c).get_json()["job_id"]
    final = _wait(c, job_id)
    assert final["status"] == "done", final.get("error")

    assert "Stub Doc" in c.get(f"/api/umr/{job_id}").get_data(as_text=True)
    assert c.get(f"/api/result/{job_id}").get_json()["id"] == "doc_stub"
    assert c.get(f"/api/download/{job_id}").status_code == 200


def test_unsupported_extension_rejected(app):
    c = app.test_client()
    _signup(c)
    r = _upload(c, name="virus.exe", body=b"MZ")
    assert r.status_code == 400
    assert "unsupported file type" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def test_chat_with_no_documents_does_not_call_a_model(app):
    c = app.test_client()
    _signup(c)
    r = c.post("/api/chat", json={"message": "what is this about?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["grounded"] is False
    assert body["citations"] == []
    assert "haven't converted any documents" in body["answer"]


def test_chat_requires_a_message(app):
    c = app.test_client()
    _signup(c)
    assert c.post("/api/chat", json={"message": "  "}).status_code == 400


def test_chat_never_retrieves_another_users_document(app, monkeypatch):
    """Bob asks a question; retrieval must not see Alice's uploaded document."""
    alice = app.test_client()
    _signup(alice, "alice@example.com")
    _wait(alice, _upload(alice).get_json()["job_id"])

    seen: list = []
    from uir_pipeline import chat as chat_mod
    monkeypatch.setattr(chat_mod, "retrieve", lambda paths, q, **kw: seen.append(list(paths)) or [])

    bob = app.test_client()
    _signup(bob, "bob@example.com")
    r = bob.post("/api/chat", json={"message": "what does the fox do?"})
    assert r.status_code == 200
    # Bob owns nothing done -> we short-circuit before retrieval entirely.
    assert seen == []
    assert r.get_json()["grounded"] is False


def test_chat_retrieves_only_own_documents(app, monkeypatch):
    alice = app.test_client()
    _signup(alice, "alice@example.com")
    _wait(alice, _upload(alice).get_json()["job_id"])

    captured: dict = {}

    def _fake_retrieve(paths, q, **kw):
        captured["paths"] = list(paths)
        return []

    from uir_pipeline import chat as chat_mod
    monkeypatch.setattr(chat_mod, "retrieve", _fake_retrieve)
    monkeypatch.setattr(chat_mod, "answer", lambda q, ctx, **kw: {
        "success": True, "answer": "stub", "citations": ctx, "grounded": bool(ctx),
    })

    r = alice.post("/api/chat", json={"message": "what does the fox do?"})
    assert r.status_code == 200
    assert len(captured["paths"]) == 1


def test_chat_surfaces_model_failure_instead_of_faking_an_answer(app, monkeypatch):
    c = app.test_client()
    _signup(c)
    _wait(c, _upload(c).get_json()["job_id"])

    from uir_pipeline import chat as chat_mod
    monkeypatch.setattr(chat_mod, "retrieve", lambda paths, q, **kw: [
        {"doc_id": "d", "doc_title": "T", "chunk_id": "c1", "page": 1, "text": "x", "score": 0.9},
    ])
    monkeypatch.setattr(chat_mod, "answer", lambda q, ctx, **kw: {
        "success": False, "error": "Chat model call failed (HTTP 503).",
        "answer": "", "citations": ctx, "model": "m", "usage": {},
    })

    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert "Chat model call failed" in r.get_json()["error"]
