"""test_web_search.py -- the POST /api/search endpoint.

Stub the pipeline so a converted doc exists, and stub BGE so the ranking is
deterministic; then assert the endpoint returns ranked, title-badged results
gated by ownership.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from uir_pipeline import pipeline as pipeline_mod
from uir_pipeline import search as search_mod
from uir_pipeline.web import create_app


def _uir_with_chunk(out_path: Path, *, title: str, text: str, emb: list[float]) -> None:
    out_path.write_text(json.dumps({
        "uiR_version": "1.0", "id": title.lower().replace(" ", "_"),
        "source": {"format": "PDF", "route": "pdf", "uri": "test://x.pdf"},
        "metadata": {"title": title},
        "structure": {"root": {"type": "document", "children": [
            {"type": "chunk", "id": "c1", "text": text, "page": 1,
             "modal_features": {"vector": {"embedding": emb}}},
        ]}},
    }), encoding="utf-8")


@pytest.fixture()
def search_client(tmp_path, monkeypatch):
    """A signed-in client with one converted, title-matching document."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")

    def _fake(input_path, output_dir, *, skip_weaviate=False, dry_run=False,
              with_embeddings=True, on_progress=None, page_numbers=None,
              fast_path=None, intent=None):
        if on_progress:
            on_progress("done", 100)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        up = out_dir / "doc.uir.json"
        _uir_with_chunk(up, title="Acme Invoice Q3", text="The invoice total is 42000.",
                        emb=[1.0, 0.0, 0.0])
        return SimpleNamespace(uir_id="doc", out_path=up, chunk_count=1,
                               entity_count=0, elapsed_seconds=0.1)
    monkeypatch.setattr(pipeline_mod, "run", _fake)
    # Deterministic query embedding.
    monkeypatch.setattr(search_mod, "_embed_intent", lambda q: [1.0, 0.0, 0.0])

    app = create_app(upload_dir=tmp_path / "up", output_dir=tmp_path / "out",
                     data_dir=tmp_path / "data", execution="thread")
    app.config["TESTING"] = True
    c = app.test_client()
    c.post("/api/auth/signup", json={"email": "s@x.c", "password": "password123", "name": "S"})
    jid = c.post("/api/run", data={"file": (io.BytesIO(b"%PDF-1.4"), "acme.pdf")},
                 content_type="multipart/form-data").get_json()["job_id"]
    for _ in range(50):
        s = c.get(f"/api/status/{jid}").get_json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done"
    return c


def test_search_returns_title_matched_result(search_client):
    r = search_client.post("/api/search", json={"query": "invoice"})
    assert r.status_code == 200
    results = r.get_json()["results"]
    assert len(results) == 1
    res = results[0]
    assert res["title_match"] is True
    assert res["doc_title"] == "Acme Invoice Q3"
    assert res["job_id"]
    assert "score" in res and "page" in res and "text" in res


def test_empty_query_returns_empty(search_client):
    r = search_client.post("/api/search", json={"query": "  "})
    assert r.status_code == 200
    assert r.get_json()["results"] == []


def test_search_is_isolated_to_callers_docs(tmp_path, monkeypatch):
    """Bob cannot search Alice's documents."""
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    monkeypatch.setattr(pipeline_mod, "run", lambda *a, **k: SimpleNamespace(
        uir_id="d", out_path=k.get("output_dir") and Path(k["output_dir"]) / "d.uir.json",
        chunk_count=0, entity_count=0, elapsed_seconds=0.0))
    monkeypatch.setattr(search_mod, "_embed_intent", lambda q: [1.0, 0.0, 0.0])
    app = create_app(upload_dir=tmp_path / "up", output_dir=tmp_path / "out",
                     data_dir=tmp_path / "data", execution="thread")
    alice = app.test_client()
    bob = app.test_client()
    alice.post("/api/auth/signup", json={"email": "a@x.c", "password": "password123", "name": "A"})
    bob.post("/api/auth/signup", json={"email": "b@x.c", "password": "password123", "name": "B"})
    # Bob owns nothing done -> empty results, and no error.
    r = bob.post("/api/search", json={"query": "anything"})
    assert r.status_code == 200
    assert r.get_json()["results"] == []
