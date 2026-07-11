"""test_search.py -- semantic + title-priority passage search.

Covers :func:`uir_pipeline.search.search` without loading BGE: the embed
entry point is stubbed in ``search``'s own namespace (it imports ``_embed_intent``
by name), and chunk embeddings are hand-set so cosine scores are deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uir_pipeline import search as S


def _write_uir(path: Path, *, title: str, chunks: list[dict]) -> None:
    path.write_text(json.dumps({
        "uiR_version": "1.0", "id": title.lower().replace(" ", "_"),
        "metadata": {"title": title},
        "structure": {"root": {"type": "document", "children": chunks}},
    }), encoding="utf-8")


def _chunk(cid: str, text: str, emb: list[float], page: int = 1) -> dict:
    return {"type": "chunk", "id": cid, "text": text, "page": page,
            "modal_features": {"vector": {"embedding": emb}}}


@pytest.fixture()
def stub_embed(monkeypatch):
    """Make the query embedding a fixed vector the chunk embeddings dot against."""
    monkeypatch.setattr(S, "_embed_intent", lambda q: [1.0, 0.0, 0.0])


def test_title_match_boosts_above_equal_content(tmp_path, stub_embed):
    """Two docs with identical content score; the title-matching one ranks first."""
    p1 = tmp_path / "d1.uir.json"
    p2 = tmp_path / "d2.uir.json"
    _write_uir(p1, title="Invoices Q3", chunks=[_chunk("c1", "x", [1.0, 0.0, 0.0])])
    _write_uir(p2, title="Other Notes", chunks=[_chunk("c2", "x", [1.0, 0.0, 0.0])])
    docs = [{"job_id": "j1", "uir_path": str(p1), "filename": "invoices-q3.pdf"},
            {"job_id": "j2", "uir_path": str(p2), "filename": "other.txt"}]
    res = S.search(docs, "invoices", top_k=5)
    assert res[0]["job_id"] == "j1"
    assert res[0]["title_match"] is True
    assert res[0]["score"] == pytest.approx(1.0 + S.TITLE_BOOST)
    assert res[1]["job_id"] == "j2"
    assert res[1]["title_match"] is False
    assert res[1]["score"] == pytest.approx(1.0)


def test_strong_content_outranks_title_only_match(tmp_path, stub_embed):
    """A title match with zero content signal still appears (title priority),
    but a high-content passage ranks above it."""
    p1 = tmp_path / "d1.uir.json"
    p2 = tmp_path / "d2.uir.json"
    _write_uir(p1, title="Invoices Q3",
               chunks=[_chunk("c1", "unrelated", [0.0, 1.0, 0.0])])  # content 0
    _write_uir(p2, title="Random",
               chunks=[_chunk("c2", "the invoice total", [1.0, 0.0, 0.0])])  # content 1.0
    docs = [{"job_id": "j1", "uir_path": str(p1), "filename": "invoices-q3.pdf"},
            {"job_id": "j2", "uir_path": str(p2), "filename": "rand.txt"}]
    res = S.search(docs, "invoices", top_k=5)
    # c2 (content 1.0, no title) ranks above c1 (content 0 + 0.30 title boost).
    assert res[0]["job_id"] == "j2" and res[0]["score"] == pytest.approx(1.0)
    assert res[1]["job_id"] == "j1" and res[1]["title_match"] is True
    assert res[1]["score"] == pytest.approx(S.TITLE_BOOST)


def test_zero_signal_non_title_chunk_is_dropped(tmp_path, stub_embed):
    p1 = tmp_path / "d1.uir.json"
    _write_uir(p1, title="Nothing Here",
               chunks=[_chunk("c1", "unrelated text", [0.0, 1.0, 0.0])])
    docs = [{"job_id": "j1", "uir_path": str(p1), "filename": "n.txt"}]
    assert S.search(docs, "invoices", top_k=5) == []


def test_empty_query_or_docs_returns_empty(tmp_path, stub_embed):
    assert S.search([], "anything") == []
    p1 = tmp_path / "d1.uir.json"
    _write_uir(p1, title="X", chunks=[_chunk("c1", "x", [1.0, 0.0, 0.0])])
    docs = [{"job_id": "j1", "uir_path": str(p1), "filename": "x.txt"}]
    assert S.search(docs, "   ") == []


def test_results_carry_job_id_and_title_match_flag(tmp_path, stub_embed):
    p1 = tmp_path / "d1.uir.json"
    _write_uir(p1, title="Acme Invoice", chunks=[_chunk("c1", "total 42", [1.0, 0.0, 0.0], page=3)])
    docs = [{"job_id": "job-abc", "uir_path": str(p1), "filename": "acme.pdf"}]
    res = S.search(docs, "invoice", top_k=5)
    assert len(res) == 1
    r = res[0]
    assert r["job_id"] == "job-abc"
    assert r["title_match"] is True
    assert r["page"] == 3
    assert r["doc_title"] == "Acme Invoice"
    assert "score" in r and "text" in r and "chunk_id" in r
