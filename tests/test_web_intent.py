"""test_web_intent -- integration tests for the /api/run?intent= path.

What this file asserts:
    1. ``Job.intent`` is populated from the multipart ``intent`` form
       field on ``/api/run`` (and stays ``None`` when the field is empty).
    2. ``Job.to_public()`` surfaces the intent-filter summary ONLY when
       an intent was provided (reduce JSON noise on the LAN UI).
    3. ``/api/result/<job_id>`` serves the narrowed ``*.intent.uir.json``
       when intent was set; serves the full UIR when not.
    4. ``/api/download/<job_id>`` ALWAYS serves the full UIR (intentional
       -- the user uploaded for archive, not for the narrowest match).
    5. The intent-filter module is invoked exactly once per job via the
       runner thread -- not twice via subsequent polls.

These tests build on the existing ``test_web.py`` patterns: monkey-patch
``uir_pipeline.pipeline.run`` to return a deterministic UIR JSON whose
chunks contain known keywords so the filter is observable.
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
    create_app,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

_SAMPLE_CHUNKS = [
    "Multi-Head Attention. We project queries, keys and values to "
    "different linear projections of dimension 64.",
    "Scaled Dot-Product Attention is computed as a weighted sum over "
    "values, scaled by the square root of the dimensionality.",
    "Table 5 reports BLEU scores on the WMT 2014 English-German "
    "translation task. The Transformer base achieves 27.3.",
    "The encoder is composed of a stack of N=6 identical layers.",
    "We compare layer normalization variants in our ablation study.",
]


def _build_sample_uir() -> dict:
    """Return a minimal UIR v1 dict whose chunks contain ``attention`` /
    ``encoder`` / ``bleu`` so the intent filter has observable signals."""
    return {
        "uiR_version": "1.0",
        "id": "fake_doc",
        "modal_type": "document",
        "source": {
            "uri": "file:///tmp/fake_doc.pdf",
            "format": "PDF",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "checksum": "sha256:deadbeef",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        "metadata": {"title": "Fake", "page_count": len(_SAMPLE_CHUNKS), "language": "en"},
        "structure": {
            "type": "hierarchical",
            "root": {
                "id": "doc_fake",
                "type": "document",
                "title": "Fake",
                "page": 1,
                "children": [
                    {
                        "id": f"chunk_{i}",
                        "type": "chunk",
                        "text": t,
                        "token_count": len(t.split()),
                        "page": 1,
                        "bounding_box": [0, 0, 1000, 1000],
                        "confidence": 1.0,
                        "modal_features": {"text": {"token_count": len(t.split())}},
                    }
                    for i, t in enumerate(_SAMPLE_CHUNKS)
                ],
            },
        },
        "semantics": {"entities": [], "relationships": [], "topics": []},
        "provenance": {
            "extraction": {
                "model": "LayoutLMv3-heuristic",
                "version": "1.0",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            "normalization": {"version": "1.0", "timestamp": "2026-01-01T00:00:00Z"},
        },
    }


@pytest.fixture()
def fake_run_intent(monkeypatch, tmp_path: Path):
    """Stub :func:`pipeline.run` so the runner thread writes the sample
    UIR and the intent-filter module is exercised on it."""
    calls: list[dict] = []

    # ``intent`` was added to the real ``pipeline.run`` signature and web.py
    # has been passing it, but this stub never grew the parameter -- these
    # four tests were failing with TypeError before the console work began.
    def _fake(input_path, output_dir, *, skip_weaviate=False, dry_run=False,
              with_embeddings=True, on_progress=None, page_numbers=None,
              fast_path=None, intent=None):
        calls.append({
            "input_path": str(input_path),
            "skip_weaviate": skip_weaviate,
            "with_embeddings": with_embeddings,
            "fast_path": fast_path,
            "intent": intent,
        })
        for stage, pct in [
            ("ingest", 5), ("extract_text", 20), ("synthesize_ocr", 30),
            ("layout", 45), ("tables", 55), ("figure_caption", 60),
            ("chunk", 70), ("enrich", 80), ("embed", 90),
            ("assemble", 95), ("done", 100),
        ]:
            if on_progress:
                on_progress(stage, pct)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        uir_path = out_dir / "fake_doc.uir.json"
        uir_path.write_text(json.dumps(_build_sample_uir()))
        from types import SimpleNamespace
        return SimpleNamespace(
            uir_id="fake_doc",
            out_path=uir_path,
            chunk_count=len(_SAMPLE_CHUNKS),
            entity_count=0,
            elapsed_seconds=0.42,
        )

    monkeypatch.setattr(pipeline_mod, "run", _fake)
    return calls


@pytest.fixture()
def client_intent(tmp_path: Path, fake_run_intent, monkeypatch):
    """A signed-in client. ``/api/run`` requires a session; see test_web_auth.py."""
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
        json={"email": "tester@example.com", "password": "test-password-123"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return c


def _wait_for_done(client, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    final = None
    while time.monotonic() < deadline:
        s = client.get(f"/api/status/{job_id}").get_json()
        if s["status"] in (JOB_DONE, "error"):
            final = s
            break
        time.sleep(0.02)
    assert final is not None, "job did not complete within timeout"
    return final


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

class TestJobIntentField:
    """``Job.intent`` is correct given the multipart field."""
    def test_intent_form_sets_job_intent(self, client_intent, tmp_path, fake_run_intent):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf"), "intent": "show me attention"},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        assert final["status"] == JOB_DONE
        # Status payload surfaces intent summary.
        assert final["intent"] is not None
        assert final["intent"]["query"] == "show me attention"
        assert "attention" in final["intent"]["keywords"]
        assert fake_run_intent[-1]["fast_path"] == "docling"  # web pins Docling regardless of UIR_FAST_PATH env

    def test_blank_intent_keeps_job_intent_none(self, client_intent, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf"), "intent": "   "},  # whitespace-only
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        # No intent summary on the wire when intent was blank.
        assert final["intent"] is None

    def test_missing_intent_field_keeps_job_intent_none(
        self, client_intent, tmp_path,
    ):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            # No ``intent`` form field at all -- should behave like blank.
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        assert final["intent"] is None


class TestToPublicIntentShape:
    """``Job.to_public()`` correctly gates the intent block on the field."""
    def test_to_public_with_intent(self):
        from uir_pipeline.web import Job
        job = Job(
            job_id="x",
            intent="attention",
            intent_summary={
                "intent": "attention",
                "matched_chunks": 2,
                "total_chunks": 5,
                "keywords": ["attention"],
                "out_path": "/tmp/x.intent.uir.json",
                "no_match": False,
            },
        )
        p = job.to_public()
        assert p["intent"] == {
            "query": "attention",
            "matched_chunks": 2,
            "total_chunks": 5,
            "keywords": ["attention"],
            "no_match_fallback": False,
        }
        # Internal ``out_path`` MUST NOT leak to the LAN UI.
        assert "out_path" not in (p["intent"] or {})

    def test_to_public_without_intent_omits_block(self):
        from uir_pipeline.web import Job
        job = Job(job_id="x")
        p = job.to_public()
        assert p["intent"] is None


class TestApiResultServesFilteredUIR:
    """``/api/result/<id>`` returns the narrowed file when intent was set."""
    def test_filtered_json_matches_count(self, client_intent, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf"), "intent": "attention"},
                content_type="multipart/form-data",
            )
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        assert final["status"] == JOB_DONE

        r = client_intent.get(f"/api/result/{job_id}")
        assert r.status_code == 200
        payload = r.get_json()
        children = payload["structure"]["root"]["children"]
        # The sample UIR has 5 chunks; intent="attention" should narrow
        # it down to the 2 attention-mentioning chunks.
        assert len(children) >= 1
        assert len(children) <= 5
        assert payload["structure"]["root"]["intent_filter"]["keywords"] == ["attention"]

    def test_no_intent_serves_full_uirstream(self, client_intent, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf")},  # no intent
                content_type="multipart/form-data",
            )
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        assert final["status"] == JOB_DONE

        r = client_intent.get(f"/api/result/{job_id}")
        assert r.status_code == 200
        payload = r.get_json()
        children = payload["structure"]["root"]["children"]
        # No intent_filter block when intent was blank.
        assert "intent_filter" not in payload["structure"]["root"]
        # Full chunk count preserved.
        assert len(children) == 5


class TestApiDownloadServesFullUIR:
    """``/api/download/<id>`` always returns the FULL UIR -- even when an
    intent filter narrowed ``/api/result``."""
    def test_download_full_uirstream_when_intent_set(self, client_intent, tmp_path):
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
        with pdf.open("rb") as fh:
            resp = client_intent.post(
                "/api/run",
                data={"file": (fh, "x.pdf"), "intent": "attention"},
                content_type="multipart/form-data",
            )
        job_id = resp.get_json()["job_id"]
        final = _wait_for_done(client_intent, job_id)
        assert final["status"] == JOB_DONE

        r = client_intent.get(f"/api/download/{job_id}")
        assert r.status_code == 200
        payload = json.loads(r.data)
        children = payload["structure"]["root"]["children"]
        # Download = full 5 chunks (not 1-2 like /api/result would).
        assert len(children) == 5
