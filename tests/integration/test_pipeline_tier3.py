"""tests/integration/test_pipeline_tier3.py -- Tier 3 e2e captioning smoke.

Proves :func:`uir_pipeline.pipeline.run` actually wires Florence-2
captions into :class:`ChunkNode` records with non-empty ``text`` AND
``modal_features.figure.image_b64`` set. Slow + tier3-marked so the
default ``pytest`` invocation excludes the test; run explicitly with::

    pytest -m slow tests/integration/test_pipeline_tier3.py
    pytest -m tier3 tests/integration/test_pipeline_tier3.py

Florence-2 is monkeypatched to a stub processor/model so the test never
downloads real weights. ``PyMuPDF`` is required (used by
:func:`uir_pipeline.caption.render_figure_crop`); the test uses
``pytest.importorskip`` so a missing dep yields a clean skip rather
than a confusing traceback.
"""
from __future__ import annotations

import base64
import subprocess
import sys
import time
from pathlib import Path

import pytest


FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "sample_pdfs" / "figure_rich.pdf"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ensure_fixture_present() -> None:
    """Generate the figure-rich fixture if missing.

    Wire-compatible with :func:`tests.integration.test_pipeline_smoke._ensure_fixture_present`
    -- cold-cache dev machines and CI runners both rely on the script's
    idempotent re-generation. We invoke ``generate_fixtures.py figure_rich``
    so only the missing profile is rebuilt (other fixtures stay templated).
    """
    if FIXTURE_PATH.is_file():
        return
    script = Path(__file__).parent.parent.parent / "scripts" / "generate_fixtures.py"
    subprocess.check_call([sys.executable, str(script), "figure_rich"])


def _stub_florence2(monkeypatch: pytest.MonkeyPatch, canned_caption: str) -> None:
    """Force :func:`uir_pipeline.caption._get_florence2` to return stubs.

    Mirrors the ``fake_run`` pattern from :mod:`tests.test_web` and
    :mod:`tests.test_caption` -- the pipeline never sees the real
    Florence-2 weights, which is what keeps the test fast.

    Stub classes live in :mod:`tests.stubs` (promoted out of
    :mod:`tests.test_caption` to avoid the cross-test-directory import
    code smell flagged by ``code-reviewer-minimax-m3`` 2026).
    """
    pytest.importorskip("transformers")  # noqa: F821  -- outer scope
    import uir_pipeline.caption as caption_mod
    from tests.stubs import _StubProcessor, _StubModel
    processor = _StubProcessor(canned_caption)
    model = _StubModel([canned_caption])
    monkeypatch.setattr(caption_mod, "_get_florence2", lambda **kw: (processor, model))


def _walk_structure_chunks(root: object) -> list[object]:
    """Return every :class:`ChunkNode` under :class:`Structure.root`.

    Walks the StructureNode tree depth-first so sections + their nested
    chunks are all inspected. A figure chunk nested inside a section
    (PLAN_TIER3 invariant: caption text becomes ChunkNode.text) is
    discovered here, which a flat iteration over root.children would miss.

    The ``uir_schema`` import is module-level so the cost is paid once
    per test process (not per call).
    """
    from uir_pipeline.uir_schema import ChunkNode
    found: list[object] = []
    stack: list[object] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, ChunkNode):
            found.append(node)
            continue
        children = getattr(node, "children", None) or []
        # ``StructureChild`` is a discriminated union; chunks live alongside
        # sections under root, so recurse into either type.
        for c in children:
            stack.append(c)
    return found


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.tier3
@pytest.mark.slow
def test_pipeline_emits_figure_chunk_with_image_b64_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: pipeline.run on figure_rich fixture emits exactly one
    figure ChunkNode with non-empty ``text`` AND a base64-decodable PNG
    crop persisting onto ``modal_features.figure.image_b64``.
    """
    pytest.importorskip("pymupdf")
    _ensure_fixture_present()
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"figure_rich fixture missing: {FIXTURE_PATH}")

    _stub_florence2(monkeypatch, canned_caption="a bar chart with five colored rectangles")

    from uir_pipeline.pipeline import run
    from uir_pipeline.uir_schema import UIRV1

    t0 = time.monotonic()
    result = run(
        FIXTURE_PATH,
        output_dir=tmp_path,
        skip_weaviate=True,
        with_embeddings=True,
    )
    elapsed = time.monotonic() - t0
    assert result.out_path.is_file()

    parsed = UIRV1.model_validate_json(result.out_path.read_text())

    all_chunks = _walk_structure_chunks(parsed.structure.root)
    assert all_chunks, "no ChunkNodes emitted (sanity regression -- pipeline emitted 0 chunks)"

    figure_chunks = [
        cn for cn in all_chunks
        if (cn.modal_features or {}).get("figure", {}).get("image_b64")
        and (cn.text or "").strip()
    ]
    # ``==1`` (not ``>=1``) so a duplicate-detection regression (e.g.
    # caption_figures_in_pdf looping twice) fails loudly here rather than
    # silently doubling the figure-chunk count downstream.
    assert len(figure_chunks) == 1, (
        f"expected exactly 1 figure chunk with image_b64 + non-empty text; "
        f"got {len(figure_chunks)} (total chunks: {len(all_chunks)})"
    )
    fig_chunk = figure_chunks[0]
    # Caption text flows from the stubbed Florence-2 -- spot-check the
    # stub-anchored phrase so a future change to caption_figures_in_pdf
    # that swallows the model output (e.g. wrong task= arg to
    # post_process_generation) is caught here.
    assert "bar chart" in fig_chunk.text.lower(), (
        f"figure chunk text does not contain the expected caption phrase; "
        f"got: {fig_chunk.text!r}"
    )
    # image_b64 decodes to a valid PNG (magic-byte check).
    decoded = base64.b64decode(fig_chunk.modal_features["figure"]["image_b64"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n", (
        "image_b64 payload did not decode to PNG bytes"
    )
    # Prompt + model fields propagated.
    fig_mf = fig_chunk.modal_features["figure"]
    assert fig_mf["caption_prompt"] == "<MORE_DETAILED_CAPTION>"
    from uir_pipeline.caption import MODEL_ID
    assert fig_mf["caption_model"] == MODEL_ID
    # Logged progress (Tier 3 fix #4 surfaced this via on_progress; we
    # can't inspect the callback here, but elapsed < 30s on the fixture
    # confirms the heavy path didn't engage).
    assert elapsed < 30.0, f"pipeline took {elapsed:.1f}s (expected <30 on stubbed stub)"


@pytest.mark.tier3
@pytest.mark.slow
def test_pipeline_emits_no_figure_chunk_on_text_only_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """:func:`uir_pipeline.pipeline.run` on the text-only flat_text fixture
    emits ZERO figure chunks (because pdfplumber detects no images).

    This is the negative-space test: figure-chunk plumbing can be wired
    without regressing text-only documents. Without this guard, a future
    bug that emits a synthetic figure chunk could slip through review.
    """
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pymupdf")
    from uir_pipeline.pipeline import run
    from uir_pipeline.uir_schema import UIRV1

    text_fixture = Path(__file__).parent.parent / "fixtures" / "sample_pdfs" / "flat_text.pdf"
    if not text_fixture.is_file():
        pytest.skip(f"text-only fixture missing: {text_fixture}")

    _stub_florence2(monkeypatch, canned_caption="unused")

    result = run(text_fixture, output_dir=tmp_path, skip_weaviate=True, with_embeddings=True)
    parsed = UIRV1.model_validate_json(result.out_path.read_text())

    all_chunks = _walk_structure_chunks(parsed.structure.root)
    figure_chunks = [
        cn for cn in all_chunks
        if (cn.modal_features or {}).get("figure") is not None
    ]
    assert figure_chunks == [], (
        f"text-only PDF unexpectedly emitted {len(figure_chunks)} figure chunks; "
        f"see chunk ids: {[cn.id for cn in figure_chunks]}"
    )
