"""test_caption.py -- Tier 3 / Phase O image-awareness unit tests.

The unit tests stub ``caption._get_florence2`` so Florence-2 weights never
download during ``pytest`` (per PLAN_TIER3.md risk 6 + the existing
OCR-leader stub pattern from ``tests/test_web.py``).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# Florence-2 processor/model stubs + a Pillow synthetic-image helper live
# in :mod:`tests.stubs` (promoted out of this file -- cross-test-directory
# imports of test-private classes were a code smell that broke the
# ``tests/integration/test_pipeline_tier3.py`` review). The stubs keep the
# exact contracts the production code in :mod:`uir_pipeline.caption`
# consumes; anything that does ``tensor.shape`` / ``.tolist()`` /
# tensorial arithmetic will FAIL loudly under stub mode (by design).

from tests.stubs import (  # noqa: F401  -- _install_stub references _StubProcessor/_StubModel; _StubInputs/_make_pil_stub are re-exported for downstream tests that import from tests.test_caption directly
    _StubInputs,
    _StubModel,
    _StubProcessor,
    _make_pil_stub,
)


def _install_stub(monkeypatch, canned: str | list[str]):
    """Force :func:`caption._get_florence2` to return a (processor, model) stub.

    Mirrors the ``fake_run`` pattern in ``tests/test_web.py``.
    """
    import uir_pipeline.caption as caption_mod
    processor = _StubProcessor(canned)
    model = _StubModel(canned if isinstance(canned, list) else [canned])
    monkeypatch.setattr(caption_mod, "_get_florence2", lambda **kw: (processor, model))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_caption_images_stub_round_trip_single(monkeypatch):
    """caption_images returns one string per input image (stub mode)."""
    pytest.importorskip("PIL")
    import uir_pipeline.caption as caption_mod

    _install_stub(monkeypatch, canned="a bar chart with three rows")
    out = caption_mod.caption_images([_make_pil_stub(64, 64)])
    assert len(out) == 1
    assert out[0] == "a bar chart with three rows"


def test_caption_images_stub_round_trip_batched(monkeypatch):
    """caption_images batches multiple images; order matches input."""
    pytest.importorskip("PIL")
    import uir_pipeline.caption as caption_mod

    _install_stub(monkeypatch, canned=["caption a", "caption b", "caption c"])
    out = caption_mod.caption_images(
        [_make_pil_stub(64, 64) for _ in range(3)],
    )
    assert out == ["caption a", "caption b", "caption c"]


def test_caption_images_empty_list_returns_empty():
    """caption_images([]) returns [] without ever loading the model."""
    import uir_pipeline.caption as caption_mod
    assert caption_mod.caption_images([]) == []


def test_caption_images_fail_soft_on_load_failure(monkeypatch):
    """If _get_florence2 raises, caption_images returns [''] * len(images)."""
    pytest.importorskip("PIL")
    import uir_pipeline.caption as caption_mod

    def _raise(**kw):
        raise OSError("florence-2 weights missing -- offline dev box")

    monkeypatch.setattr(caption_mod, "_get_florence2", _raise)
    out = caption_mod.caption_images([_make_pil_stub(64, 64), _make_pil_stub(64, 64)])
    assert out == ["", ""]


def test_caption_images_fail_soft_on_generate_failure(monkeypatch):
    """If `model.generate` raises, caption_images returns [''] * len."""
    pytest.importorskip("PIL")
    import uir_pipeline.caption as caption_mod

    class _BoomProcessor:
        def __call__(self, *, text, images, return_tensors, padding=True):
            return _StubInputs(text=text, images=images)

        def batch_decode(self, *a, **kw):
            return []

        def post_process_generation(self, *a, **kw):
            return {caption_mod.DEFAULT_PROMPT: ""}

    class _BoomModel:
        device = "cpu"
        dtype = None
        def generate(self, **kw):
            raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(
        caption_mod, "_get_florence2",
        lambda **kw: (_BoomProcessor(), _BoomModel()),
    )
    out = caption_mod.caption_images([_make_pil_stub(64, 64)])
    assert out == [""]


def test_detect_figure_regions_filters_tiny_bboxes(tmp_path):
    """detect_figure_regions should drop bboxes < min_dim_px in caption_figures_in_pdf."""
    pytest.importorskip("PIL")
    pytest.importorskip("pymupdf")
    from PIL import Image

    # Render a tiny test PDF with a small image (decorative dot) via reportlab.
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas

    pdf = tmp_path / "tiny_dot.pdf"
    pil = _make_pil_stub(10, 10)
    pil_path = tmp_path / "dot.png"
    pil.save(pil_path)
    c = rl_canvas.Canvas(str(pdf), pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "Tiny dot test")
    # reportlab's drawImage needs a file path (not a PIL.Image) on this version.
    c.drawImage(str(pil_path), 72, 600, 10, 10)
    c.showPage()
    c.save()

    from uir_pipeline.caption import caption_figures_in_pdf, MIN_FIGURE_DIM_PX
    # 10x10 < MIN_FIGURE_DIM_PX (50) -> the figure should be filtered out.
    out = caption_figures_in_pdf(pdf)
    assert out == []


def test_detect_figure_regions_returns_shape():
    """detect_figure_regions_from_docling reads `pictures` off a DoclingResult.

    Renamed from `detect_figure_regions` in ea0e9ad, which also changed the
    argument from a PDF path to an already-converted DoclingResult -- so the
    function no longer runs the 2 GB converter itself.
    """
    from types import SimpleNamespace

    import uir_pipeline.caption as caption_mod

    empty = SimpleNamespace(pictures=[])
    assert caption_mod.detect_figure_regions_from_docling(empty) == []

    pic = {"page": 1, "bbox": (72, 600, 192, 681),
           "bbox_pixel": (72, 600, 192, 681), "kind": "picture"}
    dr = SimpleNamespace(pictures=[pic])
    assert caption_mod.detect_figure_regions_from_docling(dr) == [pic]


def test_detect_figure_regions_honours_page_numbers():
    from types import SimpleNamespace

    from uir_pipeline.caption import detect_figure_regions_from_docling

    dr = SimpleNamespace(pictures=[
        {"page": 1, "bbox": (0, 0, 100, 100), "kind": "picture"},
        {"page": 3, "bbox": (0, 0, 100, 100), "kind": "picture"},
    ])
    assert [p["page"] for p in detect_figure_regions_from_docling(dr, page_numbers=[3])] == [3]


def test_encode_image_b64_round_trip():
    """encode_image_b64 produces a valid base64 string for a valid PIL Image."""
    pytest.importorskip("PIL")
    from uir_pipeline.caption import encode_image_b64
    import base64

    img = _make_pil_stub(8, 8)
    encoded = encode_image_b64(img)
    assert isinstance(encoded, str) and len(encoded) > 0
    decoded = base64.b64decode(encoded)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_encode_image_b64_handles_none():
    from uir_pipeline.caption import encode_image_b64
    assert encode_image_b64(None) is None


def test_caption_figures_in_pdf_smoke_stub(monkeypatch, tmp_path):
    """caption_figures_in_pdf end-to-end with stubbed Florence-2 + PyMuPDF.

    Verifies the public API returns properly-shaped records (canvas bbox,
    caption text, model id, prompt, base64 crop) without ever loading the
    real models.
    """
    pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")
    pytest.importorskip("PIL")
    pytest.importorskip("pymupdf")
    import uir_pipeline.caption as caption_mod
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas

    # Build a PDF with one big image (>= MIN_FIGURE_DIM_PX).
    pil = _make_pil_stub(120, 80)
    pil.save(tmp_path / "fig.png")
    pdf = tmp_path / "fig_rich.pdf"
    c = rl_canvas.Canvas(str(pdf), pagesize=letter)
    c.drawString(72, 720, "Figure caption: bar chart example")
    c.drawImage(str(tmp_path / "fig.png"), 72, 600, 120, 80)
    c.showPage()
    c.save()

    # Stub Florence-2 to return a canned caption.
    _install_stub(monkeypatch, canned="four colored bars increasing left to right")
    out = caption_mod.caption_figures_in_pdf(pdf)
    # Pin to exactly 1 figure -- a `>= 1` assertion could regress
    # silently if detect_figure_regions doubles its output (e.g. loops
    # over both pdfplumber.Page.images and rect candidates).
    assert len(out) == 1
    rec = out[0]
    assert rec["page"] == 1
    # (x1n, y1n, x2n, y2n) -- all 0..1000 (utils.bbox_from_pixel clamps).
    b = rec["bbox_canvas"]
    assert all(isinstance(v, int) for v in b)
    assert 0 <= b[0] <= b[2] <= 1000 and 0 <= b[1] <= b[3] <= 1000
    assert rec["caption"] == "four colored bars increasing left to right"
    assert rec["caption_prompt"] == caption_mod.DEFAULT_PROMPT
    assert rec["caption_model"] == caption_mod.MODEL_ID
    assert isinstance(rec["image_b64"], str)


def test_caption_figures_handles_missing_pymupdf(monkeypatch, tmp_path):
    """If PyMuPDF import fails at render time, the function returns an empty list.

    Real on a Dev box without pymupdf (shouldn't happen here, but the
    fail-soft path must hold).
    """
    pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")
    pytest.importorskip("PIL")
    import uir_pipeline.caption as caption_mod
    import builtins as _builtins

    real_import = _builtins.__import__

    def _hide_pymupdf(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("simulated: pymupdf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _hide_pymupdf)
    # Build a valid PDF (so detect_figure_regions succeeds).
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    pdf = tmp_path / "nopymupdf.pdf"
    c = rl_canvas.Canvas(str(pdf), pagesize=letter)
    c.drawString(72, 720, "Hi")
    c.showPage()
    c.save()
    out = caption_mod.caption_figures_in_pdf(pdf)
    assert out == []


# Doc / API surface tests

def test_public_api_exports_present():
    import uir_pipeline.caption as caption_mod
    for name in [
        "caption_images", "detect_figure_regions_from_docling", "render_figure_crop",
        "encode_image_b64", "caption_figures_in_pdf", "is_available",
        "MODEL_ID", "DEFAULT_PROMPT",
    ]:
        assert hasattr(caption_mod, name), f"missing public symbol {name}"


def test_min_dim_filter_measures_on_the_canvas_not_a_rescaled_value():
    """A half-page figure must survive the tiny-bbox filter.

    `min_dim_px` (50) is expressed on the 0-1000 UIR canvas -- about 5% of the
    page. Rescaling the bbox by 50/1000 before comparing, as the code once did,
    turned a 120-unit figure into 6 and rejected everything under 833 units
    wide: every figure but a full-bleed one was silently dropped.
    """
    from types import SimpleNamespace

    import uir_pipeline.caption as caption_mod

    half_page = {"page": 1, "bbox": (100, 100, 600, 500), "kind": "picture"}
    tiny = {"page": 1, "bbox": (10, 10, 30, 30), "kind": "picture"}
    dr = SimpleNamespace(pictures=[half_page, tiny])

    captured: list[dict] = []
    caption_mod_images = caption_mod.caption_images

    def _fake_caption_images(images, **kw):
        return ["a caption"] * len(images)

    caption_mod.caption_images = _fake_caption_images
    try:
        out = caption_mod.caption_figures_in_pdf(docling_result=dr)
    finally:
        caption_mod.caption_images = caption_mod_images

    assert len(out) == 1, "the half-page figure must not be filtered out"
    assert out[0]["bbox_canvas"] == (100, 100, 600, 500)


def test_module_import_does_not_load_florence_weights():
    """Importing the module must NOT trigger Florence-2 downloads.

    We assert by checking ``_MODEL_CACHE`` is empty after a plain import.
    """
    import uir_pipeline.caption as caption_mod
    assert caption_mod._MODEL_CACHE == {}
