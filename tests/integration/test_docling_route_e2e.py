"""End-to-end smoke for the DOCLING route on synthetic Office fixtures.

PLAN \u00a717 \u00a7Multi-format follow-up: the orchestrator's DOCLING route is
plumbed through Stages 2-5 of :func:`uir_pipeline.pipeline.run`, but
until now it's only been unit-tested with a :class:`_FakeConverter` mock
in :file:`tests/test_docling_extract.py`. This integration test:

1. **Generates** a minimal valid ``.docx`` / ``.pptx`` / ``.xlsx`` in
   the test's ``tmp_path`` (no binary blobs committed to the repo).
2. **Runs** :func:`uir_pipeline.pipeline.run` with
   ``fast_path="docling"`` and ``skip_weaviate=True``.
3. **Asserts** the UIR + UMR contract is satisfied for each format:
   a ``.uir.json`` is written, a companion ``.umr.md`` is written,
   and ``PipelineResult.chunk_count > 0``.

**No cache probe.** The user explicitly asked for the e2e to run for real
even when the docling HF model cache is empty -- the first call to
``DocumentConverter()`` will pull the weights (~2 GB) and subsequent
calls are fast. The cost is paid once per cold machine.

**Why ``tests/integration/``?** Per :file:`tests/conftest.py`, tests
under this directory are auto-marked ``slow`` so the marker
``@pytest.mark.slow`` is implicit. Pytest's ``--strict-markers``
configuration in :file:`pytest.ini` won't fail collection even if
the test runs on a non-integration path (we also tag the slow
marker explicitly for safety).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Defensive: bail at import time if any of the office-format libs is
# missing. ``pytest.importorskip`` raises ``Skipped`` immediately so the
# test file's import doesn't drag docx / pptx / openpyxl failures into
# other tests in the run.
docx = pytest.importorskip("docx", reason="python-docx not installed")
pptx = pytest.importorskip("pptx", reason="python-pptx not installed")
openpyxl = pytest.importorskip("openpyxl", reason="openpyxl not installed")
pytest.importorskip("docling", reason="docling not installed")


# Module-level slow marker. ``pytest.mark.slow`` is the existing
# project convention; conftest auto-applies it to ``tests/integration/``
# but we re-apply here in case the file moves. The e2e tests are
# genuinely slow: docling's first call downloads ~2 GB of HF weights
# AND runs the layout model on the synthetic fixture.
pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------------
# Fixture generators (in-memory; no binary blobs)
# ----------------------------------------------------------------------------

def _write_docx(path: Path) -> Path:
    """Write a minimal valid ``.docx`` (heading + 2 paragraphs + 2x2 table).

    The shape is small enough that Docling's PDF-equivalent layout model
    still finds a paragraph / table / heading region -- it doesn't rely
    on size, only on the OOXML structural elements.
    """
    document = docx.Document()
    document.add_heading("Smoke Test Heading", level=1)
    document.add_paragraph(
        "This is a synthetic body paragraph for end-to-end smoke testing. "
        "It contains enough tokens for the BGE embedder to produce a "
        "non-empty vector."
    )
    document.add_paragraph(
        "Second paragraph ensures chunking exercises multi-region paths."
    )
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Header A"
    table.cell(0, 1).text = "Header B"
    table.cell(1, 0).text = "Cell 1"
    table.cell(1, 1).text = "Cell 2"
    document.save(str(path))
    return path


def _write_pptx(path: Path) -> Path:
    """Write a minimal valid ``.pptx`` (one slide, title + body bullets)."""
    prs = pptx.Presentation()
    # ``slide_layouts[1]`` is the "Title and Content" layout bundled
    # with the default pptx template.
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Smoke Test Slide"
    body = slide.placeholders[1]
    body.text = "Bullet one\nBullet two\nBullet three"
    prs.save(str(path))
    return path


def _write_xlsx(path: Path) -> Path:
    """Write a minimal valid ``.xlsx`` (3-row, 3-col sheet with header)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Column A", "Column B", "Column C"])
    ws.append(["Row 1 A", "Row 1 B", "Row 1 C"])
    ws.append(["Row 2 A", "Row 2 B", "Row 2 C"])
    wb.save(str(path))
    return path


# ----------------------------------------------------------------------------
# UIR contract validators
# ----------------------------------------------------------------------------

def _count_chunks(node: dict) -> int:
    """Recursive count of ``type='chunk'`` leaves under a UIR structure node."""
    if node.get("type") == "chunk":
        return 1
    return sum(_count_chunks(child) for child in node.get("children", []))


def _validate_uir_output(
    out_dir: Path,
    expected_format: str,
    expected_route: str = "docling",
) -> None:
    """Assert the orchestrator wrote both ``.uir.json`` and ``.umr.md`` for a fixture.

    Reads the UIR JSON, walks the structure tree, and confirms
    ``chunk_count > 0``. The UMR markdown file's existence is a soft
    contract: the orchestrator can fail-soft on UMR rendering (we've
    observed this in CI before) -- a 0-byte UMR file with a fallback
    placeholder is acceptable here as long as the file is non-empty.

    The UIR ``Source.format`` and ``Source.route`` are checked against
    the per-format expectation so a future refactor that silently routes
    e.g. a ``.docx`` through pdfplumber (which would crash) instead of
    the DOCLING branch can be caught here.
    """
    uir_files = sorted(out_dir.glob("*.uir.json"))
    umr_files = sorted(out_dir.glob("*.umr.md"))
    assert uir_files, f"no .uir.json in {out_dir}"
    assert umr_files, f"no .umr.md in {out_dir}"
    for uir_path in uir_files:
        uir = json.loads(uir_path.read_text())
        # Source.format + route should reflect the input format.
        assert uir["source"]["format"] == expected_format, (
            f"source.format={uir['source']['format']!r} "
            f"(expected {expected_format!r}) in {uir_path.name}"
        )
        assert uir["source"]["route"] == expected_route, (
            f"source.route={uir['source']['route']!r} "
            f"(expected {expected_route!r}) in {uir_path.name}"
        )
        chunk_count = _count_chunks(uir["structure"]["root"])
        assert chunk_count > 0, f"zero chunks in {uir_path.name}; UIR id={uir.get('id')!r}"
    for umr_path in umr_files:
        # UMR may be a fail-soft placeholder; require non-empty.
        assert umr_path.stat().st_size > 0, f"empty UMR file: {umr_path}"


# ----------------------------------------------------------------------------
# E2E tests
# ----------------------------------------------------------------------------

def test_docling_route_docx_e2e(tmp_path: Path):
    """End-to-end: synthetic ``.docx`` -> ``run(fast_path='docling')`` -> ``.uir.json`` + ``.umr.md`` with chunks."""
    src = _write_docx(tmp_path / "smoke.docx")
    out = tmp_path / "out"
    out.mkdir()
    from uir_pipeline.pipeline import run
    result = run(
        src,
        output_dir=out,
        skip_weaviate=True,
        with_embeddings=False,
        fast_path="docling",
    )
    assert result.chunk_count > 0, f"docx: chunk_count={result.chunk_count}"
    _validate_uir_output(out, expected_format="DOCX")


def test_docling_route_pptx_e2e(tmp_path: Path):
    """End-to-end: synthetic ``.pptx`` -> ``run(fast_path='docling')`` -> ``.uir.json`` + ``.umr.md`` with chunks."""
    src = _write_pptx(tmp_path / "smoke.pptx")
    out = tmp_path / "out"
    out.mkdir()
    from uir_pipeline.pipeline import run
    result = run(
        src,
        output_dir=out,
        skip_weaviate=True,
        with_embeddings=False,
        fast_path="docling",
    )
    assert result.chunk_count > 0, f"pptx: chunk_count={result.chunk_count}"
    _validate_uir_output(out, expected_format="PPTX", expected_route="pptx")


def test_docling_route_xlsx_e2e(tmp_path: Path):
    """End-to-end: synthetic ``.xlsx`` -> ``run(fast_path='docling')`` -> ``.uir.json`` + ``.umr.md`` with chunks."""
    src = _write_xlsx(tmp_path / "smoke.xlsx")
    out = tmp_path / "out"
    out.mkdir()
    from uir_pipeline.pipeline import run
    result = run(
        src,
        output_dir=out,
        skip_weaviate=True,
        with_embeddings=False,
        fast_path="docling",
    )
    assert result.chunk_count > 0, f"xlsx: chunk_count={result.chunk_count}"
    _validate_uir_output(out, expected_format="XLSX")
