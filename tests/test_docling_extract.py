"""tests/test_docling_extract.py -- Docling fast-path wrapper regression tests.

PLAN §17 §OCR follow-up: tests pin the fail-soft + shape contract of
``src/uir_pipeline/docling_extract.extract_with_docling`` + the
``_docling_to_table_draft`` / ``_resolve_fast_path`` helpers in
``src/uir_pipeline/pipeline.py``. All tests use mock DocumentConverter
instances and ``monkeypatch``-rolled env vars so the 2 GB HuggingFace
weight download is never triggered -- these are pure-shape regression
tests, not integration tests.
"""
from __future__ import annotations

import pytest

from uir_pipeline.docling_extract import (
    DoclingResult,
    DoclingUnavailable,
    docling_environment_enabled,
    extract_with_docling,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _FakeProv:
    """Minimal stand-in for ``docling_core.types.doc.provenance.Provenance``."""

    def __init__(self, page: int = 1, bbox=(0, 0, 100, 100)):
        self.page = page
        self.bbox = bbox


class _FakeItem:
    """Minimal stand-in for a ``DoclingTextItem`` (section_header / paragraph
    / list_item shapes). The wrapper reads ``label_name``, ``label``,
    ``text``, and ``prov`` attrs -- this fake supplies exactly those."""

    def __init__(
        self,
        label: str,
        text: str,
        bbox: tuple[int, int, int, int] = (0, 0, 100, 100),
        page: int = 1,
    ):
        self.label = label
        self.label_name = label
        self.text = text
        self.prov = _FakeProv(page=page, bbox=bbox)


class _FakeTableItem:
    """Minimal stand-in for a ``DoclingTableItem``. Exposes
    ``export_to_markdown()`` because that's the wrapper's text source."""

    def __init__(
        self,
        markdown: str,
        bbox: tuple[int, int, int, int] = (0, 0, 200, 200),
        page: int = 1,
    ):
        self.label = "table"
        self.label_name = "table"
        self.prov = _FakeProv(page=page, bbox=bbox)
        self._md = markdown

    def export_to_markdown(self) -> str:
        return self._md


class _FakePage:
    """Minimal stand-in for ``docling_core.types.doc.pages.DoclingPage``."""

    def __init__(self, page_no: int, items: list):
        self.page_no = page_no
        self.items = items


class _FakeDocument:
    """Minimal stand-in for a ``DoclingDocument``."""

    def __init__(self, tables=None, pages=None):
        self.tables = tables or []
        self.pages = pages or []


class _FakeConverter:
    """Minimal stand-in for ``docling.document_converter.DocumentConverter``.

    The wrapper's ``extract_with_docling`` accepts an injected converter
    so tests never trigger the real (heavyweight) ``DocumentConverter``.
    """

    def __init__(self, document=None, raises: Exception | None = None):
        self._doc = document
        self._raises = raises

    def convert(self, path: str):  # noqa: ARG002 -- signature mirrors upstream
        if self._raises is not None:
            raise self._raises
        # Wrapper reads ``result.document`` OR ``result.output``. We return
        # a simple object that exposes both attributes so the wrapper's
        # duck-typed attribute access resolves cleanly.
        out = type("ConversionResult", (), {})()
        out.document = self._doc
        out.output = self._doc
        return out


# ---------------------------------------------------------------------------
# Pure-shape tests for extract_with_docling
# ---------------------------------------------------------------------------


def test_extract_with_docling_unavailable_when_import_fails(monkeypatch):
    """When docling is not importable, extract_with_docling raises DoclingUnavailable."""
    from uir_pipeline import docling_extract

    def _raise():
        raise DoclingUnavailable("simulated missing-docling import")

    monkeypatch.setattr(docling_extract, "_import_docling_or_raise", _raise)

    with pytest.raises(DoclingUnavailable):
        extract_with_docling("/path/to/any.pdf")


def test_extract_with_docling_converter_failure_raises_unavailable(tmp_path):
    """Converter that throws surfaces as DoclingUnavailable."""
    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(DoclingUnavailable):
        extract_with_docling(
            fake_pdf, converter=_FakeConverter(raises=RuntimeError("boom")),
        )


def test_extract_with_docling_shape_contract(tmp_path):
    """Mocked Docling converter returns a populated DoclingResult with
    the expected region labels + per-page ordering + native markdown for tables.
    """
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    fake_table = _FakeTableItem(
        markdown="| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n",
        bbox=(10, 20, 990, 800),
        page=1,
    )
    fake_page1 = _FakePage(
        page_no=1,
        items=[
            _FakeItem("section_header", "Introduction", page=1),
            _FakeItem("paragraph", "This is the intro paragraph.", page=1),
            _FakeItem("list_item", "First bullet", page=1),
        ],
    )
    fake_page2 = _FakePage(
        page_no=2,
        items=[_FakeItem("text", "Page two prose.", page=2)],
    )
    doc = _FakeDocument(tables=[fake_table], pages=[fake_page1, fake_page2])
    converter = _FakeConverter(document=doc)

    result = extract_with_docling(fake_pdf, converter=converter)
    assert isinstance(result, DoclingResult)

    # Three typed items + one paragraph on page 2 == 4 regions, no duplication
    # of the table (it's emitted only via ``tables``).
    assert len(result.regions) == 4
    labels = [r["label"] for r in result.regions]
    assert labels == ["heading", "paragraph", "list", "paragraph"]
    pages = [r["page"] for r in result.regions]
    assert pages == [1, 1, 1, 2]

    # Tables: 1 markdown table; the GFM shape is preserved verbatim.
    assert len(result.tables) == 1
    assert "a | b" in result.tables[0]["markdown"]
    assert result.tables[0]["page"] == 1

    # page_texts: 2 entries (one per page), joined with double-newlines.
    assert len(result.page_texts) == 2
    assert any("Introduction" in txt for _, txt in result.page_texts)


def test_extract_with_docling_handles_missing_output_gracefully(tmp_path):
    """Converter that returns a result without ``document``/``output``
    surfaces as DoclingUnavailable (the wrapper's defensive guard)."""
    fake_pdf = tmp_path / "weird.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")

    class _OddResult:
        # Deliberately omit ``document`` AND ``output``.
        pass

    class _OddConverter:
        def convert(self, path: str):  # noqa: ARG002
            return _OddResult()

    with pytest.raises(DoclingUnavailable):
        extract_with_docling(fake_pdf, converter=_OddConverter())


def test_extract_with_docling_dedupes_tables(tmp_path):
    """Tables emitted both in ``doc.tables`` AND in a page's items list
    are counted only once."""
    fake_pdf = tmp_path / "dup.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    md = "| col1 | col2 |\n|---|---|\n| a | b |\n"
    fake_table_top = _FakeTableItem(markdown=md, page=1)
    fake_table_inline = _FakeTableItem(markdown=md, page=1)
    fake_page1 = _FakePage(
        page_no=1,
        items=[
            _FakeItem("section_header", "Header", page=1),
            _FakeItem("table", "label-only", page=1),
        ],
    )
    doc = _FakeDocument(
        tables=[fake_table_top, fake_table_inline],
        pages=[fake_page1],
    )
    converter = _FakeConverter(document=doc)

    result = extract_with_docling(fake_pdf, converter=converter)
    # Same markdown + same page -> dedup -> 1 unique table.
    assert len(result.tables) == 1
    # The inline label-only table is filtered from regions (no markdown text).
    assert len(result.regions) == 1
    assert result.regions[0]["label"] == "heading"


# ---------------------------------------------------------------------------
# pipeline.py helpers
# ---------------------------------------------------------------------------


def test_docling_to_table_draft_synthesizes_counts():
    """_docling_to_table_draft derives row + col counts from markdown."""
    from uir_pipeline.pipeline import _docling_to_table_draft

    t = {
        "markdown": (
            "| col1 | col2 | col3 |\n"
            "|---|---|---|\n"
            "| a | b | c |\n"
            "| d | e | f |\n"
        ),
        "page": 7,
        "bbox": (10, 20, 990, 800),
    }
    draft = _docling_to_table_draft(t)
    assert draft.page_number == 7
    assert draft.bbox == (10, 20, 990, 800)
    # 3 pipes - 1 = 3 columns on the header row
    assert draft.col_count == 3
    # Header + 2 data rows = 3 total (separator row excluded by ``---`` filter)
    assert draft.row_count == 3
    assert draft.confidence == 0.9
    assert isinstance(draft.markdown, str)


def test_docling_to_table_draft_handles_empty_markdown():
    """Empty markdown yields zero-count TableDraft without raising."""
    from uir_pipeline.pipeline import _docling_to_table_draft

    draft = _docling_to_table_draft(
        {"markdown": "", "page": 1, "bbox": (0, 0, 0, 0)},
    )
    assert draft.row_count == 0
    assert draft.col_count == 0


def test_resolve_fast_path_explicit_arg_wins_over_env(monkeypatch, caplog):
    """Explicit fast_path arg wins over UIR_FAST_PATH env var.

    ``"pdfplumber"`` is now a deprecated alias for ``"docling"`` --
    the resolver logs a one-shot warning and returns ``"docling"``.
    The legal-value half (``"docling"`` arg + ``"pdfplumber"`` env)
    keeps the strict "explicit arg wins over env" semantics.
    """
    from uir_pipeline.pipeline import _resolve_fast_path
    import logging as _logging

    # Alias case: explicit ``pdfplumber`` arg resolves to ``docling`` with
    # a deprecation warning regardless of the env var.
    with caplog.at_level(_logging.WARNING):
        monkeypatch.setenv("UIR_FAST_PATH", "docling")
        assert _resolve_fast_path("pdfplumber") == "docling"
    assert any(
        "deprecated" in r.message and "pdfplumber" in r.message
        for r in caplog.records
    )

    # Legal-value case: explicit ``docling`` arg still wins, even when
    # the env var says ``pdfplumber``.
    caplog.clear()
    monkeypatch.setenv("UIR_FAST_PATH", "pdfplumber")
    assert _resolve_fast_path("docling") == "docling"


def test_resolve_fast_path_env_var_used_when_arg_none(monkeypatch, caplog):
    """When arg is None, UIR_FAST_PATH env var is honored (with alias handling).

    ``UIR_FAST_PATH=pdfplumber`` is now a deprecated alias -- the resolver
    logs a one-shot warning and returns ``"docling"``. The legal-value case
    (``UIR_FAST_PATH=docling``) returns ``"docling"`` cleanly.
    """
    from uir_pipeline.pipeline import _resolve_fast_path
    import logging as _logging

    # Alias case: env=pdfplumber resolves to docling with a deprecation warning.
    with caplog.at_level(_logging.WARNING):
        monkeypatch.setenv("UIR_FAST_PATH", "pdfplumber")
        assert _resolve_fast_path(None) == "docling"
    assert any(
        "deprecated" in r.message and "pdfplumber" in r.message
        for r in caplog.records
    )

    # Legal-value case: env=docling returns docling cleanly.
    caplog.clear()
    monkeypatch.setenv("UIR_FAST_PATH", "docling")
    assert _resolve_fast_path(None) == "docling"


def test_resolve_fast_path_defaults_to_docling(monkeypatch):
    """Absent arg + absent env var -> 'docling' (production default)."""
    from uir_pipeline.pipeline import _resolve_fast_path

    monkeypatch.delenv("UIR_FAST_PATH", raising=False)
    assert _resolve_fast_path(None) == "docling"


def test_resolve_fast_path_unknown_env_falls_back_to_docling(monkeypatch, caplog):
    """Unknown UIR_FAST_PATH value logs a warning and falls back to docling.

    Defensive: a typo in the env var shouldn't silently route to a
    non-existent backend.
    """
    from uir_pipeline.pipeline import _resolve_fast_path

    monkeypatch.setenv("UIR_FAST_PATH", "totally-not-a-real-backend")
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        assert _resolve_fast_path(None) == "docling"
    assert any("unknown UIR_FAST_PATH" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Orchestrator cascade (smoke-level)
# ---------------------------------------------------------------------------


def test_resolve_fast_path_pdfplumber_alias_logs_deprecation(monkeypatch, caplog):
    """``fast_path="pdfplumber"`` is now a deprecated alias for "docling".
    The resolver emits a one-shot warning and returns "docling".
    """
    from uir_pipeline.pipeline import _resolve_fast_path
    import logging as _logging
    monkeypatch.delenv("UIR_FAST_PATH", raising=False)
    with caplog.at_level(_logging.WARNING):
        assert _resolve_fast_path("pdfplumber") == "docling"
    assert any(
        "deprecated" in r.message and "pdfplumber" in r.message
        for r in caplog.records
    )


@pytest.mark.skipif(
    not docling_environment_enabled(),
    reason="alias fast_path='pdfplumber' now routes through docling",
)
def test_orchestrator_routes_via_fast_path_arg(tmp_data_dir):
    """Smoke: ``fast_path='pdfplumber'`` is now a deprecated alias for the
    docling backend -- the orchestrator emits a valid UIR with chunks via
    docling, with a one-shot deprecation warning logged. Skipped when
    docling isn't importable OR its model weights are unreachable.
    """
    from pathlib import Path
    import shutil

    src_pdf = Path("tests/fixtures/sample_pdfs/flat_text.pdf")
    if not src_pdf.is_file():
        pytest.skip(f"fixture missing: {src_pdf}")

    pdf = tmp_data_dir / "input" / src_pdf.name
    shutil.copy2(src_pdf, pdf)

    from uir_pipeline.pipeline import run

    try:
        result = run(
            pdf,
            output_dir=tmp_data_dir / "output",
            skip_weaviate=True,
            with_embeddings=False,
            page_numbers=[1],
            fast_path="pdfplumber",  # deprecated alias -- routed to docling with warning
            include_semantics=False,
        )
    except DoclingUnavailable as exc:
        # Routed through docling but the backend couldn't run (e.g. HF
        # model download blocked). Skip rather than FAIL for an env issue.
        pytest.skip(f"alias routes through docling ({exc}); skip when backend unreachable")
    assert result.out_path.is_file()
    # UMR companion always emitted (Phase 17 §UMR).
    assert getattr(result, "umr_path", None) is not None
    assert Path(result.umr_path).is_file()
    # Chunks emitted via the docling branch (the alias routes through it).
    assert result.chunk_count > 0


def test_orchestrator_propagates_when_docling_unavailable(tmp_data_dir, monkeypatch):
    """Smoke: when the docling import is monkeypatched to raise
    :class:`DoclingUnavailable`, the orchestrator RE-RAISES (no silent
    cascade to a legacy backend). Pre-refactor this cascaded to the
    pdfplumber path; that cascade was removed because the pdfplumber output
    was column-interleaved and broke double-column reading order.
    """
    from pathlib import Path
    import shutil

    src_pdf = Path("tests/fixtures/sample_pdfs/flat_text.pdf")
    if not src_pdf.is_file():
        pytest.skip(f"fixture missing: {src_pdf}")

    pdf = tmp_data_dir / "input" / src_pdf.name
    shutil.copy2(src_pdf, pdf)

    from uir_pipeline import docling_extract

    def _raise_during_import():
        raise DoclingUnavailable("simulated-propagate-test")

    monkeypatch.setattr(
        docling_extract, "_import_docling_or_raise", _raise_during_import,
    )

    from uir_pipeline.pipeline import run

    with pytest.raises(DoclingUnavailable):
        run(
            pdf,
            output_dir=tmp_data_dir / "output",
            skip_weaviate=True,
            with_embeddings=False,
            page_numbers=[1],
            fast_path="docling",
            include_semantics=False,
        )


# ---------------------------------------------------------------------------
# Issue fix: table confidence (OCR + structural sanity) -- Bug 1
# ---------------------------------------------------------------------------

def test_ocr_table_gets_lowered_confidence():
    """A table emitted from an OCR-converted (scanned) page inherits a
    lowered confidence (0.6) and an ``ocr`` flag, regardless of whether
    its markdown is structurally clean."""
    fake_table = _FakeTableItem(
        markdown="| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n",
        bbox=(10, 20, 990, 800), page=1,
    )
    doc = _FakeDocument(tables=[fake_table], pages=[])
    from uir_pipeline.docling_extract import _walk_doc
    result = _walk_doc(doc, ocr_applied=True)
    assert len(result.tables) == 1
    assert result.tables[0]["confidence"] == 0.6
    assert result.tables[0]["ocr"] is True


def test_borndigital_table_keeps_full_confidence():
    """A clean born-digital table (no OCR) keeps confidence 1.0."""
    fake_table = _FakeTableItem(
        markdown="| a | b |\n|---|---|\n| 1 | 2 |\n",
        bbox=(10, 20, 990, 800), page=1,
    )
    doc = _FakeDocument(tables=[fake_table], pages=[])
    from uir_pipeline.docling_extract import _walk_doc
    result = _walk_doc(doc)
    assert result.tables[0]["confidence"] == 1.0
    assert result.tables[0]["ocr"] is False


def test_table_with_numeric_column_rowshift_fails_sanity(caplog):
    """Reproduces the 02_scan_simulated_invoice.pdf symptom: UNIT/EXT are
    shifted so the currency column ends up mostly empty. The structural
    sanity check flags the table (confidence -> 0.4) and logs why."""
    import logging as _logging
    from uir_pipeline.docling_extract import _walk_doc
    md = (
        "| SKU | Description | QTY | UNIT $ | EXT $ |\n"
        "|---|---|---|---|---|\n"
        "| A1 | Widget | 2 | 10.00 | |\n"
        "| A2 | Gadget | 3 | 15.00 | |\n"
    )
    fake_table = _FakeTableItem(markdown=md, page=1)
    doc = _FakeDocument(tables=[fake_table], pages=[])
    with caplog.at_level(_logging.WARNING):
        result = _walk_doc(doc)
    assert result.tables[0]["confidence"] == 0.4
    assert any(
        "structural sanity" in r.message and "EXT" in r.message
        for r in caplog.records
    )


def test_table_with_ragged_columns_fails_sanity():
    """A data row with a different column count than the header is caught
    by the sanity check and flagged (confidence -> 0.4)."""
    from uir_pipeline.docling_extract import _walk_doc
    md = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| 1 | 2 | 3 | 4 |\n"
    )
    fake_table = _FakeTableItem(markdown=md, page=1)
    doc = _FakeDocument(tables=[fake_table], pages=[])
    result = _walk_doc(doc)
    assert result.tables[0]["confidence"] == 0.4


def test_docling_to_table_draft_reads_confidence():
    """_docling_to_table_draft threads the per-table confidence (set in
    docling_extract) into TableDraft; missing key falls back to 0.9."""
    from uir_pipeline.pipeline import _docling_to_table_draft
    low = _docling_to_table_draft({
        "markdown": "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "page": 3, "bbox": (1, 2, 3, 4), "confidence": 0.6,
    })
    assert low.confidence == 0.6
    legacy = _docling_to_table_draft({
        "markdown": "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "page": 3, "bbox": (1, 2, 3, 4),
    })
    assert legacy.confidence == 0.9


# ---------------------------------------------------------------------------
# Issue fix: decorative / watermark text (raw-label discard + bbox overlap) -- Bug 2
# ---------------------------------------------------------------------------

def test_decorative_raw_label_is_dropped():
    """A region whose raw Docling label is a furniture/decorative name is
    skipped entirely -- it must not reach the "paragraph" fallback."""
    from uir_pipeline.docling_extract import _walk_doc, _DECORATIVE_LABELS
    # Use a label that is NOT in _LABEL_MAP but IS in _DECORATIVE_LABELS.
    stamp_label = sorted(_DECORATIVE_LABELS)[0]
    page = _FakePage(page_no=1, items=[
        _FakeItem("paragraph", "Real body text on the page.", page=1,
                  bbox=(50, 50, 300, 200)),
        _FakeItem(stamp_label, "DRAFT - DO NOT DISTRIBUTE", page=1,
                  bbox=(50, 50, 300, 200)),
    ])
    doc = _FakeDocument(tables=[], pages=[page])
    result = _walk_doc(doc)
    texts = [r["text"] for r in result.regions]
    assert "Real body text on the page." in texts
    assert "DRAFT - DO NOT DISTRIBUTE" not in texts


def test_empty_list_region_is_dropped():
    """Regression: _LABEL_MAP normalizes both "list_item" and "list" to the
    canonical "list", so the empty-text noise filter must check "list" (not
    the raw "list_item") or empty list regions leak into the stream."""
    from uir_pipeline.docling_extract import _walk_doc
    page = _FakePage(page_no=1, items=[
        _FakeItem("list_item", "", page=1, bbox=(50, 50, 300, 200)),
        _FakeItem("list_item", "A real bullet", page=1, bbox=(50, 260, 300, 360)),
    ])
    doc = _FakeDocument(tables=[], pages=[page])
    result = _walk_doc(doc)
    assert len(result.regions) == 1
    assert result.regions[0]["text"] == "A real bullet"


def test_overlap_stamp_dropped_body_kept():
    """Label-independent backstop: a large rotated "DRAFT" stamp that
    Docling mislabeled as "paragraph" (so it escapes the label discard)
    overlaps several distinct body paragraphs; it is dropped while the
    body paragraphs survive."""
    from uir_pipeline.docling_extract import _walk_doc
    stamp = _FakeItem("paragraph", "DRAFT - DO NOT DISTRIBUTE", page=1,
                      bbox=(40, 40, 460, 460))
    para1 = _FakeItem("paragraph", "Body paragraph one of section three.",
                       page=1, bbox=(100, 100, 300, 200))
    para2 = _FakeItem("paragraph", "Body paragraph two of section three.",
                       page=1, bbox=(100, 260, 300, 360))
    page = _FakePage(page_no=1, items=[stamp, para1, para2])
    doc = _FakeDocument(tables=[], pages=[page])
    result = _walk_doc(doc)
    texts = [r["text"] for r in result.regions]
    assert "DRAFT - DO NOT DISTRIBUTE" not in texts
    assert "Body paragraph one of section three." in texts
    assert "Body paragraph two of section three." in texts


def test_shared_bbox_cluster_not_nuked():
    """Regression guard: several regions that merely share one bounding box
    (e.g. a heading and its body text Docling bounds identically) must NOT
    be deleted by the overlap filter -- only a region overlapping several
    DISTINCT geometries is a decorative overlay."""
    from uir_pipeline.docling_extract import _walk_doc
    items = [
        _FakeItem("section_header", "Section title", page=1,
                   bbox=(0, 0, 100, 100)),
        _FakeItem("paragraph", "Paragraph bound to the same box", page=1,
                   bbox=(0, 0, 100, 100)),
        _FakeItem("list_item", "A bullet in the same box", page=1,
                   bbox=(0, 0, 100, 100)),
    ]
    page = _FakePage(page_no=1, items=items)
    doc = _FakeDocument(tables=[], pages=[page])
    result = _walk_doc(doc)
    # All three survive (distinct-bbox count is 1, below the drop threshold).
    assert len(result.regions) == 3


def test_chunk_text_honors_injected_confidence():
    """chunk_text passes an injected confidence through to ChunkDraft so a
    low-confidence (OCR / sanity-failed) table carries its flag downstream."""
    from uir_pipeline.chunk import chunk_text
    drafts = chunk_text(
        "| a | b |\n|---|---|\n| 1 | 2 |\n", page=1,
        bbox=(0, 0, 100, 100), region_kind="table", confidence=0.6,
    )
    assert len(drafts) >= 1
    assert all(d.confidence == 0.6 for d in drafts)


def test_strip_watermark_text_removes_phrase_only():
    """_strip_watermark_text removes a fused stamp phrase but leaves the
    surrounding body text (the case Docling produces on a rotated stamp)."""
    from uir_pipeline.docling_extract import _strip_watermark_text
    fused = (
        "The quick brown fox jumps over the lazy dog while the system "
        "ingests documents\r\nDRAFT - DO NOT DISTRIBUTE"
    )
    out = _strip_watermark_text(fused)
    assert "DRAFT" not in out.upper()
    assert "DISTRIBUTE" not in out.upper()
    assert "quick brown fox" in out


def test_strip_watermark_text_idempotent_on_clean():
    """Stamp-free body text is returned unchanged (no false positives on
    ordinary prose mentioning 'a draft proposal')."""
    from uir_pipeline.docling_extract import _strip_watermark_text
    clean = "We prepared a draft proposal for the confidential review."
    assert _strip_watermark_text(clean) == clean


@pytest.mark.skipif(
    not docling_environment_enabled(),
    reason="docling not importable in this environment",
)
def test_fixture_watermark_absent_from_regions():
    """Integration: re-run 01_messy_multicolumn_report.pdf end-to-end; the
    DRAFT stamp must NOT appear in any emitted region."""
    from pathlib import Path

    from uir_pipeline.docling_extract import (
        DoclingUnavailable,
        extract_with_docling,
    )
    pdf = Path("tests/fixtures/sample_pdfs/01_messy_multicolumn_report.pdf")
    if not pdf.is_file():
        pytest.skip(f"fixture missing: {pdf}")
    try:
        r = extract_with_docling(pdf)
    except DoclingUnavailable as exc:
        pytest.skip(f"docling backend unavailable ({exc}); skip")
    joined = " ".join(x["text"] for x in r.regions).upper()
    assert "DRAFT" not in joined
    assert "DO NOT DISTRIBUTE" not in joined


@pytest.mark.skipif(
    not docling_environment_enabled(),
    reason="docling not importable in this environment",
)
def test_fixture_scan_invoice_table_flagged_low_confidence():
    """Integration: re-run 02_scan_simulated_invoice.pdf (image-only, so it
    routes through OCR); its table must carry confidence < 1.0 and ocr=True."""
    from pathlib import Path

    from uir_pipeline.docling_extract import (
        DoclingUnavailable,
        extract_with_docling,
    )
    pdf = Path("tests/fixtures/sample_pdfs/02_scan_simulated_invoice.pdf")
    if not pdf.is_file():
        pytest.skip(f"fixture missing: {pdf}")
    try:
        r = extract_with_docling(pdf)
    except DoclingUnavailable as exc:
        pytest.skip(f"docling/OCR backend unavailable ({exc}); skip")
    assert len(r.tables) >= 1, "scan fixture yielded no table to flag"
    assert all(t["confidence"] < 1.0 for t in r.tables)
    assert any(t.get("ocr") for t in r.tables)


