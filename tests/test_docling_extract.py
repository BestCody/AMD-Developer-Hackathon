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


