"""Regression tests for Docling 2.x object shapes in ``docling_extract``.

Three bugs shipped together and made every PDF convert to a *successful*
job with zero chunks:

1. ``doc.pages`` is a ``dict[int, PageItem]``. Iterating it yields ints, so
   ``getattr(page, "items")`` returned None for every page and the flat
   ``doc.texts`` fallback was unreachable -> 0 regions.
2. ``item.prov`` is a *list* of provenance records. Reading ``.page_no`` /
   ``.bbox`` off the list silently yielded None -> every region on page 1
   with a zero bbox.
3. Docling's ``BoundingBox`` uses a BOTTOMLEFT origin, so ``t > b``. Mapping
   l/t/r/b straight onto x1/y1/x2/y2 gives ``y1 > y2``, which UIR's
   ChunkNode validator rejects. This one was masked by bug 2 -- once the
   bbox stopped being (0,0,0,0), every conversion started failing.

The existing ``test_docling_extract.py`` coverage skips without a
``tests/fixtures/sample_pdfs/flat_text.pdf`` fixture (gitignored), which is
how these reached main. These tests use fakes: no Docling, no fixture.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from uir_pipeline.docling_extract import (
    DoclingPartialConversion,
    DoclingUnavailable,
    _bbox_xyxy,
    _first_prov,
    _page_number,
    _walk_doc,
    extract_with_docling,
)


class _BBox:
    """Mimics docling's BoundingBox with a BOTTOMLEFT origin (t > b)."""

    def __init__(self, l, t, r, b):  # noqa: E741
        self.l, self.t, self.r, self.b = l, t, r, b


def _text_item(text: str, page: int, bbox: _BBox):
    """A docling TextItem: label str, .text, and prov as a LIST."""
    return SimpleNamespace(
        label="text",
        text=text,
        prov=[SimpleNamespace(page_no=page, bbox=bbox)],
    )


# ---------------------------------------------------------------------------
# bug 2 -- prov is a list
# ---------------------------------------------------------------------------

def test_first_prov_unwraps_a_list():
    p = SimpleNamespace(page_no=4)
    assert _first_prov(SimpleNamespace(prov=[p])) is p


def test_first_prov_passes_through_a_bare_object():
    p = SimpleNamespace(page_no=4)
    assert _first_prov(SimpleNamespace(prov=p)) is p


def test_first_prov_handles_missing_and_empty():
    assert _first_prov(SimpleNamespace()) is None
    assert _first_prov(SimpleNamespace(prov=[])) is None


def test_page_number_reads_through_the_prov_list():
    it = _text_item("x", page=7, bbox=_BBox(0, 10, 5, 0))
    assert _page_number(it, fallback=1) == 7, "prov-as-list must not fall back to page 1"


# ---------------------------------------------------------------------------
# bug 3 -- bottom-left origin means t > b
# ---------------------------------------------------------------------------

def test_bbox_orders_axes_under_bottomleft_origin():
    # Real value observed from docling: l=56 t=782 r=447 b=691
    assert _bbox_xyxy(_BBox(56, 782, 447, 691)) == (56, 691, 447, 782)


def test_bbox_is_always_validator_safe():
    """UIR's ChunkNode requires x1 <= x2 and y1 <= y2."""
    for box in (_BBox(56, 782, 447, 691), _BBox(447, 691, 56, 782), _BBox(0, 0, 0, 0)):
        x1, y1, x2, y2 = _bbox_xyxy(box)
        assert x1 <= x2 and y1 <= y2


def test_bbox_clamps_to_the_uir_canvas():
    assert _bbox_xyxy(_BBox(-40, 5000, 20, -3)) == (0, 0, 20, 1000)


# ---------------------------------------------------------------------------
# bug 1 -- doc.pages is a dict; content lives in the flat doc.texts
# ---------------------------------------------------------------------------

def test_walk_doc_reads_flat_texts_when_pages_is_a_dict():
    """The Docling 2.x shape: dict pages, no per-page item stream."""
    doc = SimpleNamespace(
        pages={1: SimpleNamespace(page_no=1)},  # PageItem has no `.items`
        texts=[
            _text_item("First block of prose.", 1, _BBox(56, 782, 447, 691)),
            _text_item("Second block of prose.", 1, _BBox(56, 717, 462, 463)),
        ],
        tables=[],
        pictures=[],
    )
    out = _walk_doc(doc)
    assert len(out.regions) == 2, "dict-shaped doc.pages must not yield zero regions"
    assert [r["text"] for r in out.regions] == [
        "First block of prose.",
        "Second block of prose.",
    ]


def test_walk_doc_assigns_the_provenance_page_not_a_fallback():
    doc = SimpleNamespace(
        pages={1: SimpleNamespace(page_no=1), 2: SimpleNamespace(page_no=2)},
        texts=[
            _text_item("on page one", 1, _BBox(0, 10, 5, 0)),
            _text_item("on page two", 2, _BBox(0, 10, 5, 0)),
            _text_item("also page two", 2, _BBox(0, 20, 5, 10)),
        ],
        tables=[],
        pictures=[],
    )
    out = _walk_doc(doc)
    assert {r["page"] for r in out.regions} == {1, 2}
    assert sum(r["page"] == 2 for r in out.regions) == 2


def test_walk_doc_emits_validator_safe_bboxes():
    doc = SimpleNamespace(
        pages={1: SimpleNamespace(page_no=1)},
        texts=[_text_item("prose", 1, _BBox(56, 782, 447, 691))],
        tables=[],
        pictures=[],
    )
    (region,) = _walk_doc(doc).regions
    x1, y1, x2, y2 = region["bbox"]
    assert (x1, y1, x2, y2) == (56, 691, 447, 782)
    assert x1 <= x2 and y1 <= y2


def test_walk_doc_still_prefers_a_page_item_stream_when_offered():
    """Older/other builds that attach `.items` to pages must keep working."""
    page = SimpleNamespace(page_no=3, items=[_text_item("from page stream", 3, _BBox(0, 10, 5, 0))])
    doc = SimpleNamespace(pages=[page], texts=[], tables=[], pictures=[])
    out = _walk_doc(doc)
    assert len(out.regions) == 1
    assert out.regions[0]["text"] == "from page stream"
    assert out.regions[0]["page"] == 3


def test_walk_doc_survives_an_empty_document():
    doc = SimpleNamespace(pages={}, texts=[], tables=[], pictures=[])
    assert _walk_doc(doc).regions == []


# ---------------------------------------------------------------------------
# bug 4 -- convert() returns normally on a PARTIAL_SUCCESS
# ---------------------------------------------------------------------------
# Docling does not raise when some pages fail (std::bad_alloc under memory
# pressure is the common cause). It hands back a DoclingDocument holding only
# the pages that survived. Accepting that produced a `done` job whose UIR was
# missing most of the document -- silent, unbounded data loss.

def _doc_with_one_page():
    return SimpleNamespace(
        pages={1: SimpleNamespace(page_no=1)},
        texts=[_text_item("only the first page survived", 1, _BBox(0, 10, 5, 0))],
        tables=[], pictures=[],
    )


class _FakeConverter:
    def __init__(self, status, errors=()):
        self._status, self._errors = status, list(errors)

    def convert(self, _path):
        return SimpleNamespace(
            document=_doc_with_one_page(),
            status=SimpleNamespace(name=self._status),
            errors=self._errors,
        )


def test_partial_success_raises_instead_of_returning_a_truncated_document(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    conv = _FakeConverter("PARTIAL_SUCCESS", ["page 4: std::bad_alloc"])
    with pytest.raises(DoclingPartialConversion, match="only partially"):
        extract_with_docling(pdf, converter=conv)


def test_partial_success_error_names_the_page_failure(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    conv = _FakeConverter("PARTIAL_SUCCESS", ["page 4: std::bad_alloc"])
    with pytest.raises(DoclingPartialConversion) as ei:
        extract_with_docling(pdf, converter=conv)
    assert "std::bad_alloc" in str(ei.value)
    assert "1 page error" in str(ei.value)


def test_partial_success_can_be_opted_into(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCLING_ALLOW_PARTIAL", "1")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    conv = _FakeConverter("PARTIAL_SUCCESS", ["page 4: std::bad_alloc"])
    out = extract_with_docling(pdf, converter=conv)
    assert len(out.regions) == 1  # the caller explicitly accepted the truncation


def test_partial_conversion_is_catchable_as_docling_unavailable():
    """The orchestrator's existing `except DoclingUnavailable` must still fire."""
    assert issubclass(DoclingPartialConversion, DoclingUnavailable)


def test_failure_status_raises(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(DoclingUnavailable, match="status=FAILURE"):
        extract_with_docling(pdf, converter=_FakeConverter("FAILURE"))


def test_success_status_passes_through(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    out = extract_with_docling(pdf, converter=_FakeConverter("SUCCESS"))
    assert len(out.regions) == 1


def test_converter_without_a_status_attribute_is_accepted(tmp_path):
    """Older fakes / converters that expose no `status` must keep working."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class _NoStatus:
        def convert(self, _p):
            return SimpleNamespace(document=_doc_with_one_page())

    assert len(extract_with_docling(pdf, converter=_NoStatus()).regions) == 1


# ---------------------------------------------------------------------------
# bug 5 -- the default PDF backend segfaults on multi-page documents
# ---------------------------------------------------------------------------
# docling-parse v4 raises a native std::bad_alloc in its `preprocess` stage,
# which is *before* OCR and unaffected by page_batch_size. On a 15-page arXiv
# paper it took the whole server down; with OCR off it silently dropped 11 of
# 15 pages. pypdfium2 converts the same file with zero errors. These tests pin
# the backend choice so a refactor cannot quietly restore the default.

def test_converter_is_built_on_the_pypdfium_backend():
    pytest.importorskip("docling")
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from uir_pipeline.docling_extract import _build_converter

    conv = _build_converter()
    opt = conv.format_to_options[InputFormat.PDF]
    assert opt.backend is PyPdfiumDocumentBackend, (
        "PDF conversion must not use docling's default backend: it segfaults"
    )


def test_build_converter_propagates_docling_unavailable(monkeypatch):
    """A missing docling must still raise, not fall through to a default."""
    from uir_pipeline import docling_extract

    def _raise():
        raise DoclingUnavailable("no docling")

    monkeypatch.setattr(docling_extract, "_import_docling_or_raise", _raise)
    with pytest.raises(DoclingUnavailable):
        docling_extract._build_converter()


def test_build_converter_honours_the_ocr_flag():
    pytest.importorskip("docling")
    from docling.datamodel.base_models import InputFormat
    from uir_pipeline.docling_extract import _build_converter

    for ocr in (True, False):
        opts = _build_converter(ocr=ocr).format_to_options[InputFormat.PDF]
        assert opts.pipeline_options.do_ocr is ocr


# ---------------------------------------------------------------------------
# OCR strategy -- scanned PDFs must not convert to a well-formed empty UIR
# ---------------------------------------------------------------------------
# A scan read without OCR does not fail: pypdfium finds no embedded glyphs and
# docling returns a document with no text. `done` over an empty UIR is the same
# silent data loss as PARTIAL_SUCCESS, so `auto` detects it and re-converts.

def _doc_with(n_pages: int, chars_per_page: int):
    return SimpleNamespace(
        pages={i: SimpleNamespace(page_no=i) for i in range(1, n_pages + 1)},
        texts=[_text_item("x" * chars_per_page, i, _BBox(0, 10, 5, 0))
               for i in range(1, n_pages + 1)],
        tables=[], pictures=[],
    )


class _RecordingConverter:
    """Stands in for `_build_converter`; records the ocr= it was asked for."""

    def __init__(self, docs):
        self.docs, self.calls = list(docs), []

    def __call__(self, *, ocr):
        self.calls.append(ocr)
        doc = self.docs[min(len(self.calls) - 1, len(self.docs) - 1)]
        return SimpleNamespace(
            convert=lambda _p: SimpleNamespace(
                document=doc, status=SimpleNamespace(name="SUCCESS"), errors=[]
            )
        )


@pytest.fixture
def _patched(monkeypatch):
    from uir_pipeline import docling_extract

    def install(docs):
        rec = _RecordingConverter(docs)
        monkeypatch.setattr(docling_extract, "_build_converter", rec)
        return rec

    return install


def test_ocr_mode_resolution(monkeypatch):
    from uir_pipeline.docling_extract import _resolve_ocr

    monkeypatch.delenv("DOCLING_OCR", raising=False)
    assert _resolve_ocr() == "auto"
    for on in ("1", "true", "ON", "force"):
        monkeypatch.setenv("DOCLING_OCR", on)
        assert _resolve_ocr() == "on"
    for off in ("0", "false", "Off", "never"):
        monkeypatch.setenv("DOCLING_OCR", off)
        assert _resolve_ocr() == "off"
    monkeypatch.setenv("DOCLING_OCR", "banana")
    assert _resolve_ocr() == "auto", "an unknown value must not disable OCR"


def test_auto_skips_ocr_on_a_born_digital_pdf(tmp_path, monkeypatch, _patched):
    monkeypatch.delenv("DOCLING_OCR", raising=False)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    rec = _patched([_doc_with(3, 800)])

    out = extract_with_docling(pdf)
    assert rec.calls == [False], "a text PDF must not pay for OCR"
    assert len(out.regions) == 3


def test_auto_reconverts_with_ocr_when_the_pdf_looks_scanned(tmp_path, monkeypatch, _patched):
    monkeypatch.delenv("DOCLING_OCR", raising=False)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    # first pass: no glyphs. second pass (with OCR): text appears.
    rec = _patched([_doc_with(3, 0), _doc_with(3, 900)])

    out = extract_with_docling(pdf)
    assert rec.calls == [False, True], "a scan must trigger an OCR re-convert"
    assert sum(len(r["text"]) for r in out.regions) == 2700


def test_auto_does_not_reconvert_a_sparse_but_real_text_pdf(tmp_path, monkeypatch, _patched):
    """A title page + figures still clears the threshold on average."""
    monkeypatch.delenv("DOCLING_OCR", raising=False)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    rec = _patched([_doc_with(4, 60)])

    extract_with_docling(pdf)
    assert rec.calls == [False]


def test_ocr_off_never_reconverts_even_for_a_scan(tmp_path, monkeypatch, _patched):
    monkeypatch.setenv("DOCLING_OCR", "off")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    rec = _patched([_doc_with(3, 0)])

    out = extract_with_docling(pdf)
    assert rec.calls == [False]
    assert out.regions == [] or all(not r["text"] for r in out.regions)


def test_ocr_on_converts_once_with_ocr(tmp_path, monkeypatch, _patched):
    monkeypatch.setenv("DOCLING_OCR", "on")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    rec = _patched([_doc_with(2, 900)])

    extract_with_docling(pdf)
    assert rec.calls == [True], "forced OCR must not do a throwaway first pass"


def test_injected_converter_bypasses_ocr_resolution(tmp_path, monkeypatch):
    """Tests inject a converter; it must be used verbatim, exactly once."""
    monkeypatch.setenv("DOCLING_OCR", "auto")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    calls = []

    class _Conv:
        def convert(self, _p):
            calls.append(1)
            return SimpleNamespace(document=_doc_with(1, 0),
                                   status=SimpleNamespace(name="SUCCESS"), errors=[])

    extract_with_docling(pdf, converter=_Conv())
    assert calls == [1], "an injected converter must not be re-run for OCR"


def test_looks_scanned_survives_a_page_count_of_zero():
    from uir_pipeline.docling_extract import _looks_scanned
    assert _looks_scanned(SimpleNamespace(pages={}, texts=[])) is True


# ---------------------------------------------------------------------------
# bug 6 -- PDF glyph extraction splits decimal points
# ---------------------------------------------------------------------------
# pypdfium reports per-glyph positions; docling joins on advance width, so a
# kerned decimal point becomes its own token. "Attention Is All You Need"
# extracts `Pdrop = 0.1` as `Pdrop = 0 . 1` and `28.4` BLEU as `28 . 4` --
# 15 occurrences in 15 pages. The chat prompt says "quote figures exactly",
# so the model dutifully quoted the corrupted number.

def test_split_decimal_is_rejoined():
    from uir_pipeline.docling_extract import normalize_extracted_text

    assert normalize_extracted_text("Pdrop = 0 . 1 .") == "Pdrop = 0.1 ."
    assert normalize_extracted_text("BLEU score of 28 . 4.") == "BLEU score of 28.4."
    assert normalize_extracted_text("beta1 = 0 . 9 , beta2 = 0 . 98") == "beta1 = 0.9 , beta2 = 0.98"


def test_split_decimal_without_trailing_space_is_rejoined():
    from uir_pipeline.docling_extract import normalize_extracted_text

    assert normalize_extracted_text("instead of 0 .3") == "instead of 0.3"


def test_a_sentence_boundary_before_a_number_is_not_joined():
    """`...in 2017. 5 of them...` must not become `2017.5`.

    Prose never puts a space *before* a period, which is what distinguishes
    the extraction artifact from a real sentence end.
    """
    from uir_pipeline.docling_extract import normalize_extracted_text

    for text in (
        "Published in 2017. 5 of the authors were at Google.",
        "See Table 3. 5 configurations were tried.",
        "Ends here. 1 more thing.",
    ):
        assert normalize_extracted_text(text) == text


def test_normalization_leaves_ordinary_prose_alone():
    from uir_pipeline.docling_extract import normalize_extracted_text

    for text in ("no digits here.", "version 1.0 exactly", "a . b", "3 . x", "x . 4"):
        assert normalize_extracted_text(text) == text


def test_text_of_normalizes_region_text():
    from uir_pipeline.docling_extract import _text_of

    assert _text_of(SimpleNamespace(text="rate of 0 . 1")) == "rate of 0.1"


def test_text_of_normalizes_exported_table_markdown():
    class _Table:
        def export_to_markdown(self):
            return "| BLEU |\n| 28 . 4 |"

    from uir_pipeline.docling_extract import _text_of

    assert "28.4" in _text_of(_Table())


def test_walk_doc_emits_normalized_text():
    doc = SimpleNamespace(
        pages={1: SimpleNamespace(page_no=1)},
        texts=[_text_item("we use a rate of 0 . 1 here", 1, _BBox(0, 10, 5, 0))],
        tables=[], pictures=[],
    )
    (region,) = _walk_doc(doc).regions
    assert "0.1" in region["text"]
    assert "0 . 1" not in region["text"]
