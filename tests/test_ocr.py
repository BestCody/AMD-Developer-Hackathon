"""tests/test_ocr.py -- OCR layer (Phase F).

Tests cover:
    -- :func:`polygon_to_bbox`: canonical EasyOCR-shape + degenerate cases.
    -- :class:`DetectedWord` + :class:`OCRPage`: frozen-dataclass immutability.
    -- :class:`EasyOCRReader`: read_page path with monkeypatched ``easyocr.Reader``.
    -- :class:`TesseractReader`: read_page with monkeypatched ``pytesseract.image_to_data``.
    -- :class:`OCREngine`: auto-fallback path on primary exception.
    -- :func:`default_engine`: ``$EASYOCR_LANGS`` env-var parsing.

The tests do NOT exercise real easyocr/pytesseract -- both readers are
mocked at the heavy-dep boundary so pytest is fast and deterministic.
"""
from __future__ import annotations

import pytest

from uir_pipeline.ocr import (
    DetectedWord,
    EasyOCRReader,
    OCREngine,
    OCRPage,
    TesseractReader,
    default_engine,
    polygon_to_bbox,
)


# ----------------------------------------------------------------------------
# Fixtures / builders
# ----------------------------------------------------------------------------

def _word(text: str, conf: float, x1: int, y1: int, x2: int, y2: int, page: int = 1) -> DetectedWord:
    """Convenience constructor for a DetectedWord."""
    return DetectedWord(text=text, confidence=conf, bbox=(x1, y1, x2, y2), page=page)


# ----------------------------------------------------------------------------
# polygon_to_bbox
# ----------------------------------------------------------------------------

def test_polygon_to_bbox_canonical_easyocr_shape():
    """EasyOCR returns 4-point polygons -- canonical CCW rectangle."""
    poly = [(100, 50), (300, 50), (300, 90), (100, 90)]
    assert polygon_to_bbox(poly) == (100, 50, 300, 90)


def test_polygon_to_bbox_unordered_points():
    """Order of vertices is irrelevant -- we take min/max of xs / ys."""
    poly = [(300, 90), (100, 50), (100, 90), (300, 50)]
    assert polygon_to_bbox(poly) == (100, 50, 300, 90)


def test_polygon_to_bbox_with_more_than_four_points():
    """Some OCR engines (none in MVP) may emit >4 points. Function still works."""
    poly = [(0, 0), (50, 0), (50, 50), (50, 50), (0, 50)]  # a duplicate
    assert polygon_to_bbox(poly) == (0, 0, 50, 50)


def test_polygon_to_bbox_zero_size():
    poly = [(100, 100), (100, 100), (100, 100), (100, 100)]
    assert polygon_to_bbox(poly) == (100, 100, 100, 100)


def test_polygon_to_bbox_negative_coordinates():
    """Defensive: handle negative coords (rare, but possible after affine)."""
    poly = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    assert polygon_to_bbox(poly) == (-10, -10, 10, 10)


def test_polygon_to_bbox_coerces_to_int():
    """Any (RealStrOfInts, RealStrOfInts) tuple OK; coerces to int."""
    poly = [(100.0, 50.0), (300.0, 50.0), (300.0, 90.0), (100.0, 90.0)]
    assert polygon_to_bbox(poly) == (100, 50, 300, 90)


# ----------------------------------------------------------------------------
# DetectedWord + OCRPage
# ----------------------------------------------------------------------------

def test_detected_word_is_frozen():
    w = _word("hi", 0.9, 0, 0, 10, 10)
    with pytest.raises((AttributeError, TypeError)):
        w.text = "no"  # type: ignore[misc]


def test_detected_word_is_hashable():
    w1 = _word("hi", 0.9, 0, 0, 10, 10)
    w2 = _word("hi", 0.9, 0, 0, 10, 10)
    assert hash(w1) == hash(w2)
    assert {w1, w2} == {w1}  # set dedup works


def test_ocr_page_is_frozen():
    page = OCRPage(page_number=1, words=(_word("a", 0.9, 0, 0, 10, 10),))
    with pytest.raises((AttributeError, TypeError)):
        page.page_number = 2  # type: ignore[misc]


def test_ocr_page_preserves_word_count():
    page = OCRPage(
        page_number=3,
        words=(
            _word("a", 0.9, 0, 0, 10, 10),
            _word("b", 0.5, 0, 0, 10, 10),
        ),
    )
    assert page.page_number == 3
    assert len(page.words) == 2


def test_ocr_page_accepts_empty_words():
    page = OCRPage(page_number=1, words=())
    assert page.words == ()


# ----------------------------------------------------------------------------
# EasyOCRReader (mocked)
# ----------------------------------------------------------------------------

class _StubEasyOCRRaw:
    """Stand-in for ``easyocr.Reader`` -- returns canned EasyOCR shapes."""

    def __init__(self, canned):
        self._canned = canned  # list[tuple[polygon, text, confidence]]
        self.calls = 0

    def readtext(self, image):
        self.calls += 1
        return list(self._canned)


@pytest.fixture
def stub_easyocr():
    """Patch ``easyocr.Reader`` so EasyOCRReader can call into a stub.

    We install a fake ``easyocr`` module on ``sys.modules``; the
    lazy ``import easyocr`` inside :class:`EasyOCRReader._ensure_reader`
    picks it up via the import cache. (We do NOT mock numpy --
    ``EasyOCRReader.read_page`` does its own function-local
    ``import numpy as np``, and NumPy 2.x falls back to ``dtype=object``
    for non-array-like inputs, which is sufficient for the unit tests.)
    """
    import sys
    import types

    orig_easyocr = sys.modules.get("easyocr")
    module = types.ModuleType("easyocr")
    module.Reader = lambda languages, gpu=False, verbose=False: _StubEasyOCRRaw(canned=[])  # type: ignore[attr-defined]
    sys.modules["easyocr"] = module
    yield
    # Teardown -- restore eager easyocr or drop the stub.
    if orig_easyocr is not None:
        sys.modules["easyocr"] = orig_easyocr
    else:
        sys.modules.pop("easyocr", None)


def test_easyocr_reader_returns_ocr_page(stub_easyocr):
    reader = EasyOCRReader(languages=("en",), gpu=False, verbose=False)
    # Inject canned results INTO the freshly-created stub.
    reader._reader = _StubEasyOCRRaw(canned=[
        ([(100, 50), (300, 50), (300, 90), (100, 90)], "Hello", 0.95),
        ([(310, 50), (500, 50), (500, 90), (310, 90)], "world", 0.88),
    ])
    # PIL-like stub that has ``convert`` method.
    image_stub = type("_Img", (), {"convert": lambda self, mode: self})()

    page = reader.read_page(image_stub, page_number=1)

    assert isinstance(page, OCRPage)
    assert page.page_number == 1
    assert len(page.words) == 2
    assert page.words[0].text == "Hello"
    assert page.words[0].confidence == pytest.approx(0.95)
    assert page.words[0].bbox == (100, 50, 300, 90)
    assert page.words[1].text == "world"
    assert page.words[1].bbox == (310, 50, 500, 90)


def test_easyocr_reader_trims_whitespace(stub_easyocr):
    reader = EasyOCRReader()
    reader._reader = _StubEasyOCRRaw(canned=[
        ([(0, 0), (10, 0), (10, 10), (0, 10)], "   spaced   ", 0.9),
    ])
    image_stub = type("_Img", (), {"convert": lambda self, mode: self})()
    page = reader.read_page(image_stub, 1)
    assert page.words[0].text == "spaced"


def test_easyocr_reader_skips_empty_strings(stub_easyocr):
    reader = EasyOCRReader()
    reader._reader = _StubEasyOCRRaw(canned=[
        ([(0, 0), (10, 0), (10, 10), (0, 10)], "   ", 0.9),
        ([(10, 0), (20, 0), (20, 10), (10, 10)], "real", 0.9),
    ])
    image_stub = type("_Img", (), {"convert": lambda self, mode: self})()
    page = reader.read_page(image_stub, 1)
    assert len(page.words) == 1
    assert page.words[0].text == "real"


def test_easyocr_reader_clamps_confidence(stub_easyocr):
    """Defensive clamp -- EasyOCR confidences can be >1 or <0 in edge cases."""
    reader = EasyOCRReader()
    reader._reader = _StubEasyOCRRaw(canned=[
        ([(0, 0), (10, 0), (10, 10), (0, 10)], "up",   1.50),  # > 1
        ([(10, 0), (20, 0), (20, 10), (10, 10)], "down", -0.10),  # < 0
    ])
    image_stub = type("_Img", (), {"convert": lambda self, mode: self})()
    page = reader.read_page(image_stub, 1)
    assert page.words[0].confidence == 1.0
    assert page.words[1].confidence == 0.0


def test_easyocr_reader_name_is_easyocr():
    assert EasyOCRReader.name == "easyocr"


def test_easyocr_reader_attributes_are_immutable_internals():
    """``_reader`` is None at construction; populated on first call."""
    r = EasyOCRReader(languages=("en", "fr"), gpu=True, verbose=False)
    assert r._reader is None
    assert r._languages == ("en", "fr")
    assert r._gpu is True
    assert r._verbose is False


def test_easyocr_reader_bare_str_wrapped_to_one_tuple():
    """EasyOCRReader(languages="en") would otherwise produce ("e", "n") via tuple(str)."""
    r = EasyOCRReader(languages="en", gpu=False, verbose=False)
    assert r._languages == ("en",)


# ----------------------------------------------------------------------------
# TesseractReader (mocked)
# ----------------------------------------------------------------------------

class _StubImageOps:
    """Stand-in for PIL Image (any object supporting ``convert``)."""

    def __init__(self, underlying):
        self._underlying = underlying

    def convert(self, mode):
        return self._underlying


@pytest.fixture
def stub_pytesseract():
    """Install a fake pytesseract module + PIL into ``sys.modules``.

    Save-and-restore originals so other tests after this one can re-import
    the real packages without paying the cold-import cost.
    """
    import sys
    import types

    orig_pt = sys.modules.get("pytesseract")
    orig_pil = sys.modules.get("PIL")
    orig_pil_image = sys.modules.get("PIL.Image")

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.image_to_data = None  # type: ignore[attr-defined]  # set per-test
    pt_mod.Output = types.SimpleNamespace(DICT="dict")  # type: ignore[attr-defined]
    sys.modules["pytesseract"] = pt_mod

    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.fromarray = None  # type: ignore[attr-defined]  # set per-test
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = image_mod
    yield pt_mod, image_mod

    # Teardown.
    for name, orig in [
        ("pytesseract", orig_pt),
        ("PIL", orig_pil),
        ("PIL.Image", orig_pil_image),
    ]:
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


def _install_pytesseract_data(stub_pytesseract, data):
    """Wire up the canned ``image_to_data`` return shape."""
    pt_mod, image_mod = stub_pytesseract
    pt_mod.image_to_data = lambda img, lang=None, output_type=None: data
    image_mod.fromarray = lambda arr: arr


def _tesseract_dict_row(text: str, conf: float, left: int, top: int, width: int, height: int) -> dict:
    """Build one row of the Tesseract dict-output shape."""
    return {
        "text": [text], "conf": [conf], "left": [left],
        "top": [top], "width": [width], "height": [height],
    }


def test_tesseract_reader_happy_path(stub_pytesseract):
    data = _tesseract_dict_row("Hello", 95, 10, 20, 50, 20)
    _install_pytesseract_data(stub_pytesseract, data)
    r = TesseractReader()
    page = r.read_page(_StubImageOps("img"), page_number=1)
    assert page.words[0].text == "Hello"
    assert page.words[0].confidence == pytest.approx(0.95)
    assert page.words[0].bbox == (10, 20, 60, 40)
    assert page.page_number == 1


def test_tesseract_reader_drops_negative_confidence(stub_pytesseract):
    """Tesseract's '-1' for "no confidence estimate" must be filtered out."""
    data = _tesseract_dict_row("?uncertain?", -1, 0, 0, 10, 10)
    _install_pytesseract_data(stub_pytesseract, data)
    r = TesseractReader()
    page = r.read_page(_StubImageOps("img"), 1)
    assert page.words == ()


def test_tesseract_reader_drops_empty_text(stub_pytesseract):
    data = _tesseract_dict_row("", 95, 0, 0, 10, 10)
    _install_pytesseract_data(stub_pytesseract, data)
    r = TesseractReader()
    page = r.read_page(_StubImageOps("img"), 1)
    assert page.words == ()


def test_tesseract_reader_normalizes_tesseract_0_to_100_scale(stub_pytesseract):
    data = _tesseract_dict_row("OK", 75, 0, 0, 10, 10)
    _install_pytesseract_data(stub_pytesseract, data)
    r = TesseractReader()
    page = r.read_page(_StubImageOps("img"), 1)
    assert page.words[0].confidence == pytest.approx(0.75)


def test_tesseract_reader_handles_multiple_words(stub_pytesseract):
    data = {
        "text": ["Hello", "world", "foo"],
        "conf": [90.0, 85.0, 70.0],
        "left": [10, 60, 110], "top": [20, 20, 30],
        "width": [40, 50, 30], "height": [20, 20, 18],
    }
    _install_pytesseract_data(stub_pytesseract, data)
    r = TesseractReader()
    page = r.read_page(_StubImageOps("img"), 1)
    assert [w.text for w in page.words] == ["Hello", "world", "foo"]


def test_tesseract_reader_multi_row_to_single_word(stub_pytesseract):
    """``image_to_data`` returns a single dict with N rows -- reader must iterate."""
    data = {
        "text": ["alpha", "beta", "gamma"],
        "conf": [80.0, 85.0, 90.0],
        "left": [0, 100, 200], "top": [10, 10, 10],
        "width": [80, 80, 80], "height": [12, 12, 12],
    }
    _install_pytesseract_data(stub_pytesseract, data)
    page = TesseractReader().read_page(_StubImageOps("img"), 1)
    assert len(page.words) == 3


def test_tesseract_reader_name_is_tesseract():
    assert TesseractReader.name == "tesseract"


# ----------------------------------------------------------------------------
# OCREngine (auto-fallback)
# ----------------------------------------------------------------------------

class _StubReader:
    """Generic stub reader with configurable behavior.

    Mirrors real-reader semantics: returns an OCRPage stamped with the
    requested ``page_number`` (callers in production code honor it).
    """

    def __init__(self, *, name: str, page_or_exc):
        self.name = name
        self._result = page_or_exc
        self.last_page_number: int | None = None

    def read_page(self, image, page_number):
        self.last_page_number = page_number
        if isinstance(self._result, BaseException) or (
            isinstance(self._result, type) and issubclass(self._result, BaseException)
        ):
            raise self._result
        if isinstance(self._result, OCRPage):
            # Stamp the requested page_number onto a fresh OCRPage.
            return OCRPage(
                page_number=page_number,
                words=self._result.words,
            )
        return self._result


def test_engine_returns_primary_output_on_success():
    primary = _StubReader(name="A", page_or_exc=OCRPage(page_number=1, words=()))
    fallback = _StubReader(name="B", page_or_exc=OCRPage(page_number=99, words=()))
    engine = OCREngine(primary=primary, fallback=fallback)
    page = engine.read_page("img", 7)
    assert page.page_number == 7  # primary wins, honors caller


def test_engine_falls_back_on_primary_exception():
    fallback_page = OCRPage(page_number=99, words=(
        DetectedWord(text="back", confidence=0.5, bbox=(0, 0, 1, 1), page=99),
    ))
    primary = _StubReader(name="A", page_or_exc=ValueError("easyocr broke"))
    fallback = _StubReader(name="B", page_or_exc=fallback_page)
    engine = OCREngine(primary=primary, fallback=fallback)
    page = engine.read_page("img", 7)
    assert page.page_number == 7  # fallback wins, honors caller
    assert page.words[0].text == "back"


def test_engine_re_raises_when_no_fallback():
    primary = _StubReader(name="A", page_or_exc=RuntimeError("boom"))
    engine = OCREngine(primary=primary, fallback=None)
    with pytest.raises(RuntimeError, match="boom"):
        engine.read_page("img", 1)


def test_engine_names_exposed():
    primary = _StubReader(name="EASY", page_or_exc=OCRPage(page_number=1, words=()))
    fallback = _StubReader(name="TESS", page_or_exc=OCRPage(page_number=2, words=()))
    engine = OCREngine(primary=primary, fallback=fallback)
    assert engine.primary_name == "EASY"
    assert engine.fallback_name == "TESS"


def test_engine_fallback_name_none_when_only_primary():
    primary = _StubReader(name="EASY", page_or_exc=OCRPage(page_number=1, words=()))
    engine = OCREngine(primary=primary, fallback=None)
    assert engine.fallback_name is None


# ----------------------------------------------------------------------------
# default_engine factory
# ----------------------------------------------------------------------------

def test_default_engine_reads_env_var(monkeypatch):
    monkeypatch.setenv("EASYOCR_LANGS", "en,fr")
    engine = default_engine(
        languages=None, gpu=False, page_timeout_s=12.0,
    )
    assert engine.primary_name == "easyocr"
    assert engine.fallback_name == "tesseract"
    # The EasyOCRReader is built but not initialized until use -- confirm
    # languages propagated.
    assert engine._primary._languages == ("en", "fr")
    assert engine._timeout_s == 12.0


def test_default_engine_with_explicit_languages(monkeypatch):
    monkeypatch.setenv("EASYOCR_LANGS", "en")  # env ignored when explicit
    engine = default_engine(languages=("fr",), gpu=False)
    assert engine._primary._languages == ("fr",)


def test_default_engine_env_var_blank_falls_back(monkeypatch):
    monkeypatch.setenv("EASYOCR_LANGS", "")
    engine = default_engine(languages=None)
    # Empty / blank -> fall back to default ("en").
    assert engine._primary._languages == ("en",)
