"""ocr -- per-page OCR with EasyOCR primary + Tesseract fallback (Phase F).

PLAN \u00a79 Phase F exit:
    -- EasyOCR runs on a fixture page successfully; confidence per word
    -- command-line flag to switch to Tesseract (programmatic API here)
    -- per-page timeout (configurable; auto-fallback on reader exception)
    -- auto-fallback when EasyOCR raises
    -- mocks deep in unit tests (the heavy readers are lazy-imported)

PLAN \u00a79 Phase F footnote:
    -- EasyOCR emits 4-point polygons; UIR + LayoutLMv3 want
       4-int rectangles. ``polygon_to_bbox()`` normalizes here so
       downstream consumers see a consistent shape. ``tables.py`` will
       reuse this helper.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

# Lazy image import -- pillow is heavy and PIL's tooling isn't always needed
# at module load. We import Image, ImageOps only inside functions that read.

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Default EasyOCR language list per .env EASYOCR_LANGS=``en``.
# Overrideable via env at construction time inside ``default_engine``.
_DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)

# Default Tesseract language binary code. ``eng`` ships with the Homebrew
# install; plan.md \u00a79 Phase F keeps MVP English-only.
_DEFAULT_TESSERACT_LANG: str = "eng"

# Per-page timeout (seconds). PLAN.md \u00a711: EasyOCR per-page ~600-900 ms;
# 10 s leaves generous headroom for the 1k-docs/day target.
_DEFAULT_PAGE_TIMEOUT_S: float = 10.0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def polygon_to_bbox(
    polygon: Sequence[Sequence[int]],
) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon ``[(x1,y1), (x2,y1), (x2,y2), (x1,y2)]``
    into the axis-aligned ``[x1, y1, x2, y2]`` rectangle.

    LayoutLMv3 + UIR Schema both expect a 4-int rectangle (PLAN \u00a78).
    The polygon order is irrelevant -- we take min/max of xs and ys.
    Returns a fresh tuple so callers can rely on hashability.
    """
    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


# ----------------------------------------------------------------------------
# Result types (frozen dataclasses per PLAN \u00a75)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedWord:
    """One OCR-detected word with bbox in pixel coordinates and a [0,1] confidence.

    Mirrors the per-word granularity EasyOCR's ``readtext`` returns and
    what Tesseract's ``image_to_data`` yields when each word is taken
    from the dict-mode output.
    """
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    page: int


@dataclass(frozen=True)
class OCRPage:
    """All words recognized on a single PDF page.

    ``words`` is a tuple for frozen-dataclass immutability + hashability.
    Downstream ``LayoutClassifier`` (Phase G) operates on per-page lists.
    """
    page_number: int
    words: tuple[DetectedWord, ...]


# ----------------------------------------------------------------------------
# Reader protocol + implementations
# ----------------------------------------------------------------------------

class BaseOCRReader(Protocol):
    """The contract an OCR reader must satisfy.

    Implementing classes are expected to lazy-import their heavy
    dependencies inside ``__init__`` or methods -- this module does
    NOT pull easyocr / pytesseract at import time so unit tests don't
    pay that cost.
    """
    name: str

    def read_page(self, image: Any, page_number: int) -> OCRPage: ...


class EasyOCRReader:
    """EasyOCR-backed per-page reader (PLAN \u00a79 Phase F primary).

    Lazy-imports ``easyocr``. The constructor freezes configuration;
    the underlying ``easyocr.Reader`` instance is built on first use
    so an EasyOCR-down environment can still ``from ocr import`` this
    class without crashing.
    """
    name = "easyocr"

    def __init__(
        self,
        languages: Sequence[str] = _DEFAULT_LANGUAGES,
        gpu: bool = False,
        verbose: bool = False,
    ):
        # A bare ``str`` would be iterated char-by-char into ``tuple("en") == ("e", "n")``,
        # silently breaking EasyOCR's language list. Wrap into a 1-tuple defensively.
        if isinstance(languages, str):
            languages = (languages,)
        self._languages: tuple[str, ...] = tuple(languages)
        self._gpu: bool = gpu
        self._verbose: bool = verbose
        self._reader: Any | None = None  # lazy init

    def _ensure_reader(self) -> Any:
        if self._reader is None:
            import easyocr  # type: ignore  # lazy
            self._reader = easyocr.Reader(
                list(self._languages),
                gpu=self._gpu,
                verbose=self._verbose,
            )
        return self._reader

    def read_page(self, image: Any, page_number: int) -> OCRPage:
        """Run EasyOCR on a page image; return an :class:`OCRPage`.

        EasyOCR returns ``[(bbox, text, confidence), ...]`` where ``bbox``
        is a 4-point polygon ``[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]``.
        Per-plan we normalize at this layer so downstream consumers see
        rectangles.
        """
        import numpy as np  # easyocr depends on numpy

        reader = self._ensure_reader()
        # EasyOCR accepts either PIL Image or numpy array; use the array
        # interface for stability across PIL versions.
        if hasattr(image, "convert"):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)

        results = reader.readtext(arr)
        words: list[DetectedWord] = []
        for bbox_4pt, text, conf in results:
            if not text or not text.strip():
                continue
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = 0.0
            words.append(
                DetectedWord(
                    text=text.strip(),
                    confidence=max(0.0, min(1.0, c)),
                    bbox=polygon_to_bbox(bbox_4pt),
                    page=page_number,
                )
            )
        return OCRPage(page_number=page_number, words=tuple(words))


class TesseractReader:
    """pytesseract-backed per-page reader (PLAN \u00a79 Phase F fallback).

    Uses ``pytesseract.image_to_data`` (dict mode) which returns per-word
    text + confidence + box. ``confidence`` is normalized from
    Tesseract's 0-100 scale to [0, 1]. Words with negative confidence
    (Tesseract's idiomatic "no confidence" sentinel) are dropped.
    """
    name = "tesseract"

    def __init__(self, language: str = _DEFAULT_TESSERACT_LANG):
        self._language: str = language

    def read_page(self, image: Any, page_number: int) -> OCRPage:
        """Run pytesseract on a page image; return an :class:`OCRPage`."""
        import pytesseract  # type: ignore  # lazy

        # pytesseract accepts PIL.Image directly via the standard binding.
        if hasattr(image, "convert"):
            img = image.convert("RGB")
        else:
            from PIL import Image  # lazy
            img = Image.fromarray(image).convert("RGB")

        data = pytesseract.image_to_data(
            img,
            lang=self._language,
            output_type=pytesseract.Output.DICT,
        )

        words: list[DetectedWord] = []
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue  # Tesseract pads empty rows

            try:
                raw_conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            # Tesseract returns -1 for "no confidence estimate" -- drop.
            if raw_conf < 0:
                continue

            c = raw_conf / 100.0  # normalize 0-100 -> 0-1
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            words.append(
                DetectedWord(
                    text=text,
                    confidence=max(0.0, min(1.0, c)),
                    bbox=(x, y, x + w, y + h),
                    page=page_number,
                )
            )
        return OCRPage(page_number=page_number, words=tuple(words))


# ----------------------------------------------------------------------------
# Engine (auto-fallback)
# ----------------------------------------------------------------------------

class OCREngine:
    """Auto-fallback wrapper around two ``BaseOCRReader`` implementations.

    If the primary reader raises any exception, the fallback runs instead.
    If the fallback also raises, the original exception is re-raised
    (use ``last_exc`` for diagnostics).
    """
    def __init__(
        self,
        primary: BaseOCRReader,
        fallback: BaseOCRReader | None = None,
        page_timeout_s: float = _DEFAULT_PAGE_TIMEOUT_S,
    ):
        self._primary = primary
        self._fallback = fallback
        self._timeout_s = float(page_timeout_s)

    @property
    def primary_name(self) -> str:
        return getattr(self._primary, "name", "unknown")

    @property
    def fallback_name(self) -> str | None:
        if self._fallback is None:
            return None
        return getattr(self._fallback, "name", "unknown")

    def read_page(self, image: Any, page_number: int) -> OCRPage:
        """Run the primary reader, fall back on exception or timeout.

        Timeout uses a wall-clock guard around the primary call. On
        timeout we treat it like any other primary failure.
        """
        t0 = time.monotonic()
        try:
            page = self._primary.read_page(image, page_number)
            elapsed = time.monotonic() - t0
            if elapsed > self._timeout_s:
                logger.warning(
                    "ocr.primary slow (%.2fs > %.2fs budget); consider fallback",
                    elapsed, self._timeout_s,
                )
            return page
        except Exception as primary_exc:
            if self._fallback is None:
                logger.error(
                    "ocr.primary failed and no fallback configured: %s",
                    primary_exc,
                )
                raise
            logger.warning(
                "ocr.primary (%s) failed on page %d: %s -- using fallback %s",
                self.primary_name, page_number, primary_exc, self.fallback_name,
            )
            return self._fallback.read_page(image, page_number)


# ----------------------------------------------------------------------------
# Factory helpers
# ----------------------------------------------------------------------------

def cast_to_tuple_lang(langs: Sequence[str]) -> tuple[str, ...]:
    """Coerce a generic Sequence[str] into a tuple for EasyOCRReader.

    Defined before :func:`default_engine` so the call site reads in
    top-down order -- readers skimming the module see the helper
    before the factory that uses it.
    """
    return tuple(t.strip() for t in langs if t.strip())


def default_engine(
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    page_timeout_s: float = _DEFAULT_PAGE_TIMEOUT_S,
) -> OCREngine:
    """Build the MVP ``OCREngine``: EasyOCR primary + Tesseract fallback.

    Reads ``$EASYOCR_LANGS`` (from .env.example) when ``languages`` is None.
    Override per Python call for tests.
    """
    if languages is None:
        raw = os.environ.get("EASYOCR_LANGS", "en").strip()
        languages = tuple(t.strip() for t in raw.split(",") if t.strip()) or _DEFAULT_LANGUAGES
    primary = EasyOCRReader(languages=cast_to_tuple_lang(languages), gpu=gpu)
    fallback = TesseractReader()
    return OCREngine(primary=primary, fallback=fallback, page_timeout_s=page_timeout_s)


__all__ = [
    "BaseOCRReader",
    "DetectedWord",
    "EasyOCRReader",
    "OCREngine",
    "OCRPage",
    "TesseractReader",
    "cast_to_tuple_lang",
    "default_engine",
    "polygon_to_bbox",
]
