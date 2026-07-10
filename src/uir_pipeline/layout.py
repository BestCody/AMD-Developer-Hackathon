"""layout -- per-page region classification + LayoutLMv3 backend (Phase G).

PLAN \\u00a79 Phase G exit:
    -- loads model once (cached), runs inference per page
    -- returns ``LayoutRegion[]`` with labels from set
       ``{heading, paragraph, table, list, figure, caption, header, footer}``
    -- MPS auto-falls-back to CPU on ``NotImplementedError``

PLAN \\u00a79 Phase G AMD-portability decision:
    -- Uses the **text+bbox-only branch** by setting
       ``LayoutLMv3Config.visual_embed=False`` (then ``from_pretrained``).
    -- Rationale (validated by Phase A.5 spike 2026-07-07): trim model
       size and per-page latency to stay within MVP's <10 s/doc budget.
       Historical worry about Detectron2 conflict on macOS MPS / AMD
       ROCm does NOT apply in ``transformers`` 5.x.
    -- Path A (``visual_embed=True``) is also viable but adds ~250 MB and
       ~30-50% per-page slowdown for ~2 pp accuracy on text-heavy PDFs.
      Default = Path B; expose ``LAYOUTLMV3_USE_VISUAL`` env flag for A<->B
      (Phase G polish, deferred unless MVP accuracy regresses).

PLAN \\u00a79 Phase G coordinate normalization:
    -- LayoutLMv3 expects bboxes on a 0-1000 normalized scale (PDF
       document-image convention).
    -- ``layout.py`` normalizes raw pixel bboxes relative to the page's
       pixel dimensions before model input. **UIR output keeps PIXEL
       coordinates** (PLAN \\u00a78) for round-trip fidelity from PDF
       rendering.

MVP scope (PLAN \\u00a73): the classifier below is **heuristic** (line-
clustering + label rules). Fine-tuned token classification on LayoutLMv3's
hidden states is Phase 2 work -- the ``LayoutLMv3Backend`` here exercises
the actual model graph for downstream feature reuse but does NOT publish
labels from the model head.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uir_pipeline.ocr import DetectedWord, OCRPage

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Heuristic thresholds (px). Kept conservative to avoid mis-grouping on
# common M-series PDF page widths (~612 px letter, ~595 px A4).
_HEADER_Y_PX: int = 80
_FOOTER_Y_PX: int = 80
_LINE_CLUSTER_PX: int = 12
_REGION_GAP_PX: int = 24
_HEADING_MAX_WORDS: int = 6

# LayoutLMv3 default (per PLAN \\u00a76 -- standard transformers hub id).
_DEFAULT_LAYOUTLMV3_MODEL_ID: str = "microsoft/layoutlmv3-base"
#: Public alias. ``__all__`` has always named this, but only the private form
#: existed, so ``from uir_pipeline.layout import *`` raised AttributeError.
DEFAULT_LAYOUTLMV3_MODEL_ID: str = _DEFAULT_LAYOUTLMV3_MODEL_ID


# ----------------------------------------------------------------------------
# Label enum (string-valued so it serializes cleanly to UIR JSON).
# ----------------------------------------------------------------------------

class LayoutLabel(str, Enum):
    """Region labels per PLAN \\u00a79 Phase G spec."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"


# ----------------------------------------------------------------------------
# Result type (frozen dataclass per PLAN \\u00a75)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutRegion:
    """One spatially-coherent region on a page.

    Fields:
        label -- one of the seven ``LayoutLabel`` values
        text  -- concatenated words (in left-to-right order on the line)
        confidence -- mean per-word OCR confidence [0, 1]
        bbox  -- pixel rectangle ``(x1, y1, x2, y2)`` covering all words
        page  -- 1-based page number (matches OCRPage.page_number)
        reading_order -- 1-based ordering within the page (top-down)
    """
    label: LayoutLabel
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    page: int
    reading_order: int


# ----------------------------------------------------------------------------
# Heuristic classifier (MVP)
# ----------------------------------------------------------------------------

def _text_has_terminal_punct(text: str) -> bool:
    """True iff ``text`` ends with sentence-final punctuation."""
    if not text:
        return False
    return text.rstrip().endswith((".", "!", "?", ":", ";"))


class LayoutClassifier:
    """Heuristic region-grouping + labeling for MVP (PLAN \\u00a79 Phase G).

    Algorithm:
        1. Sort words top-to-bottom, then left-to-right.
        2. Cluster words into lines by y-proximity (``LINE_CLUSTER_PX``).
        3. Cluster lines into regions by vertical gap (``REGION_GAP_PX``).
        4. Label each region:
            - First region + within ``HEADER_Y_PX`` of top -> ``header``
            - Last region + within ``FOOTER_Y_PX`` of bottom -> ``footer``
            - Short region (<= ``HEADING_MAX_WORDS``) without sentence-final
              punctuation -> ``heading``
            - Otherwise -> ``paragraph``
        5. ``list`` / ``table`` / ``figure`` / ``caption`` are Phase H/J
           extensions (regex / layout heuristics) -- not produced here
           in MVP. Tests assert this contract.

    The classifier is a pure function of (OCRPage, page_height_px) and
    has no model dependency, so it is fast and trivially testable.
    """

    #: Top-of-page cutoff for the HEADER label.
    header_y_px: int = _HEADER_Y_PX
    #: Bottom-of-page cutoff for the FOOTER label.
    footer_y_px: int = _FOOTER_Y_PX
    #: Vertical gap within which words share a line.
    line_cluster_px: int = _LINE_CLUSTER_PX
    #: Vertical gap within which lines share a region.
    region_gap_px: int = _REGION_GAP_PX
    #: Maximum word count for a region to be considered a HEADING.
    heading_max_words: int = _HEADING_MAX_WORDS

    def classify(
        self,
        page: "OCRPage",
        page_height_px: int,
    ) -> list[LayoutRegion]:
        if not page.words:
            return []

        # 1. Reading order sort.
        sorted_words = sorted(page.words, key=lambda w: (w.bbox[1], w.bbox[0]))

        # 2. Lines via y-clustering.
        lines = self._cluster_into_lines(sorted_words)

        # 3. Regions via vertical gap.
        regions = self._cluster_into_regions(lines)
        n = len(regions)

        # 4. Label + assemble.
        out: list[LayoutRegion] = []
        for i, (y_top, y_bot, words) in enumerate(regions):
            text = " ".join(w.text for w in sorted(words, key=lambda w: w.bbox[0]))
            avg_conf = sum(w.confidence for w in words) / max(1, len(words))
            label = self._label(y_top, y_bot, words, page_height_px, i, n, text)
            xs_min = min(w.bbox[0] for w in words)
            xs_max = max(w.bbox[2] for w in words)
            out.append(
                LayoutRegion(
                    label=label,
                    text=text,
                    confidence=max(0.0, min(1.0, avg_conf)),
                    bbox=(xs_min, y_top, xs_max, y_bot),
                    page=page.page_number,
                    reading_order=i + 1,
                )
            )
        return out

    # ------------------------------------------------------------------ helpers

    def _cluster_into_lines(self, words: list["DetectedWord"]) -> list[list["DetectedWord"]]:
        lines: list[list["DetectedWord"]] = []
        for w in words:
            if lines and abs(w.bbox[1] - lines[-1][-1].bbox[1]) < self.line_cluster_px:
                lines[-1].append(w)
            else:
                lines.append([w])
        return lines

    def _cluster_into_regions(
        self,
        lines: list[list["DetectedWord"]],
    ) -> list[tuple[int, int, list["DetectedWord"]]]:
        regions: list[tuple[int, int, list["DetectedWord"]]] = []
        for line in lines:
            y_top = min(w.bbox[1] for w in line)
            y_bot = max(w.bbox[3] for w in line)
            if regions and (y_top - regions[-1][1]) < self.region_gap_px:
                prev_top, prev_bot, prev_words = regions[-1]
                regions[-1] = (
                    prev_top,
                    max(prev_bot, y_bot),
                    prev_words + line,
                )
            else:
                regions.append((y_top, y_bot, line))
        return regions

    def _label(
        self,
        y_top: int,
        y_bot: int,
        words: list["DetectedWord"],
        page_height_px: int,
        idx: int,
        n: int,
        text: str,
    ) -> LayoutLabel:
        # Header / Footer rules require positional + bookend context.
        if idx == 0 and y_top < self.header_y_px:
            return LayoutLabel.HEADER
        if idx == n - 1 and (page_height_px - y_bot) < self.footer_y_px:
            return LayoutLabel.FOOTER
        # Heading rule: short and unsentenced.
        if (
            len(words) <= self.heading_max_words
            and not _text_has_terminal_punct(text)
        ):
            return LayoutLabel.HEADING
        return LayoutLabel.PARAGRAPH


# ----------------------------------------------------------------------------
# LayoutLMv3 backend (Path B -- lazy; MPS -> CPU auto-fallback)
# ----------------------------------------------------------------------------

class LayoutLMv3Backend:
    """Lazy-loaded LayoutLMv3 model in Path B (visual_embed=False).

    The actual model forward pass is consumed by Phase K (embed.py) for
    features; Phase G does not surface labels from the model head. This
    class exists to:
        -- Exercise the actual model graph (validates PLAN \\u00a79 Phase G's
           "loads model once (cached), runs inference per page" half).
        -- Auto-fallback MPS -> CPU on ``NotImplementedError`` per
           PLAN \\u00a79 Phase G.
        -- Provide a ``features()`` shape for downstream Phase K use.

    The model is NOT loaded at ``__init__`` time -- the first ``.to(device)``
    call happens lazily inside ``_ensure_loaded`` so unit tests can
    monkeypatch ``transformers`` without paying the 500 MB download cost.
    """

    DEFAULT_MODEL_ID: str = _DEFAULT_LAYOUTLMV3_MODEL_ID

    def __init__(
        self,
        model_id: str = _DEFAULT_LAYOUTLMV3_MODEL_ID,
        device: str = "cpu",
    ):
        self._model_id = model_id
        self._requested_device = device
        self._model: Any | None = None
        self._config: Any | None = None

    @property
    def model_id(self) -> str:
        """Return the HuggingFace model id being served."""
        return self._model_id

    @property
    def config(self) -> Any:
        """Return the loaded ``LayoutLMv3Config`` (or trigger a load)."""
        if self._config is None:
            self._ensure_loaded()
        return self._config

    def _ensure_loaded(self) -> Any:
        """Lazy model loader + MPS -> CPU fallback (PLAN \\u00a79 Phase G).

        Imports ``transformers`` lazily so test environments without
        network (cold pytest) can still monkeypatch the module.
        """
        if self._model is None:
            from transformers import LayoutLMv3Config, LayoutLMv3Model

            self._config = LayoutLMv3Config.from_pretrained(self._model_id)
            self._config.visual_embed = False  # Path B

            model = LayoutLMv3Model.from_pretrained(
                self._model_id, config=self._config,
            )
            try:
                model = model.to(self._requested_device)
            except NotImplementedError as exc:
                logger.warning(
                    "LayoutLMv3 .to(%s) raised NotImplementedError (%s) -- "
                    "falling back to CPU per PLAN \\u00a79 Phase G.",
                    self._requested_device, exc,
                )
                model = model.to("cpu")
            self._model = model
        return self._model

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Compute last_hidden_state for a batched ``input_ids``/``bbox`` dict.

        Caller is responsible for constructing input tensors (Phase K's
        ``embed.py`` will produce them from ``OCRPage`` data). Output is
        the model's ``last_hidden_state`` tensor, ready for downstream
        pooling.
        """
        model = self._ensure_loaded()
        return model(**inputs)


__all__ = [
    "DEFAULT_LAYOUTLMV3_MODEL_ID",
    "LayoutClassifier",
    "LayoutLabel",
    "LayoutLMv3Backend",
    "LayoutRegion",
]
