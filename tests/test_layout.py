"""tests/test_layout.py -- layout region classification (Phase G).

Tests cover:
    -- :class:`LayoutLabel` enum: values match the UIR spec set.
    -- :class:`LayoutRegion` dataclass: frozen + serializable.
    -- :class:`LayoutClassifier` heuristic: positional HEADER/FOOTER,
       HEADING short + unsentenced, default PARAGRAPH.
    -- :class:`LayoutLMv3Backend` lazy loader: visual_embed=False,
       MPS -> CPU auto-fallback on NotImplementedError.

The LayoutLMv3 model is NOT loaded -- ``transformers`` is monkeypatched
so the tests stay fast and deterministic (no 500 MB download).
"""
from __future__ import annotations

import sys
import types
import pytest

from uir_pipeline.layout import (
    LayoutClassifier,
    LayoutLabel,
    LayoutLMv3Backend,
    LayoutRegion,
)
from uir_pipeline.ocr import DetectedWord, OCRPage


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _w(text: str, conf: float = 0.95, y: int = 0, x1: int = 0, x2: int = 100,
       page: int = 1) -> DetectedWord:
    """Convenience: a DetectedWord at row y, columns [x1, x2]."""
    return DetectedWord(text=text, confidence=conf, bbox=(x1, y, x2, y + 12), page=page)


def _page(*words: DetectedWord, page_number: int = 1) -> OCRPage:
    return OCRPage(page_number=page_number, words=tuple(words))


# ----------------------------------------------------------------------------
# LayoutLabel
# ----------------------------------------------------------------------------

def test_layout_label_values_match_spec():
    """The 8 labels per PLAN \u00a79 Phase G -- all present, exact strings."""
    expected = {"heading", "paragraph", "table", "list", "figure", "caption", "header", "footer"}
    actual = {m.value for m in LayoutLabel}
    assert actual == expected


def test_layout_label_is_string_enum():
    """``LayoutLabel(str, Enum)`` -- ``LayoutLabel.HEADING.value`` returns the spec string for serialization."""
    assert LayoutLabel.HEADING.value == "heading"
    # str(L) returns Python's repr ("LayoutLabel.HEADING") -- downstream
    # call sites use ``.value`` for JSON. Smoke-check both work.
    assert "HEADING" in str(LayoutLabel.HEADING)


# ----------------------------------------------------------------------------
# LayoutRegion
# ----------------------------------------------------------------------------

def test_layout_region_frozen():
    r = LayoutRegion(
        label=LayoutLabel.HEADING, text="A", confidence=0.9,
        bbox=(0, 0, 100, 20), page=1, reading_order=1,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.label = LayoutLabel.PARAGRAPH  # type: ignore[misc]


def test_layout_region_equality():
    """Two LayoutRegion with identical fields are equal (frozen dataclass hash)."""
    a = LayoutRegion(LayoutLabel.HEADING, "Hi", 0.9, (0, 0, 100, 20), 1, 1)
    b = LayoutRegion(LayoutLabel.HEADING, "Hi", 0.9, (0, 0, 100, 20), 1, 1)
    c = LayoutRegion(LayoutLabel.PARAGRAPH, "Hi", 0.9, (0, 0, 100, 20), 1, 1)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


# ----------------------------------------------------------------------------
# LayoutClassifier (heuristic MVP)
# ----------------------------------------------------------------------------

def test_empty_page_returns_empty_regions():
    out = LayoutClassifier().classify(_page(), page_height_px=1000)
    assert out == []


def test_single_word_becomes_one_region():
    p = _page(_w("alpha", y=0))
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert len(out) == 1
    assert out[0].text == "alpha"
    assert out[0].page == 1
    assert out[0].reading_order == 1


def test_words_at_same_y_share_a_line():
    p = _page(
        _w("alpha", y=100),
        _w("beta",  y=100, x1=110, x2=200),
    )
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert len(out) == 1
    assert out[0].text == "alpha beta"
    assert out[0].bbox == (0, 100, 200, 112)


def test_words_far_apart_in_y_split_into_regions():
    """Two lines 50 px apart (> LINE_CLUSTER_PX=12) -> two regions."""
    p = _page(
        _w("line one", y=100),
        _w("line two", y=200),  # > 24 px gap from line 1
    )
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert len(out) == 2
    assert [r.reading_order for r in out] == [1, 2]


def test_header_label_at_top_of_page():
    """First region within HEADER_Y_PX (80) of page top -> HEADER."""
    p = _page(_w("Doc Title", y=10))
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert out[0].label == LayoutLabel.HEADER


def test_footer_label_at_bottom_of_page():
    """Last region within FOOTER_Y_PX (80) of page bottom -> FOOTER.

    ``_w(..., y=910)`` produces bbox ending at y=922; on a 1000 px-page
    that's 78 px from the bottom -- comfortably under the 80 px cutoff.
    """
    p = _page(_w("Body", y=200), _w("page 1 of 10", y=910))
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert out[-1].label == LayoutLabel.FOOTER


def test_short_unsentenced_region_is_heading():
    """<= HEADING_MAX_WORDS (6) words, no terminating punctuation -> HEADING."""
    p = _page(_w("Chapter", y=200), _w("One", y=200, x1=110))
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert out[0].label == LayoutLabel.HEADING


def test_long_sentenced_region_is_paragraph():
    """> HEADING_MAX_WORDS with terminal period -> PARAGRAPH."""
    words = [_w(f"w{i}", y=200, x1=i * 100) for i in range(8)]
    p = _page(*words, _w(".", y=200, x1=800))
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert out[0].label == LayoutLabel.PARAGRAPH


def test_reading_order_is_consecutive():
    """Reading_order is sequential 1..N across the page."""
    p = _page(
        _w("a", y=0), _w("b", y=0, x1=110),
        _w("c", y=200), _w("d", y=200, x1=110),
        _w("e", y=400),
    )
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert [r.reading_order for r in out] == [1, 2, 3]


def test_page_number_propagates_through_each_region():
    p = _page(_w("a", y=0), _w("b", y=400), page_number=7)
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert all(r.page == 7 for r in out)


def test_confidence_is_average_of_word_confidences():
    p = _page(
        _w("hi", conf=0.8, y=300),
        _w("there", conf=1.0, y=300, x1=120),
    )
    out = LayoutClassifier().classify(p, page_height_px=1000)
    expected = (0.8 + 1.0) / 2.0
    assert out[0].confidence == pytest.approx(expected)


def test_confidence_clamped_to_unit_interval():
    p = _page(_w("hi", conf=2.0, y=300))  # artifactual > 1.0
    out = LayoutClassifier().classify(p, page_height_px=1000)
    assert 0.0 <= out[0].confidence <= 1.0


def test_classifier_thresholds_are_tunable():
    """Subclassing lets us override thresholds without recompiling defaults."""
    class Strict(LayoutClassifier):
        heading_max_words = 3
        header_y_px = 30

    # 4-word < 6 = heading per default; > 3 = paragraph per Strict.
    p = _page(
        _w("w0", y=300), _w("w1", y=300, x1=110),
        _w("w2", y=300, x1=210), _w("w3", y=300, x1=310),
    )
    out_default = LayoutClassifier().classify(p, page_height_px=1000)
    assert out_default[0].label == LayoutLabel.HEADING
    out_strict = Strict().classify(p, page_height_px=1000)
    assert out_strict[0].label == LayoutLabel.PARAGRAPH


def test_classifier_returns_only_four_labels_in_mvp():
    """For MVP, only HEADER / FOOTER / HEADING / PARAGRAPH are emitted.
    (table / list / figure / caption are Phase H/J extensions.)
    """
    p = _page(
        _w("Title", y=10),
        _w("Body", y=200),
        _w("footnote", y=910),
    )
    out = LayoutClassifier().classify(p, page_height_px=1000)
    seen = {r.label for r in out}
    assert seen <= {
        LayoutLabel.HEADER, LayoutLabel.FOOTER,
        LayoutLabel.HEADING, LayoutLabel.PARAGRAPH,
    }


# ----------------------------------------------------------------------------
# LayoutLMv3Backend (lazy loader, mocked transformers)
# ----------------------------------------------------------------------------

class _StubLayoutLMv3Config:
    """Stand-in for transformers.LayoutLMv3Config.

    Real ``transformers.LayoutLMv3Config`` defaults ``visual_embed=True``;
    our backend flips it off (Path B per PLAN \u00a79 Phase G). The stub
    doesn't pre-set ``visual_embed`` so the production-side assignment
    is observable on the instance.
    """

    def __init__(self):
        self.model_id: str | None = None
        # NOTE: deliberately not pre-setting visual_embed; the production
        # code does ``self._config.visual_embed = False`` -- test
        # asserts after that assignment.
        self.visual_embed: bool | None = None

    @classmethod
    def from_pretrained(cls, model_id):
        instance = cls()
        instance.model_id = model_id
        return instance


class _StubLayoutLMv3Model:
    """Stand-in for transformers.LayoutLMv3Model with .from_pretrained + .to()."""

    last_to_targets: list[str] = []
    instances: list = []

    def __init__(self):
        self.model_id: str | None = None
        self.config = None
        self._raise_on_to = None  # optional override: device -> bool

    @classmethod
    def from_pretrained(cls, model_id, config=None):
        instance = cls()
        instance.model_id = model_id
        instance.config = config
        cls.instances.append(instance)
        return instance

    def to(self, device):
        if self._raise_on_to is not None and self._raise_on_to(device):
            raise NotImplementedError(f"mock .to({device}) unsupported")
        _StubLayoutLMv3Model.last_to_targets.append(device)
        return self

    def __call__(self, **kwargs):
        return {"last_hidden_state": "stub-tensor"}


@pytest.fixture
def stub_transformers_layoutlmv3():
    """Install a fake ``transformers`` module with stub LayoutLMv3 symbols.

    Save-and-restore the original ``transformers`` so tests that come
    after this fixture can re-import the real package without paying
    the cold-import cost.
    """
    orig = sys.modules.get("transformers")
    fake = types.ModuleType("transformers")
    fake.LayoutLMv3Config = _StubLayoutLMv3Config
    fake.LayoutLMv3Model = _StubLayoutLMv3Model
    sys.modules["transformers"] = fake
    _StubLayoutLMv3Model.last_to_targets = []
    _StubLayoutLMv3Model.instances = []
    yield
    if orig is not None:
        sys.modules["transformers"] = orig
    else:
        sys.modules.pop("transformers", None)


def test_backend_disables_visual_embed_for_path_b(stub_transformers_layoutlmv3):
    """Plan \u00a79 Phase G AMD-portability decision: visual_embed=False, every load."""
    backend = LayoutLMv3Backend(model_id="X", device="cpu")
    backend._ensure_loaded()
    # Check the loaded config instance -- production code does
    # ``self._config.visual_embed = False`` so the instance attribute
    # is the source of truth.
    assert backend._config.visual_embed is False


def test_backend_moves_to_requested_device(stub_transformers_layoutlmv3):
    backend = LayoutLMv3Backend(model_id="X", device="mps")
    backend._ensure_loaded()
    assert _StubLayoutLMv3Model.last_to_targets == ["mps"]


def test_backend_falls_back_to_cpu_on_not_implemented_error(stub_transformers_layoutlmv3):
    """PLAN \u00a79 Phase G: MPS -> CPU on NotImplementedError.

    Approach: monkeypatch ``_StubLayoutLMv3Model.from_pretrained`` to
    install ``_raise_on_to(mps) -> True`` on the freshly-built instance.
    The production ``_ensure_loaded()`` calls this factory, then
    ``.to("mps")`` raises NotImplementedError, which is caught and a
    fallback ``.to("cpu")`` is attempted -- it succeeds and ``self._model``
    binds the model.
    """
    def _raise_factory(cls, model_id, config=None):
        instance = _StubLayoutLMv3Model()
        instance.model_id = model_id
        instance.config = config
        instance._raise_on_to = lambda device: device == "mps"
        _StubLayoutLMv3Model.instances.append(instance)
        return instance

    original_factory = _StubLayoutLMv3Model.from_pretrained
    _StubLayoutLMv3Model.from_pretrained = classmethod(_raise_factory)
    try:
        backend = LayoutLMv3Backend(model_id="X", device="mps")
        backend._ensure_loaded()
        # First ``to("mps")`` raised; the except clause then called
        # ``to("cpu")`` which succeeded. ``last_to_targets`` records
        # only successful calls.
        assert _StubLayoutLMv3Model.last_to_targets == ["cpu"]
    finally:
        _StubLayoutLMv3Model.from_pretrained = original_factory


def test_backend_laziness_no_load_at_construction(stub_transformers_layoutlmv3):
    """No network access / 500 MB download until ``_ensure_loaded``."""
    backend = LayoutLMv3Backend(model_id="X", device="cpu")
    assert backend._model is None
    assert backend._config is None
    assert _StubLayoutLMv3Model.instances == []


def test_backend_model_id_exposed(stub_transformers_layoutlmv3):
    assert LayoutLMv3Backend(model_id="custom/id").model_id == "custom/id"


def test_backend_default_model_id_is_layoutlmv3_base():
    assert LayoutLMv3Backend.DEFAULT_MODEL_ID == "microsoft/layoutlmv3-base"


def test_backend_forward_returns_model_output(stub_transformers_layoutlmv3):
    backend = LayoutLMv3Backend(model_id="X", device="cpu")
    out = backend.forward({"input_ids": "stub", "bbox": "stub"})
    assert out == {"last_hidden_state": "stub-tensor"}
