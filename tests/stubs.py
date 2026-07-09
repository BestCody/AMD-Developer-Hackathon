"""tests/stubs.py -- shared test-stub helpers (Florence-2 + PIL).

Promoted out of :mod:`tests.test_caption` after the
``tests/integration/test_pipeline_tier3.py`` cross-test-directory import
was flagged as a code smell. The stubs here preserve the exact
contracts the production code in :mod:`uir_pipeline.caption` consumes:

    -- :class:`_StubProcessor` mimics ``transformers.AutoProcessor``:
       supports ``(text=..., images=..., return_tensors=...)``,
       ``batch_decode``, ``post_process_generation``.
    -- :class:`_StubModel` mimics ``AutoModelForCausalLM``: supports
       ``.generate(input_ids=..., pixel_values=..., max_new_tokens=...,
       num_beams=...)`` and exposes ``.device`` / ``.dtype`` so
       :func:`uir_pipeline.caption.caption_images` doesn't blow up on
       ``inputs.to(model.device, dtype=model.dtype)``.
    -- :class:`_StubInputs` mimics the BatchEncoding returned by
       ``processor(...)``; supports subscripting (``inputs["input_ids"]``)
       AND ``.to(*args, **kwargs)`` (no-op in stub mode).
    -- :func:`_make_pil_stub` builds a tiny solid-color PIL image without
       pulling in heavy fixtures.

Anything requiring actual tensor arithmetic (e.g. ``tensor.shape``,
``tensor.tolist()``) will FAIL loudly under stub mode -- that is the
intended contract so future tests that lift behaviour out of the stub
are forced to mock the tensor explicitly OR skip.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Processor stub
# ---------------------------------------------------------------------------

class _StubProcessor:
    """Mimics :class:`transformers.AutoProcessor` just enough for the round-trip test.

    ``__call__`` returns a :class:`_StubInputs` -- the stub model never
    reads from tensor values, only from the processor's ``batch_decode``
    + ``post_process_generation`` outputs.
    """

    def __init__(self, canned: str | list[str]) -> None:
        self._canned = canned if isinstance(canned, list) else [canned]

    def __call__(self, *, text: Any, images: Any, return_tensors: Any, padding: bool = True) -> "_StubInputs":
        # The stub never calls .to() on real tensors, but ``inputs.to(...)``
        # is reasonable to support. Implement it as a no-op dict.
        return _StubInputs(text=text, images=images)

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        n = len(self._canned)
        return [self._canned[i % n] for i in range(len(ids) if hasattr(ids, "__len__") else 1)]

    def post_process_generation(self, text: str, *, task: str, image_size: Any) -> dict[str, str]:
        return {task: text}


class _StubModel:
    """Mimics :class:`transformers.AutoModelForCausalLM`.

    ``generate(...)`` ignores its inputs and returns a sentinel shape
    the caller never inspects -- the stub processor's ``batch_decode``
    feeds canned captions back regardless.
    """

    def __init__(self, canned: str | list[str]) -> None:
        self._canned = canned if isinstance(canned, list) else [canned]
        self.device = "cpu"
        self.dtype = None

    def generate(
        self,
        *,
        input_ids: Any,
        pixel_values: Any,
        max_new_tokens: int,
        num_beams: int,
    ) -> list:
        return [None] * (len(pixel_values) if hasattr(pixel_values, "__len__") else 1)


class _StubInputs:
    """Fakes the HF tokenized-output dict; supports ``.to(...)`` + dict subscripting.

    The real Florence-2 processor returns an object that supports both
    attribute access (``inputs.input_ids``) AND key access
    (``inputs['input_ids']``). Our stub wires up the latter so that
    :func:`uir_pipeline.caption.caption_images`'s
    ``inputs["input_ids"]`` / ``inputs["pixel_values"]`` lookups succeed.

    Stub values are plain lists (not torch tensors) shaped ``len(images)``.
    Real Florence-2 wraps them as ``torch.long`` (input_ids) and
    ``torch.float32`` (pixel_values); the stub is good enough for the
    single ``generate()`` call the production code makes, but anything
    that does ``tensor.shape`` / ``tensor.dtype`` / ``.tolist()`` would
    fail loudly in stub mode (intentional -- so a future test that
    reaches into tensor shape is forced to mock OR skip).
    """

    def __init__(self, text: Any, images: Any) -> None:
        self.text = text
        self.images = images
        n = len(images)
        self._lookup: dict[str, list] = {
            "input_ids": [0] * n,
            "pixel_values": list(images),
            "attention_mask": [1] * n,
        }
        # Invariant: every key has len == len(images) so the stub
        # never accidentally broadcasts a batch dimension mismatched
        # with what generate() will iterate over.
        for k, v in self._lookup.items():
            assert len(v) == n, f"_StubInputs[{k!r}] not batch-broadcast (len={len(v)} != {n})"

    def __getitem__(self, key: str) -> list:
        return self._lookup[key]

    def to(self, *args: Any, **kwargs: Any) -> "_StubInputs":
        # Real Florence-2 moves input_ids/pixel_values to the target
        # device + casts to dtype. Stub-mode tests don't care -- the
        # parent (Fake) model never reads the values.
        return self


# ---------------------------------------------------------------------------
# PIL stub
# ---------------------------------------------------------------------------

def _make_pil_stub(w: int = 32, h: int = 32) -> Any:
    """Build a tiny synthetic PIL ``Image`` (no PyMuPDF / Florence weights needed).

    Skips the test gracefully when Pillow is missing instead of
    failing-import. Used by Florence-2 caption round-trip tests.
    """
    import pytest
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow is required for caption tests")
    return Image.new("RGB", (w, h), (127, 127, 127))


__all__ = [
    "_StubInputs",
    "_StubModel",
    "_StubProcessor",
    "_make_pil_stub",
]
