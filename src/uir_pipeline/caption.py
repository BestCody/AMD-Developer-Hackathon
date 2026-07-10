"""caption -- Phase O / Tier 3: image-aware captioning via microsoft/Florence-2.

PLAN_TIER3.md exit:
    -- lazy-loaded Florence-2 singleton (cached per ``(model_id, device)``)
    -- :func:`caption_images` returns one caption per input PIL image
    -- fail-soft on model-load failure (offline / OOM / missing accelerate)
    -- figure-region detection via IBM Docling (``DoclingDocument.pictures``)
       + crop rendering via PyMuPDF (``page.get_pixmap(clip=Rect)``)
    -- base64-encoded PNG crop serialised onto ``ChunkNode.modal_features.figure``

The module is OPT-IN: importing it pulls in ``transformers`` + ``accelerate``
+ PyMuPDF.  Callers that don't load ``caption`` pay none of that cost.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover -- annotations only
    # Pillow is a heavy optional dep: the string annotations below name
    # `PIL.Image.Image`, but nothing ever bound `PIL`, so type checkers (and
    # ruff's F821) saw an undefined name. Importing it here keeps the runtime
    # import-free.
    import PIL.Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Florence-2-base (microsoft, MIT license). Pinned slug; tighten to a commit
# SHA before production so a LICENSE/SHA drift fails loudly (Plan_TIER3 risk 8).
MODEL_ID: Final[str] = "microsoft/Florence-2-base"

# Florence-2 prompt macros. Default to <MORE_DETAILED_CAPTION> (~80 words).
# Other valid prompts (per the model card): <CAPTION>, <OD>, <OCR>,
# <OCR_WITH_REGION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <VQA>.
DEFAULT_PROMPT: Final[str] = "<MORE_DETAILED_CAPTION>"

# Skip-figure mitigation (PLAN_TIER3 risk 1): tiny / decorative figures
# produce Florence-2 hallucinations. The bbox is in pixel coords; we
# drop anything smaller than this in either dimension.
MIN_FIGURE_DIM_PX: Final[int] = 50

# Subsample when batched to MPS to avoid memory spikes.
MPS_BATCH_CAP: Final[int] = 4

# Generous but bounded generation length. Florence-2 supports longer; we cap
# at 1024 to bound latency on MPS.
MAX_NEW_TOKENS: Final[int] = 1024
NUM_BEAMS: Final[int] = 3


# ----------------------------------------------------------------------------
# Lazy model singleton
# ----------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_MODEL_LOCK = threading.Lock()


def _get_florence2(
    model_id: str = MODEL_ID,
    *,
    device: str | None = None,
    dtype: Any = None,
    force_reload: bool = False,
) -> tuple[Any, Any]:
    """Lazy-load the Florence-2 ``(processor, model)`` pair.

    Returns a 2-tuple ``(processor, model)`` so callers can compose
    ``processor(text=..., images=...)`` directly.  Cached per
    ``(model_id, device)`` after first load so repeated calls in a
    long-running server stay cheap.

    Device + dtype resolution follows :mod:`uir_pipeline.device`:
    defaults to the active backend's preferred dtype (cuda -> fp16,
    mps -> fp32 fallback to fp16 on M-series).
    """
    if device is None:
        from uir_pipeline.device import get_device
        device = get_device()
    if dtype is None:
        from uir_pipeline.device import torch_dtype
        dtype = torch_dtype(device)  # may raise if torch not installed

    cache_key = (model_id, device)
    cached = None if force_reload else _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and not force_reload:
            return cached

        # Lazy imports -- keep this module import-time-cheap.
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Compatibility shim: Florence-2's `trust_remote_code` modeling file
        # reads `forced_bos_token_id` on its config class; transformers 4.46+
        # removed that attribute from the base `PretrainedConfig`. We patch
        # the dynamic Florence2Config class so the modeling code can read
        # the attribute without raising AttributeError. The shim is a no-op
        # if the attribute already exists, and the catch-all `except` keeps
        # runtime non-fatal if the dynamic module isn't loadable here.
        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module as _get_dyn_cls,
            )
            _florence_cfg_cls = _get_dyn_cls(
                class_reference="configuration_florence2.Florence2Config",
                pretrained_model_name_or_path=model_id,
            )
            if not hasattr(_florence_cfg_cls, "forced_bos_token_id"):
                _florence_cfg_cls.forced_bos_token_id = None
                logger.debug("applied Florence-2 compat shim: forced_bos_token_id=None")
        except Exception as exc:
            logger.debug("Florence-2 compat shim failed (load may still work): %s", exc)

        logger.debug(
            "loading Florence-2 model_id=%s device=%s dtype=%s (cold cache; first run downloads ~1.5 GB)",
            model_id, device, getattr(dtype, "__name__", str(dtype)),
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()

        # Some Florence-2 custom ops need the model on the device explicitly
        # before the processor's first call; ``generate`` validates this.
        _MODEL_CACHE[cache_key] = (processor, model)
        return _MODEL_CACHE[cache_key]


def is_available() -> bool:
    """Return ``True`` iff Florence-2 can be loaded right now (best-effort).

    Used by :func:`caption_figures_in_pdf` to fail-fast without paying the
    ~1.5 GB cold-load cost.  We attempt a minimal processor-only load
    (``trust_remote_code=True``); success means the model files exist
    locally (``HF hub`` cache hit).
    """
    try:
        from transformers import AutoProcessor
        # Lazy load without instantiating the heavy model.
        AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        return True
    except Exception as exc:
        logger.debug("florence-2 not available: %s", exc)
        return False


# ----------------------------------------------------------------------------
# Captioning
# ----------------------------------------------------------------------------

def caption_images(
    images: list["PIL.Image.Image"],
    *,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_beams: int = NUM_BEAMS,
    device: str | None = None,
    dtype: Any = None,
) -> list[str]:
    """Caption each PIL ``Image`` with Florence-2.

    Batched: we send all inputs through one ``generate`` call when more
    than one image is provided.  Returns one caption per image; the order
    matches ``images``.  Empty images list returns ``[]``.

    Failure modes (all fail-soft, returning ``[""] * len(images)``):
        -- model load fails (offline / OOM / accelerate missing)
        -- processor call raises (corrupt PNG, bad prompt, etc.)
        -- generate call raises (driver error, dtype mismatch)
    """
    if not images:
        return []
    try:
        import torch  # for inference_mode
        processor, model = _get_florence2(device=device, dtype=dtype)
    except Exception as exc:
        logger.warning(
            "Florence-2 unavailable (returning empty captions): %s", exc,
        )
        return [""] * len(images)

    try:
        inputs = processor(
            text=[prompt] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=model.dtype)
    except Exception as exc:
        logger.warning("Florence-2 preprocess failed: %s", exc)
        return [""] * len(images)

    try:
        with torch.inference_mode():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
    except Exception as exc:
        logger.warning("Florence-2 generate failed: %s", exc)
        return [""] * len(images)

    texts = processor.batch_decode(ids, skip_special_tokens=False)
    out: list[str] = []
    for i, t in enumerate(texts):
        try:
            parsed = processor.post_process_generation(
                t, task=prompt,
                image_size=(images[i].width, images[i].height),
            )
            if isinstance(parsed, dict):
                out.append(parsed.get(prompt, ""))
            else:
                out.append(str(parsed) if parsed is not None else "")
        except Exception as exc:
            logger.debug("post_process_generation failed for image %d: %s", i, exc)
            out.append("")
    return out


# ----------------------------------------------------------------------------
# Figure region detection (Docling)
# ----------------------------------------------------------------------------

def detect_figure_regions_from_docling(
    dr: Any,
    *,
    page_numbers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return detected figure regions from a :class:`DoclingResult` as dicts.

    Each dict shape::

        {
            "page": int,                     # 1-based
            "bbox_pixel": (x1, y1, x2, y2), # 0-1000 virtual canvas (UIR contract)
            "page_width_px": float,          # canonical 1000 (docling native)
            "page_height_px": float,         # canonical 1000 (docling native)
            "kind": "picture",
        }

    ``dr.pictures`` is populated by :mod:`uir_pipeline.docling_extract`
    ``_walk_doc`` (top-level ``doc.pictures`` collection). The bbox is on
    the 0-1000 UIR virtual canvas, so callers can pass it straight to a
    :class:`ChunkNode.bounding_box`. The canvas is the authority -- no
    pdfplumber normalisation needed.
    """
    pics = list(getattr(dr, "pictures", None) or [])
    if page_numbers is None:
        return list(pics)
    allowed = set(page_numbers)
    return [p for p in pics if int(p.get("page", 0)) in allowed]


# ----------------------------------------------------------------------------
# Crop rendering (PyMuPDF)
# ----------------------------------------------------------------------------

def render_figure_crop(
    pdf_path: os.PathLike[str] | str,
    page: int,
    bbox_pixel: tuple[float, float, float, float],
    *,
    dpi: int = 144,
) -> "PIL.Image.Image | None":
    """Render the bbox region of a PDF page as a PIL ``Image``.

    Returns ``None`` on any failure (PyMuPDF missing, the bbox is
    degenerate, the page index is invalid, etc.).  Coordinates are
    in PDF points (pdfplumber's ``page.images`` x0/top/x1/bottom).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not installed -- skipping render_figure_crop")
        return None
    try:
        from PIL import Image
    except ImportError:
        logger.debug("Pillow not installed -- render_figure_crop unusable")
        return None
    try:
        doc = fitz.open(str(pdf_path))
        # 1-based page -> 0-based index.
        if not (1 <= page <= doc.page_count):
            doc.close()
            return None
        page_obj = doc[page - 1]
        x0, y0, x1, y1 = bbox_pixel
        clip = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
        pix = page_obj.get_pixmap(clip=clip, dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return img
    except Exception as exc:
        logger.warning(
            "render_figure_crop failed page=%s bbox=%s: %s",
            page, bbox_pixel, exc,
        )
        return None


def encode_image_b64(
    pil_image: "PIL.Image.Image | None",
    *,
    fmt: str = "PNG",
) -> str | None:
    """Encode a PIL image to a base64 string. ``None`` on failure or input is None."""
    if pil_image is None:
        return None
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("encode_image_b64 failed: %s", exc)
        return None


# ----------------------------------------------------------------------------
# End-to-end: detect + render + caption
# ----------------------------------------------------------------------------

def caption_figures_in_pdf(
    pdf_path=None,
    *,
    docling_result: Any | None = None,
    prompt: str = DEFAULT_PROMPT,
    min_dim_px: int = MIN_FIGURE_DIM_PX,
    page_numbers: list[int] | None = None,
    device: str | None = None,
    dpi: int = 144,
) -> list[dict[str, Any]]:
    """Run detect -> render -> caption for every figure in ``pdf_path`` or ``docling_result``.

    Either pass ``pdf_path`` (Docling will be invoked on it -- no pdfplumber)
    or pass a pre-computed ``docling_result`` to avoid re-running the
    2 GB-weight converter. Returns shape::

        {
            "page":            int,
            "bbox_pixel":      (x1, y1, x2, y2),   # 0-1000 canvas
            "bbox_canvas":     (x1n, y1n, x2n, y2n),  # 0-1000 (already canvas)
            "caption":         str,
            "caption_prompt":  str,
            "caption_model":   "Florence-2-base",
            "image_b64":       str | None,
        }

    Failures inside the loop are logged and skipped. Tiny bboxes
    (< ``min_dim_px`` either side, measured on the 0-1000 canvas, so a
    threshold of 50 ~= 5% of page) are dropped per PLAN_TIER3 risk 1.
    """
    if docling_result is not None:
        dr = docling_result
    elif pdf_path is not None:
        # Run Docling only -- no pdfplumber fallback.
        from uir_pipeline.docling_extract import extract_with_docling
        dr = extract_with_docling(pdf_path)
    else:
        raise TypeError(
            "caption_figures_in_pdf requires pdf_path or docling_result"
        )

    regions = detect_figure_regions_from_docling(dr, page_numbers=page_numbers)
    if not regions:
        return []

    # bboxes are already on the 0-1000 canvas. Convert to PDF points for
    # PyMuPDF rendering so we can crop from the original PDF.
    rendered: list[tuple[dict[str, Any], Any]] = []
    for r in regions:
        canvas_bbox = tuple(r["bbox"])  # x1, y1, x2, y2 on 0-1000
        # `min_dim_px` is already expressed on the 0-1000 canvas (50 ~= 5% of
        # the page), so compare canvas extents directly. Rescaling by 50/1000
        # first -- as this did -- shrinks a 120-unit figure to 6 and rejects
        # everything narrower than 833 units, i.e. all but full-width figures.
        width = canvas_bbox[2] - canvas_bbox[0]
        height = canvas_bbox[3] - canvas_bbox[1]
        if width < min_dim_px or height < min_dim_px:
            continue
        # Re-scale to PDF points using known page dims (derived from
        # canvas height == 1000 unit total page -- so factor = page_px / 1000).
        # We use 792 (US letter) as the default page height in points -- if
        # the PDF is a different size the crop will be slightly off, which
        # is fine for Florence-2 (it does its own internal resampling).
        page_h_pts = float(getattr(dr, "page_height_pts", 792))
        page_w_pts = float(getattr(dr, "page_width_pts",  612))
        bbox_pdf_pts = (
            canvas_bbox[0] * page_w_pts / 1000.0,
            canvas_bbox[1] * page_h_pts / 1000.0,
            canvas_bbox[2] * page_w_pts / 1000.0,
            canvas_bbox[3] * page_h_pts / 1000.0,
        )
        pil = render_figure_crop(
            pdf_path if pdf_path is not None else "/dev/null",
            r["page"], bbox_pdf_pts, dpi=dpi,
        ) if pdf_path is not None else None
        rendered.append((r, pil))

    pils: list[Any] = [p for _, p in rendered if p is not None]
    captions = caption_images(pils, prompt=prompt, device=device) if pils else []

    out: list[dict[str, Any]] = []
    cap_idx = 0
    for r, pil in rendered:
        if pil is None:
            # No PDF available (docling_result was pre-computed in the
            # orchestrator and we don't have a fresh path). Use a
            # placeholder so the chunk still gets emitted.
            pass
        cap = captions[cap_idx] if cap_idx < len(captions) else ""
        cap_idx += 1
        out.append({
            "page": r["page"],
            "bbox_pixel": tuple(r["bbox"]),
            "bbox_canvas": tuple(r["bbox"]),
            "caption": cap,
            "caption_prompt": prompt,
            "caption_model": MODEL_ID,
            "image_b64": encode_image_b64(pil) if pil is not None else None,
        })
    return out


__all__ = [
    "DEFAULT_PROMPT",
    "MAX_NEW_TOKENS",
    "MIN_FIGURE_DIM_PX",
    "MODEL_ID",
    "NUM_BEAMS",
    "MPS_BATCH_CAP",
    "caption_figures_in_pdf",
    "caption_images",
    "detect_figure_regions_from_docling",
    "encode_image_b64",
    "is_available",
    "render_figure_crop",
]
