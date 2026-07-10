"""image_pipeline -- stand-alone image to UIR + UMR pipeline using Fireworks AI vision.

This module is the high-level orchestrator for processing a single image
file through the Fireworks AI vision LLM and producing the standard UIR
(Universal Intermediate Representation) and UMR (Universal Markdown
Representation) outputs.

Flow:
    1. Detect format - ensure it's an image, reject non-image.
    2. Convert to PNG - normalise any input format to PNG.
    3. Call Fireworks AI vision - either detailed description (no intent)
       or answer query (with intent).
    4. Build UIRV1 - assemble the vision result into the standard UIR
       schema with ``modal_type=\"image\"``.
    5. Build UMR - render the agent-friendly markdown view.
    6. Write outputs - ``{doc_id}.uir.json`` + ``{doc_id}.umr.md``.

All image formats (PNG, JPG, JPEG, GIF, BMP, TIFF, WebP) are supported
and transparently converted to PNG before the API call.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uir_pipeline import fireworks_vision as _fv
from uir_pipeline.utils import deterministic_node_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ImagePipelineResult:
    """Shape returned by :func:`run_image_pipeline`."""

    uir_id: str
    out_path: Path
    umr_path: Path
    description_length: int
    model_used: str
    elapsed_seconds: float
    error: str | None = None


# ---------------------------------------------------------------------------
# UIR builder
# ---------------------------------------------------------------------------


def _build_uir(
    doc_id: str,
    image_path: Path,
    description: str,
    model: str,
    intent: str | None,
    prompt: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble a UIRV1-compatible dict for an image analysis result."""
    now = datetime.now(timezone.utc)
    source_name = image_path.name
    source_uri = image_path.resolve().as_uri()

    return {
        "uiR_version": "1.0",
        "id": doc_id,
        "modal_type": "image",
        "source": {
            "uri": source_uri,
            "filename": source_name,
            "format": image_path.suffix.lstrip(".").upper() or "PNG",
        },
        "metadata": {
            "title": f"Image analysis: {source_name}",
            "author": None,
            "date": now.isoformat(),
            "language": "en",
            "page_count": 1,
            "chunk_count": 1,
            "total_tokens": usage.get("total_tokens", 0) if usage else 0,
        },
        "structure": {
            "root": {
                "id": deterministic_node_id("doc", doc_id),
                "type": "document",
                "title": f"Image: {source_name}",
                "children": [
                    {
                        "id": deterministic_node_id("figure", doc_id, "vision"),
                        "type": "figure",
                        "title": (
                            f"Vision analysis ({model})"
                            if intent
                            else "Detailed image description"
                        ),
                        "children": [
                            {
                                "id": deterministic_node_id("chunk", doc_id, "vision"),
                                "type": "chunk",
                                "text": description,
                                "page": 1,
                                "bounding_box": [0, 0, 1000, 1000],
                                "token_count": len(description.split()),
                                "confidence": 0.95,
                                "region_kind": "figure",
                                "modal_features": {
                                    "image": {
                                        "intent": intent,
                                        "model": model,
                                        "prompt": prompt,
                                        "usage": usage if usage else {},
                                    },
                                },
                                "section_path": None,
                                "is_section_first": True,
                                "is_section_last": True,
                                "preceding_chunk_id": None,
                                "following_chunk_id": None,
                            },
                        ],
                    },
                ],
                "intent_filter": None,
            },
        },
        "semantics": {
            "entities": [],
            "relationships": [],
            "topics": [],
        },
        "provenance": {
            "extraction": {
                "model": model,
                "version": "1.0",
                "timestamp": now.isoformat(),
            },
            "normalization": {
                "version": "1.0",
                "timestamp": now.isoformat(),
            },
        },
    }


# ---------------------------------------------------------------------------
# UMR builder
# ---------------------------------------------------------------------------


def _build_umr(uir_dict: dict[str, Any]) -> str:
    """Render an image-analysis UIR dict into a clean Markdown string."""
    src = uir_dict.get("source", {})
    meta = uir_dict.get("metadata", {})
    root = uir_dict.get("structure", {}).get("root", {})
    prov = uir_dict.get("provenance", {}).get("extraction", {})

    lines: list[str] = []
    lines.append(f"# Image: {src.get('filename', 'unknown')}")
    lines.append("")

    fmt = src.get("format", "?")
    model = prov.get("model", "?")
    ts = meta.get("date", "?")
    lines.append(
        f"> **Format:** {fmt} * **Vision model:** {model} * **Analysed:** {ts}"
    )
    lines.append("")

    for fig in root.get("children", []):
        if fig.get("type") == "figure":
            fig_title = fig.get("title", "")
            lines.append(f"## {fig_title}")
            lines.append("")
            for chunk in fig.get("children", []):
                if chunk.get("type") == "chunk":
                    text = chunk.get("text", "")
                    modal = chunk.get("modal_features", {}).get("image", {})
                    intent = modal.get("intent")
                    usage = modal.get("usage", {})
                    if intent:
                        lines.append(f"_Intent:_ `{intent}`")
                        lines.append("")
                    lines.append(text)
                    lines.append("")
                    if usage:
                        lines.append("---")
                        lines.append("")
                        pt = usage.get("prompt_tokens", 0)
                        ct = usage.get("completion_tokens", 0)
                        tt = usage.get("total_tokens", 0)
                        lines.append(
                            f"> _Token usage: {pt} prompt + {ct} completion = {tt} total_"
                        )
                        lines.append("")

    if not lines:
        lines.append("_No content extracted from this image._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_image_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    intent: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    on_progress: Any | None = None,
) -> ImagePipelineResult:
    """Process a single image through the Fireworks AI vision pipeline.

    Args:
        input_path: Path to the image file (PNG, JPG, JPEG, GIF, BMP,
                    TIFF, WebP).
        output_dir: Directory to write ``{doc_id}.uir.json`` and
                    ``{doc_id}.umr.md``.
        intent: Optional intent/query about the image. When provided, the
                vision model answers this specific question rather than
                generating a general description.
        model: Vision model ID override.
        dry_run: If True, don't write output files (simulate only).
        on_progress: Optional callback ``fn(stage: str, percent: int)``.

    Returns:
        An :class:`ImagePipelineResult` with paths and metadata.
    """
    t0 = time.monotonic()
    p = Path(input_path)
    out_dir = Path(output_dir)

    def _progress(stage: str, pct: int, **meta: Any) -> None:
        logger.info("image_pipeline.stage %s (%d%%) meta=%s", stage, pct, meta)
        if on_progress is not None:
            try:
                on_progress(stage, pct, **meta)
            except Exception:
                pass

    # Derive deterministic doc ID from the image file URI
    from uir_pipeline.embed import derive_doc_id

    doc_id = derive_doc_id(p.resolve().as_uri())

    # Stage 1: validate and convert to PNG
    _progress("convert_png", 10)
    try:
        png_bytes = _fv.load_image_as_png(p)
        logger.info("converted %s to PNG (%d bytes)", p.name, len(png_bytes))
    except ValueError as exc:
        logger.error("unsupported image format: %s", exc)
        return ImagePipelineResult(
            uir_id="",
            out_path=out_dir / "ERROR",
            umr_path=out_dir / "ERROR",
            description_length=0,
            model_used=model or "?",
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )

    # Stage 2: call Fireworks AI vision
    _progress("fireworks_vision", 30)
    logger.info(
        "calling Fireworks AI vision (intent=%s)", intent if intent else "none"
    )
    result = _fv.describe_image(p, intent=intent, model=model)

    if not result.get("success"):
        error_msg = result.get("error", "unknown error")
        logger.error("Fireworks vision call failed: %s", error_msg)
        return ImagePipelineResult(
            uir_id=doc_id,
            out_path=out_dir / f"{doc_id}.uir.json",
            umr_path=out_dir / f"{doc_id}.umr.md",
            description_length=0,
            model_used=result.get("model", model or "?"),
            elapsed_seconds=time.monotonic() - t0,
            error=error_msg,
        )

    description = result.get("description", "")
    resolved_model = result.get("model", model or _fv._DEFAULT_VISION_MODEL)
    usage = result.get("usage")
    prompt = result.get("prompt", "")

    logger.info(
        "vision response: model=%s length=%d tokens",
        resolved_model,
        len(description),
    )

    # Stage 3: build UIR
    _progress("assemble_uir", 70)
    uir_dict = _build_uir(
        doc_id=doc_id,
        image_path=p,
        description=description,
        model=resolved_model,
        intent=intent,
        prompt=prompt,
        usage=usage,
    )

    # Stage 4: build UMR
    _progress("assemble_umr", 85)
    umr_text = _build_umr(uir_dict)

    # Stage 5: write outputs
    out_path = out_dir / f"{doc_id}.uir.json"
    umr_path = out_dir / f"{doc_id}.umr.md"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(uir_dict, indent=2), encoding="utf-8")
        umr_path.write_text(umr_text, encoding="utf-8")
        logger.info("wrote %s and %s", out_path.name, umr_path.name)
    else:
        logger.info("dry-run: would write %s and %s", out_path.name, umr_path.name)

    _progress("done", 100)
    elapsed = time.monotonic() - t0

    return ImagePipelineResult(
        uir_id=doc_id,
        out_path=out_path,
        umr_path=umr_path,
        description_length=len(description),
        model_used=resolved_model,
        elapsed_seconds=round(elapsed, 3),
    )


__all__ = [
    "ImagePipelineResult",
    "run_image_pipeline",
]
