"""fireworks_vision -- Fireworks AI vision-LLM image analysis layer.

This module provides a client for Fireworks AI's OpenAI-compatible chat
completions API, specialised for image understanding. It converts input
images to PNG (ensuring broad format coverage), base64-encodes them, and
sends them to a Fireworks-hosted vision model.

Environment variables:
    ``FIREWORKS_API_KEY``       (required) Fireworks AI API token.
    ``FIREWORKS_VISION_MODEL``  (optional) Model ID override (default:
                                ``accounts/fireworks/models/minimax-m3``)
    ``FIREWORKS_BASE_URL``      (optional) API base URL override (default:
                                ``https://api.fireworks.ai/inference/v1``)

Model:
    Default model is ``accounts/fireworks/models/minimax-m3`` which is
    a vision-capable model available on Fireworks AI serverless.

Design:
    - Fail-soft: if the API call fails, returns a descriptive error dict
      instead of raising so callers can fall back gracefully.
    - Uses the OpenAI-compatible chat format with ``image_url`` content
      type, matching Fireworks AI's documented API.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Final

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_DEFAULT_VISION_MODEL: Final[str] = "accounts/fireworks/models/minimax-m3"
_DEFAULT_BASE_URL: Final[str] = "https://api.fireworks.ai/inference/v1"
_DEFAULT_MAX_TOKENS: Final[int] = 4096

_DETAILED_DESCRIPTION_PROMPT: Final[str] = (
    "Provide an insanely detailed description of this image. "
    "Describe every visible element including objects, people, text, "
    "colors, spatial layout, textures, lighting, context, and any "
    "notable details. Be exhaustive and precise - leave nothing out."
)

_SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({
    "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp",
    "avif", "heic", "heif",
})


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def _ensure_png(
    image_data: bytes,
    *,
    original_ext: str = "png",
) -> bytes:
    """Convert raw image bytes to PNG if not already PNG.

    Uses Pillow to decode and re-encode as PNG. Raises ``ValueError``
    on unsupported formats.
    """
    if original_ext.lower() == "png":
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return image_data
    try:
        buf = io.BytesIO(image_data)
        pil = Image.open(buf)
        out = io.BytesIO()
        pil.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        raise ValueError(f"unsupported or corrupt image data: {exc}") from exc


def load_image_as_png(path: str | Path) -> bytes:
    """Read an image file and return its PNG-encoded bytes."""
    p = Path(path)
    ext = p.suffix.lstrip(".").lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported image extension '.{ext}'; "
            f"supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )
    raw = p.read_bytes()
    return _ensure_png(raw, original_ext=ext)


def load_image_as_b64_png(path: str | Path) -> str:
    """Load image, convert to PNG, return base64 data URI."""
    png_bytes = load_image_as_png(path)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Fireworks AI vision helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Return the Fireworks API key from ``FIREWORKS_API_KEY`` env var."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "FIREWORKS_API_KEY is not set. "
            "Set it in your .env file or environment."
        )
    return key.strip()


def _get_vision_model() -> str:
    return os.environ.get("FIREWORKS_VISION_MODEL", _DEFAULT_VISION_MODEL).strip()


def _get_base_url() -> str:
    return os.environ.get("FIREWORKS_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------


def describe_image(
    image_path: str | Path,
    *,
    intent: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Analyse an image via Fireworks AI vision LLM.

    Two modes:
        * ``intent=None`` - returns an "insanely detailed description".
        * ``intent="..."`` - answers the user's specific query.

    Returns a dict with keys: ``success``, ``description``, ``model``,
    ``prompt``, ``intent``, ``error``, ``usage``.

    A missing ``FIREWORKS_API_KEY`` comes back as ``success=False``, like any
    other failure. It used to raise ``ValueError`` out of ``_get_api_key``
    while an HTTP 401 returned the error dict -- so ``run_image_pipeline``'s
    single ``if not result["success"]`` branch caught one and not the other.
    Same fail-soft contract as :func:`uir_pipeline.chat.answer`.
    """
    png_b64 = load_image_as_b64_png(image_path)

    if intent and intent.strip():
        text_prompt = (
            f"The user asks: {intent.strip()}\n\n"
            "Analyse the provided image carefully and answer the user's "
            "question based on what you see. Be thorough and precise, "
            "referencing specific visual details from the image."
        )
    else:
        text_prompt = _DETAILED_DESCRIPTION_PROMPT

    resolved_model = model or _get_vision_model()
    try:
        api_key = _get_api_key()
    except ValueError as exc:
        return {
            "success": False,
            "error": str(exc),
            "description": "",
            "model": resolved_model,
            "prompt": text_prompt,
            "intent": intent,
            "usage": {},
        }
    base_url = _get_base_url()

    request_body = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": png_b64},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info(
        "fireworks vision call: model=%s intent=%s path=%s",
        resolved_model,
        intent if intent else "(none - detailed description)",
        Path(image_path).name,
    )

    try:
        import requests as _requests

        response = _requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=request_body,
            timeout=120,
        )
        response.raise_for_status()
        response_data = response.json()

        choices = response_data.get("choices", [])
        if not choices:
            return {
                "success": False,
                "error": "empty choices in Fireworks API response",
                "model": resolved_model,
                "prompt": text_prompt,
                "intent": intent,
                "description": "",
            }

        description = choices[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})

        logger.info(
            "fireworks vision response: model=%s tokens=%s",
            resolved_model,
            usage,
        )

        return {
            "success": True,
            "description": description,
            "model": resolved_model,
            "prompt": text_prompt,
            "intent": intent,
            "usage": usage,
        }

    except _requests.exceptions.HTTPError as exc:
        error_body = exc.response.text if exc.response else str(exc)
        logger.error("fireworks vision HTTP %d: %s", exc.response.status_code if exc.response else "?", error_body[:500])
        return {
            "success": False,
            "error": f"HTTP {exc.response.status_code if exc.response else '?'}: {error_body}",
            "model": resolved_model,
            "prompt": text_prompt,
            "intent": intent,
            "description": "",
        }
    except Exception as exc:
        logger.exception("fireworks vision call failed: %s", exc)
        return {
            "success": False,
            "error": str(exc),
            "model": resolved_model,
            "prompt": text_prompt,
            "intent": intent,
            "description": "",
        }


def describe_images(
    image_paths: list[str | Path],
    *,
    intent: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    model: str | None = None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    """Analyse multiple images, one at a time."""
    results: list[dict[str, Any]] = []
    for ip in image_paths:
        try:
            result = describe_image(
                ip, intent=intent, max_tokens=max_tokens,
                model=model, temperature=temperature,
            )
        except Exception as exc:
            result = {
                "success": False,
                "error": str(exc),
                "model": model or _get_vision_model(),
                "prompt": "",
                "intent": intent,
                "description": "",
            }
        results.append(result)
    return results


__all__ = [
    "_DEFAULT_VISION_MODEL",
    "_DETAILED_DESCRIPTION_PROMPT",
    "describe_image",
    "describe_images",
    "load_image_as_b64_png",
    "load_image_as_png",
]
