"""ingest_rtf -- RTF format ingress (PLAN \u00a717 \u00a7Multi-format).

RTF (``.rtf``) is a pageless text format handled by the TextLane;
``docling`` don't accept. We:

1. Read the whole file as bytes.
2. Decode (latin-1, replace errors) so striprtf's regex layer sees
   a string. RTF is officially 7-bit ASCII but real-world files
   often carry Windows-1252 / mixed encoding; latin-1 is the
   safe lossless decode for both.
3. Run :func:`striprtf.striprtf.rtf_to_text` to strip control words.
4. Pipe the resulting plain text through :func:`chunk.paginate_pageless`
   so position semantics survive into the chunker.

Returns the same ``(DocumentInput, page_text_pairs)`` tuple shape
that ``ingest.ingest`` yields for PDF, so the orchestrator can
dispatch uniformly.

Failure modes (loud, not silent):
    -- missing dep (`striprtf` not installed) -> raises
       :class:`StriprtfUnavailable` so the orchestrator can decide
       between SKIP-fail-soft and propagate.
    -- decode failure -> falls back to strict UTF-8 with errors=replace.
    -- striprtf raises on malformed input -> wrapped to ``ValueError``
       with the source path embedded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uir_pipeline.ingest import DocumentInput, compute_sha256


# Encoding fallback order. RTF is officially 7-bit ASCII; real-world
# Windows-authored RTF uses Windows-1252 / cp1252. latin-1 is the
# safe lossless decode (every byte 0..255 is a valid latin-1 code
# point), so we never raise on decode -- the worst case is mojibake.
_RTF_DECODE_CANDIDATES: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")


class StriprtfUnavailable(RuntimeError):
    """Raised when the ``striprtf`` package isn't installed.

    Mirrors :class:`src.uir_pipeline.docling_extract.DoclingUnavailable`
    so future orchestrator fail-soft logic can recognise "missing
    conversion dep" consistently across formats.
    """


def _import_striprtf() -> Any:
    """Lazy-import :func:`striprtf.striprtf.rtf_to_text`.

    Note: the public function lives one level deep
    (``striprtf.striprtf.rtf_to_text``) -- striprtf 0.0.32 doesn't
    re-export it at the top level. We import the nested module
    and return the function. Raises :class:`StriprtfUnavailable`
    so callers can fall back without crashing.
    """
    try:
        from striprtf.striprtf import rtf_to_text
    except Exception as exc:  # noqa: BLE001 -- import-time errors are diverse
        raise StriprtfUnavailable(
            f"striprtf package not importable: {type(exc).__name__}: {exc}"
        ) from exc
    return rtf_to_text


def _decode_rtf_bytes(raw: bytes) -> str:
    """Decode RTF bytes trying utf-8 first, then cp1252, then latin-1.

    Latin-1 is the universal fallback: every byte 0..255 maps to a
    valid code point so it never raises. Worst case the caller sees
    replacement glyphs downstream; that's a known RTF tradeoff when
    the source carries non-ASCII Windows characters.
    """
    for enc in _RTF_DECODE_CANDIDATES:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def ingest_rtf(path: str | Path) -> tuple[DocumentInput, list[tuple[int, str]]]:
    """Read ``path`` as RTF, return ``(DocumentInput, page_text_pairs)``.

    ``page_text_pairs`` is ``[(page_no, joined_text), ...]`` with each
    page bounded at ~2000 BPE tokens (see
    :func:`uir_pipeline.chunk.paginate_pageless`). The default
    :attr:`DocumentInput.page_count` is the number of synthesized
    pages -- ``1`` for trivially short RTF, more for long files.

    Raises:
        FileNotFoundError: ``path`` is not a regular file.
        :class:`StriprtfUnavailable`: ``striprtf`` is not importable.
        ValueError: striprtf raises on malformed RTF input.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not a regular file")

    rtf_to_text = _import_striprtf()
    raw = p.read_bytes()
    decoded = _decode_rtf_bytes(raw)
    try:
        text = rtf_to_text(decoded)
    except Exception as exc:  # noqa: BLE001 -- striprtf's errors are heterogeneous
        raise ValueError(
            f"striprtf.rtf_to_text failed on {p.name}: {type(exc).__name__}: {exc}"
        ) from exc

    # Lazy import to avoid pulling chunk.py's deps into a lightweight
    # RTF-only import path.
    from uir_pipeline.chunk import paginate_pageless
    pages = paginate_pageless(text)
    page_count = max(1, len(pages))

    doc = DocumentInput(
        source_path=p,
        uri=p.resolve().as_uri(),
        mime_type="application/rtf",
        size_bytes=p.stat().st_size,
        sha256=compute_sha256(p),
        timestamp=datetime.now(timezone.utc),
        title=p.stem or None,
        author=None,
        created=None,
        modified=None,
        page_count=page_count,
        # New §17 §Multi-format fields. ``DocumentInput.format`` was added
        # alongside the schema widening; ``route`` is recorded so
        # provenance reads cleanly from the UIR JSON.
        format="RTF",
        route="text",
    )
    return doc, pages


def striprtf_environment_enabled() -> bool:
    """Return ``True`` iff the ``striprtf`` package is importable.

    Same shape as :func:`src.uir_pipeline.docling_extract
    .docling_environment_enabled` so the CLI can status-check both
    deps with one helper.
    """
    try:
        _import_striprtf()
        return True
    except StriprtfUnavailable:
        return False


__all__ = [
    "StriprtfUnavailable",
    "ingest_rtf",
    "striprtf_environment_enabled",
]
