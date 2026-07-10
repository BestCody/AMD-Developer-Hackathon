"""An orchestrator stub that a *child process* can import.

``web._pipeline_child`` resolves the orchestrator through
``UIR_PIPELINE_MODULE``. Pointing that at this module lets the isolation
tests exercise the real spawn/queue/crash machinery without importing
Docling, torch, or BGE -- and without relying on monkeypatches, which do
not survive ``spawn()``.

The mode selects the behaviour:

    ok      write a small UIR + UMR and return normally
    raise   raise a Python exception inside the child
    crash   raise a genuine SIGSEGV, killing the process outright

It is read from ``intent`` (per job) and falls back to ``STUB_PIPELINE_MODE``
(read once, in the child). The worker is now long-lived, so an env var set in
the *parent* after the child has spawned never reaches it -- ``intent`` is the
only per-job channel that crosses the boundary on every call.
"""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def run(
    input_path: Any,
    *,
    output_dir: Any,
    fast_path: str | None = None,
    skip_weaviate: bool = True,
    with_embeddings: bool = True,
    on_progress: Any = None,
    intent: str | None = None,
    **_kw: Any,
) -> SimpleNamespace:
    mode = intent if intent in ("ok", "raise", "crash") else None
    if mode is None:
        mode = os.environ.get("STUB_PIPELINE_MODE", "ok")

    if on_progress:
        on_progress("ingest", 5)

    if mode == "crash":
        # A genuine SIGSEGV, not sys.exit(): this is what Docling's
        # std::bad_alloc path does to the process. No Python `except` can
        # intercept it, which is exactly the condition under test.
        # (ctypes.string_at(0) is NOT usable here -- on Windows ctypes
        # installs an SEH handler and turns the access violation into a
        # catchable OSError, which would not test anything.)
        faulthandler.disable()  # keep the traceback dump out of the test log
        faulthandler._sigsegv()

    if mode == "raise":
        raise ValueError("stub pipeline exploded")

    if on_progress:
        on_progress("chunk", 65)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    uir = out / "stub.uir.json"
    uir.write_text(
        json.dumps({
            "uiR_version": "1.0",
            "id": "doc_stub",
            "metadata": {"title": "Stub Doc"},
            "source": {"format": "PDF", "route": "pdf"},
            "structure": {"root": {"type": "root", "children": [
                {"id": "chunk_001", "type": "chunk", "page": 1,
                 "text": "The quick brown fox jumps over the lazy dog."},
            ]}},
        }),
        encoding="utf-8",
    )
    umr = out / "stub.umr.md"
    umr.write_text("# Stub Doc\n\nThe quick brown fox.", encoding="utf-8")

    if on_progress:
        on_progress("embed", 92)

    return SimpleNamespace(
        uir_id="doc_stub",
        out_path=uir,
        umr_path=umr,
        chunk_count=1,
        entity_count=0,
        elapsed_seconds=0.1,
    )
