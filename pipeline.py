"""pipeline.py -- CLI entrypoint (Phase M).

Usage:
    python pipeline.py path/to/doc.pdf --output-data data/output/
    python pipeline.py data/input/ --output-data data/output/ --skip-weaviate
    python pipeline.py doc.pdf --output-data data/output/ --dry-run
    python pipeline.py doc.pdf --output-data data/output/ --log-level DEBUG

Exit codes:
    0 on success (UIR JSON written, optional Weaviate upsert succeeded)
    1 on validation/ingest failure
    2 on partial failure with retryable errors logged
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure ``src/`` is on sys.path so ``uir_pipeline`` resolves when running
# from a clone / system install.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Phase 1 PDF -> UIR pipeline (MVP CLI).",
    )
    parser.add_argument(
        "input", type=Path,
        help=(
            "Input file (or directory). Single file accepts any supported "
            "format (PDF/DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB/RTF/TXT/MD/"
            "CSV/image/audio/code); a directory is rglobbed for "
            "src/uir_pipeline/format_router::SUPPORTED_EXTENSIONS."
        ),
    )
    parser.add_argument(
        "--output-data", type=Path, default=Path("data/output/"),
        help="Directory to write UIR JSON outputs (default: data/output/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Reserved for Phase 2 async batching; MVP processes serially.",
    )
    parser.add_argument(
        "--skip-weaviate", action="store_true",
        help="Do not upsert chunks into Weaviate.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not write any files or call Weaviate.",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip BGE embedding step (faster; chunks emitted without vectors).",
    )
    parser.add_argument(
        "--include-semantics", action="store_true",
        help=(
            "Emit the verbose ``semantics`` block (entities + relationships + "
            "topics) in the UIR JSON. Default: OFF. The companion ``.umr.md`` "
            "file is always emitted regardless of this flag -- UMR is the "
            "agent-friendly view and never carries semantics. Use this flag "
            "only for corpus-analysis / debugging runs."
        ),
    )
    parser.add_argument(
        "--fast-path",
        choices=("docling",),
        default=None,
        help=(
            "Per-page text-extraction backend. ``docling`` is the only "
            "backend now -- the previous pdfplumber fast path was retired "
            "because it couldn't preserve column structure. Docling failures "
            "propagate as :class:`DoclingUnavailable`."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logger level (default: INFO).",
    )
    parser.add_argument(
        "--log-format", default="json",
        choices=("json", "text"),
        help="Stdout log format (default: json).",
    )
    args = parser.parse_args(argv)

    # Configure logging from CLI flags (env vars are also honored).
    from uir_pipeline.logging_config import configure
    configure(level=args.log_level, fmt=args.log_format)
    log = logging.getLogger("pipeline_cli")

    # Resolve inputs across all supported formats (PLAN §17 §Multi-format).
    # Single-file mode passes through dispatch directly; directory mode
    # rglobs the convertible extensions so a heterogeneous corpus
    # (e.g. ``data/inputs/paper.pdf`` + ``notes.md``) is handled in one
    # invocation. Discovery summary is logged once before the loop so
    # unknown extensions surface as ``skipping`` lines.
    #
    # CONVERTIBLE_EXTENSIONS, not SUPPORTED_EXTENSIONS: the latter also names
    # the legacy binary Office formats (.doc/.ppt/.xls), which the router
    # recognises but classifies SKIP. Sweeping them in queues work that can
    # only fail.
    from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS, route as _route
    targets: list[Path] = []
    if args.input.is_file():
        targets = [args.input]
    elif args.input.is_dir():
        seen: set[Path] = set()
        for ext in sorted(CONVERTIBLE_EXTENSIONS):
            for p in args.input.rglob(f"*{ext}"):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    targets.append(p)
        targets.sort()
    else:
        log.error("input not found: %s", args.input)
        return 1

    # PLAN §17 §Multi-format: discovery summary uses a Counter over the
    # detected format / route so users can see "5 PDF, 3 DOCX, 2 unknown
    # (skipping)" at a glance. Counts are accumulated BEFORE the
    # dispatch loop so unknown-extension files are logged but never
    # raised.
    from collections import Counter
    fmt_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for t in targets:
        fmt, route = _route(t)
        fmt_counts[fmt or "UNKNOWN"] += 1
        route_counts[route.value] += 1
    if targets:
        log.info(
            "discovered %d files: formats=%s routes=%s",
            len(targets),
            dict(fmt_counts),
            dict(route_counts),
        )
    if not targets:
        log.error("no supported files found under %s", args.input)
        return 1

    from uir_pipeline.pipeline import run
    args.output_data.mkdir(parents=True, exist_ok=True)

    rc = 0
    skipped = 0
    from uir_pipeline.format_router import FormatRoute
    for input_path in targets:
        # Skip early on unsupported formats (PLAN §17 §Multi-format).
        # ``format_router`` raised ``FormatRoute.SKIP`` for anything
        # ``route()`` couldn't classify. The Counter loop above already
        # had visibility into the underlying format string; we re-check
        # here at dispatch time so order doesn't matter between file and
        # batch invocation modes.
        _, dispatch_route = _route(input_path)
        if dispatch_route is FormatRoute.SKIP:
            log.warning("skipping %s: unsupported format", input_path.name)
            skipped += 1
            continue
        log.info("processing %s (route=%s)", input_path, dispatch_route.value)
        try:
            result = run(
                input_path,
                output_dir=args.output_data,
                skip_weaviate=args.skip_weaviate,
                dry_run=args.dry_run,
                with_embeddings=not args.no_embeddings,
                include_semantics=args.include_semantics,
                fast_path=args.fast_path,
            )
            log.info(
                "done %s: chunks=%d entities=%d elapsed=%.2fs -> %s + %s",
                input_path.name, result.chunk_count, result.entity_count,
                result.elapsed_seconds,
                result.out_path, getattr(result, "umr_path", None) or "n/a",
            )
        except Exception as exc:
            log.exception("pipeline failed for %s: %s", input_path, exc)
            rc = 1  # validation/ingest failure
    if skipped:
        log.warning("skipped %d unsupported files (see warnings above)", skipped)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
