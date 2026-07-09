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
        help="Input PDF file (or directory of PDFs)",
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
        choices=("docling", "pdfplumber"),
        default=None,
        help=(
            "Per-page text-extraction backend. ``docling`` (default when "
            "``UIR_FAST_PATH`` env var is unset) routes Stages 2-5 through "
            "IBM Docling so chunks come out pre-typed as sections / tables / "
            "figures / math instead of flattened prose. ``pdfplumber`` routes "
            "through pdfplumber + the heuristic LayoutClassifier (faster, no "
            "2 GB HuggingFace weight download). When ``docling`` is selected "
            "but unavailable (missing dep OR HF model load failure), the "
            "orchestrator transparently cascades to ``pdfplumber``."
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

    # Resolve PDF inputs.
    targets: list[Path] = []
    if args.input.is_file():
        targets = [args.input]
    elif args.input.is_dir():
        targets = sorted(args.input.glob("*.pdf"))
    else:
        log.error("input not found: %s", args.input)
        return 1

    if not targets:
        log.error("no PDFs found under %s", args.input)
        return 1

    from uir_pipeline.pipeline import run
    args.output_data.mkdir(parents=True, exist_ok=True)

    rc = 0
    for pdf in targets:
        log.info("processing %s", pdf)
        try:
            result = run(
                pdf,
                output_dir=args.output_data,
                skip_weaviate=args.skip_weaviate,
                dry_run=args.dry_run,
                with_embeddings=not args.no_embeddings,
                include_semantics=args.include_semantics,
                fast_path=args.fast_path,
            )
            log.info(
                "done %s: chunks=%d entities=%d elapsed=%.2fs -> %s + %s",
                pdf.name, result.chunk_count, result.entity_count,
                result.elapsed_seconds,
                result.out_path, getattr(result, "umr_path", None) or "n/a",
            )
        except Exception as exc:
            log.exception("pipeline failed for %s: %s", pdf, exc)
            rc = 1  # validation/ingest failure
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
