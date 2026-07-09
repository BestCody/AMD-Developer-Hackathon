"""scripts/check_caption_env.py -- Tier 3 captioning environment diagnostic.

Standalone (no ``pytest`` cost) validator for the Florence-2 image-aware
caption path. Prints a 1-screen report covering:

    -- Python version + torch backend availability
    -- uir_pipeline.device picks (device, dtype) for the active host
    -- PIL / PyMuPDF / transformers / docling imports
    -- Florence-2 weight download status (cache hit / miss)
    -- Optional: cold-load + forward-pass + synthetic ``caption_images``
       round-trip wall-clock

Exit codes (consumed by CI/Azure ML pre-flight):
    0 -- env OK (Florence-2 loadable; pipeline will run captions by default)
    1 -- degraded (weights/cache missing; will fail-soft in production)
    2 -- broken (torch missing; pipeline cannot run caption stage)

Usage::

    .venv/bin/python scripts/check_caption_env.py
    .venv/bin/python scripts/check_caption_env.py --full   # + runtime round-trip
    .venv/bin/python scripts/check_caption_env.py --json   # machine-readable

Per PLAN_TIER3.md step 5, this is the script CI runs on the AMD MI300X
runner before paying the pytest collection cost. The diagnostics also
double as the README's "Apple Silicon vs ROCm" matrix table source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Make src/ importable when running as a script (mirrors
# scripts/generate_fixtures.py). Lets users invoke
# ``python scripts/check_caption_env.py`` from the project root OR
# ``python scripts/check_caption_env.py --full`` from any cwd without
# manually exporting PYTHONPATH. If the package is already on sys.path
# (e.g. CI venv with editable install) we no-op so we never duplicate.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _print_section(title: str) -> None:
    """Emit a 1-line banner header."""
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


# NOTE: a generic ``_safe_call(fn, *args) -> (value, error_str)`` wrapper
# was considered to factor the inline ``try/except`` blocks in the report
# helpers below. After review it was cut: the report functions have
# *heterogeneous* fallback shapes (some return dict keys with a fixed
# structure, others populate an ``error`` field on miss) so a single
# (value, error_str) tuple doesn't fit. Inline try/except is clearer than
# a lossy shim. Don't reintroduce it.


def _report_environment() -> dict[str, object]:
    """Snapshot the runtime environment."""
    env: dict[str, object] = {
        "python": sys.version.split()[0],
    }
    try:
        import torch  # noqa: WPS433
        env["torch"] = torch.__version__
        env["mps_available"] = bool(torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_hip_version"] = getattr(torch.version, "hip", None)
    except ImportError as exc:
        env["torch"] = f"MISSING: {exc}"
    try:
        from uir_pipeline.device import get_device, torch_dtype
        env["uir_device"] = get_device()
        try:
            env["uir_dtype"] = str(torch_dtype(env["uir_device"]).__name__)
        except Exception as exc:  # pragma: no cover
            env["uir_dtype"] = f"unresolved: {exc}"
    except Exception as exc:  # pragma: no cover
        env["uir_device"] = f"unresolved: {exc}"
    for name in ("PIL", "fitz", "transformers", "accelerate"):
        try:
            __import__(name)
            env[f"dep_{name}"] = "ok"
        except ImportError as exc:
            env[f"dep_{name}"] = f"MISSING: {exc}"
    return env


def _report_florence2_weights() -> dict[str, object]:
    """Probe whether Florence-2 weights are loadable without paying full cost.

    ``caption.is_available`` does a tentative ``AutoProcessor.from_pretrained``
    only -- it does not load the heavy ``~1.5 GB`` causal-LM. If the weights
    are NOT cached, the probe fails fast and we report a degraded env.
    """
    info: dict[str, object] = {"attempted": True}
    try:
        from uir_pipeline.caption import is_available, MODEL_ID
        info["model_id"] = MODEL_ID
        t0 = time.monotonic()
        info["available"] = bool(is_available())
        info["probe_seconds"] = round(time.monotonic() - t0, 3)
    except Exception as exc:
        info["available"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _run_full_round_trip() -> dict[str, object]:
    """Best-effort ``caption_images`` round-trip on a 128x128 synthetic PIL.

    Returns ``{"ok": True, "elapsed_seconds": ..., "captions": [...]}`` on
    success; ``{"ok": False, "error": "..."}`` otherwise. Intentionally
    swallows all exceptions so the diagnostic stays report-only even when
    the local env is broken.
    """
    out: dict[str, object] = {"attempted": True}
    try:
        from PIL import Image
        pil = Image.new("RGB", (128, 128), (60, 110, 200))
        from uir_pipeline.caption import caption_images
        t0 = time.monotonic()
        captions = caption_images([pil])
        out["ok"] = True
        out["captions"] = captions
        out["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full", action="store_true",
        help="also do a cold-load + Florence-2 forward pass (downloads if needed)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON only (no banner lines)",
    )
    args = parser.parse_args(argv)

    env = _report_environment()
    florence = _report_florence2_weights()
    runtime = _run_full_round_trip() if args.full else {"attempted": False}

    if args.json:
        print(json.dumps({"env": env, "florence": florence, "runtime": runtime}, indent=2, default=str))
    else:
        _print_section("Environment")
        for k, v in sorted(env.items()):
            print(f"  {k:<20s} = {v}")
        _print_section("Florence-2 weights")
        for k, v in sorted(florence.items()):
            print(f"  {k:<20s} = {v}")
        if args.full:
            _print_section("Runtime round-trip")
            for k, v in sorted(runtime.items()):
                print(f"  {k:<20s} = {v}")

    # Exit code policy from the module docstring. Order: most-broken
    # first so a single broken dependency never masquerades as a
    # degraded env (``1``) when it's actually a hard-no-go (``2``).
    torch_status = str(env.get("torch", ""))
    if "MISSING" in torch_status:
        # torch missing -> caption stage cannot run; pipeline CAN emit
        # text-only UIRs but Tier 3 can't ship. Treat as broken.
        return 2
    if not florence.get("available", False):
        # torch works but Florence-2 weights not cached / load failed;
        # pipeline will surface figures WITHOUT captions (fail-soft).
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
