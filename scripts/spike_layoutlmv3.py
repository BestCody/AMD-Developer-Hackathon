"""spike_layoutlmv3.py -- Phase A.5 LayoutLMv3 de-risking spike (v3.1).

After the v2 kwarg hunt failed (`transformers` 5.13 has no
`apply_visual` / `visual_feats` / `use_visual` on `from_pretrained`),
the investigation showed the canonical control is the `visual_embed`
config field (`LayoutLMv3Config.visual_embed=True` by default). This
spike tests **both** loading paths on MPS+CPU:

  Path A (multimodal):     default config, `visual_embed=True`
  Path B (text+bbox-only): config patched with `visual_embed=False`

Either path producing an MPS forward pass counts as a Phase A.5 PASS.

Usage (after `pip install -r requirements.txt`):

    source .venv/bin/activate
    python scripts/spike_layoutlmv3.py

Exit codes:
  0  PASS -- Path B (text+bbox-only) runs inference on MPS.
  1  WARN -- Path B fails on MPS; Path A or CPU is the fallback.
  2  FAIL -- LayoutLMv3 fails on every path/device; plan pivot.

Path B is the MVP default per PLAN.md \u00a79 / \u00a7A.5: smaller model, faster
per-page inference. Path A is informational -- if it's the only path
that works, the spike exits 1 to surface the deviation rather than
silently shipping multimodal forward-ops.
"""
from __future__ import annotations

import argparse
import platform
import sys
from typing import Any

import torch


# Pivot chain (from PLAN.md \u00a79 Phase A.5). Used only when both Path A and
# Path B fail on every device -- a real-world outcome that has not yet
# been observed empirically in transformers 5.x as of mid-2026.
PIVOT_HINTS = """
Pivot chain (from PLAN.md \u00a79 Phase A.5; try in order):
  1. Update `transformers` / `torch` to a newer drop that ships an MPS
     fix for LayoutLMv3 ops (cheapest retry).
  2. Swap to LayoutParser + PubLayNet (Detectron2 layout-only detector) for
     section classification; still CPU-friendly, no multimodal branching.
  3. Fall back to pure-PDF heuristic (font-size + whitespace) -- weakest
     accuracy, always works because no ML involved.
""".strip()


def _load_config(model_id: str):
    """Load model config; returns (config_obj, error_str_or_None)."""
    try:
        from transformers import LayoutLMv3Config
        return LayoutLMv3Config.from_pretrained(model_id), None
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def try_path(model_id: str, visual_embed: bool) -> tuple[bool, str, Any]:
    """One (config_path, device) trial. Returns (ok, message, model_or_None)."""
    label = "multimodal (visual_embed=True)" if visual_embed else "text+bbox-only (visual_embed=False)"
    # Fresh config per path -- avoid cross-contamination between A and B.
    cfg, err = _load_config(model_id)
    if cfg is None:
        return (False, f"FAIL [{label}]: config load: {err}", None)
    try:
        cfg.visual_embed = visual_embed
        from transformers import LayoutLMv3Model
        model = LayoutLMv3Model.from_pretrained(model_id, config=cfg)
    except (NotImplementedError, RuntimeError, TypeError, OSError, ImportError, AttributeError, ValueError) as e:
        return (False, f"FAIL [{label}]: {type(e).__name__}: {e}", None)

    # CPU smoke first.
    try:
        model.eval()
        model.to("cpu")
    except (NotImplementedError, RuntimeError, TypeError) as e:
        return (False, f"FAIL [{label}]: model.to(cpu) {type(e).__name__}: {e}", None)

    try:
        ids = torch.tensor([[101, 2070, 102]], dtype=torch.long)
        mask = torch.ones_like(ids)
        bbox = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 100, 30], [0, 0, 0, 0]]],
            dtype=torch.long,
        )
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, bbox=bbox)
        shape = tuple(out.last_hidden_state.shape)
    except (NotImplementedError, RuntimeError, TypeError, ValueError) as e:
        return (False, f"FAIL [{label}]: CPU forward pass: {type(e).__name__}: {e}", None)

    msg = f"OK on CPU (load_path={label}, last_hidden_state={shape})"
    return (True, msg, model)


def try_inference_on_device(
    model: Any, device: str, visual_embed: bool = False,
) -> tuple[bool, str]:
    """Try moving + forward-passing the model on the given device.

    For visual_embed=True (Path A), a synthetic pixel_values tensor
    (1, 3, 224, 224) is supplied so the multimodal visual branch is
    actually exercised during the forward pass. For visual_embed=False
    (Path B), no pixel_values is sent so the text+bbox branch is tested.
    """
    try:
        model.to(device)
    except (NotImplementedError, RuntimeError, TypeError) as e:
        return (False, f"FAIL: model.to('{device}') {type(e).__name__}: {e}")

    try:
        ids = torch.tensor([[101, 2070, 102]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        bbox = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 100, 30], [0, 0, 0, 0]]],
            dtype=torch.long,
            device=device,
        )
        forward_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": mask,
            "bbox": bbox,
        }
        if visual_embed:
            # Synthetic image: 1 batch, 3 channels, input_size x input_size.
            # Reading from model.config.input_size keeps the spike robust
            # to non-default input_size in future fixtures / variant models.
            img_size = getattr(model.config, "input_size", 224)
            forward_kwargs["pixel_values"] = torch.zeros(
                (1, 3, img_size, img_size), dtype=torch.float32, device=device,
            )
        with torch.no_grad():
            out = model(**forward_kwargs)
        return (True, f"OK forward on {device} (last_hidden_state: {tuple(out.last_hidden_state.shape)})")
    except (NotImplementedError, RuntimeError, TypeError, ValueError) as e:
        return (False, f"FAIL forward on {device}: {type(e).__name__}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A.5 LayoutLMv3 MPS availability spike",
    )
    parser.add_argument(
        "--model",
        default="microsoft/layoutlmv3-base",
        help="HF model id (default: microsoft/layoutlmv3-base).",
    )
    args = parser.parse_args()

    print(f"PyTorch:    {torch.__version__}")
    print(f"Platform:   {sys.platform} / {platform.machine()}")
    print(f"MPS avail:  {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built:  {torch.backends.mps.is_built()}")

    devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    print(f"Devices to test: {devices}")
    print(f"Model id: {args.model}\n")

    overall: dict[str, dict[str, tuple[bool, str]]] = {}

    for visual_embed in (True, False):
        label = "A (multimodal)" if visual_embed else "B (text+bbox-only)"
        print("-" * 60)
        print(f"Path {label}, visual_embed={visual_embed}")
        ok, msg, model = try_path(args.model, visual_embed)
        print(msg)
        if not ok:
            overall[label] = {"mps": (False, msg), "cpu": (False, msg)}
            continue

        overall[label] = {}
        for d in devices:
            okf, msg_i = try_inference_on_device(model, d, visual_embed=visual_embed)
            print(f"   {msg_i}")
            overall[label][d] = (okf, msg_i)

    print()
    print("-" * 60)
    print("=== Summary ===")
    mps_passes: list[str] = []
    cpu_passes: list[str] = []
    for path_label, dev_results in overall.items():
        for dev, (ok, _msg) in dev_results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  Path {path_label} on {dev:>4}: {status}")
            if ok and dev == "mps":
                mps_passes.append(path_label)
            elif ok and dev == "cpu":
                cpu_passes.append(path_label)

    # Exit-code logic per PLAN.md §9 Phase A.5: Path B is MVP default.
    if "B (text+bbox-only)" in mps_passes:
        print("\nPASS: Path B (text+bbox-only) runs on MPS (MVP default per PLAN.md §9).")
        return 0
    if "A (multimodal)" in mps_passes:
        print("\nWARN: only Path A (multimodal) runs on MPS; Path B does NOT.")
        print("  Per PLAN.md \u00a79, Path B is MVP default. Path A pulls multimodal forward-ops;")
        print("  this is a deviation. (Phase G `layout.py` will gate this via the")
        print("  `LAYOUTLMV3_USE_VISUAL` env flag once that phase lands.)")
        return 1
    if cpu_passes:
        paths = ", ".join(cpu_passes)
        print(f"\nWARN: no MPS path; CPU fallback ({paths}).")
        print("  layout.py will be slower at runtime but stays on the planned timeline.")
        return 1

    print("\nFAIL: LayoutLMv3 failed on every path/device combination.")
    print("Plan pivot:")
    print(PIVOT_HINTS)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
