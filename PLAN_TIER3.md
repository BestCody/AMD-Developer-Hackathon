# PLAN — Tier 3 — Image-Aware UIR Pipeline (Florence-2)

> Companion to [`PLAN.md`](./PLAN.md). Tier 3 makes the pipeline **image-aware**:
> regions detected as `figure` are routed through a vision-language model that
> produces a short, structured text caption, which is then embedded with the
> same BGE-small text embedder used everywhere else. Vector search stays
> text-only and single-modality — no CLIP, no Weaviate schema break.

---

## Decisions (locked in by user)

| | |
|---|---|
| **Captioning model** | `microsoft/Florence-2-base` (~1.5 GB, MIT license). |
| **Default prompt** | `<MORE_DETAILED_CAPTION>` (~80 words). `<CAPTION>` (short) + `<OCR>` available behind a flag. |
| **Device matrix** | Apple Silicon (MPS, dev) + AMD ROCm (MI250X / MI300X, production). CUDA and Windows explicitly **out of scope** for Tier 3. |
| **Backwards compat** | Pre-Tier-3 `.uir.json` files remain valid (no schema change). |
| **Failure mode** | If Florence-2 fails to load (offline env, OOM, missing accelerate), the orchestrator logs and emits figures with **no caption text** instead of throwing. |

---

## Goals / non-goals

**Goals**
- Detect figure regions on the page (Tier-1's real-`extract_words()` bboxes + Tier-2's PyMuPDF crop).
- Caption each figure with Florence-2 in a single forward pass (lazy-loaded, singleton-cached).
- Embed the caption text via `bge-small-en-v1.5` into the same 384-d vector space as text chunks.
- Cross-platform: identical code path on MPS and ROCm (no `#ifdef device`, no separate model files).

**Non-goals (kept for later tiers)**
- Multimodal vectors (CLIP, SigLIP).
- LayoutLMv3 multimodal branch (Detectron2 dep).
- Per-question VQA over regions (Florence-2 supports `<VQA>` but we don't expose it yet).
- Real figure-cell OCR via Florence-2's `<OCR_WITH_REGION>` (mention in roadmap).

---

## Architecture flow

```
PDF page
  │
  ├─ IBM Docling transformer ── DoclingDocument.items (ordered, bbox on 0-1000)
  ├─ IBM Docling PicPicItem walker ── dr.pictures (bbox on 0-1000)
  │     │
  │     └─ PyMuPDF.get_pixmap(clip=Rect)    # render PNG crop (Tier 2)
  │
  ├─ LayoutClassifier.classify(...)
  │     └─ labels figure, paragraph, heading, list_item, caption
  │
  ▼ (NEW) caption_images(cropped_pils, prompt="<MORE_DETAILED_CAPTION>")
  │     └─ _get_model() singleton (lazy-loaded, cached for process lifetime)
  │     └─ processor(text=prompt, images=pils) → .to(device, float16)
  │     └─ model.generate(...) → processor.post_process_generation(...)
  │
  ▼ StructureNode(type="figure",
                   text=caption,             # reuses ChunkNode.text path → BGE embeds
                   bbox=img.bbox,
                   modal_features={"image_b64": "...",
                                   "caption_prompt": "<MORE_DETAILED_CAPTION>",
                                   "caption_model": "Florence-2-base"})
  │
  ▼ chunking/embed/indexing (existing pipeline, no change)
```

**Key invariant**: a figure becomes a `ChunkNode` whose `text` field is the caption. The BGE embedder sees only text and produces a 384-d vector aligned with the rest of the doc. No new chunking strategy, no new index.

---

## File changes (file:line-precise)

### `requirements.txt` (Tier 3 additions)
```text
accelerate>=0.27           # needed for device_map & lazy dispatch
einops>=0.7               # pulled by Florence-2 trust_remote_code path
```
- Already present: `torch>=2.1`, `transformers>=4.36`, `sentence-transformers>=2.2`, `pillow>=10`.
- ROCm path: build base image from `rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_v2.3` then `pip install -r requirements.txt` + the two new pins.
- MPS path: stock PyTorch 2.6 from `pip install torch` already includes MPS; nothing extra to install.

### `src/uir_pipeline/utils.py`
Add `base64_png(pil_or_bytes)` helper (no deps beyond `base64` + `Pillow`). Reuse existing lazy singleton pattern from `get_bge_tokenizer()`.

### `src/uir_pipeline/caption.py` *(new, ~120 LOC)*
- `MODEL_ID = "microsoft/Florence-2-base"`
- `DEFAULT_PROMPT = "<MORE_DETAILED_CAPTION>"`
- Lazy singleton `_get_florence2(device=None, dtype=torch.float16) -> tuple[AutoProcessor, AutoModelForCausalLM]`
  - First call: `AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)`, `model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device).eval()`.
  - Subsequent calls: return the cached `(processor, model)` pair.
  - Use `device.py`'s `pick_device()` / `pick_dtype()` helpers — same as `embed.py:_get_bge`.
- `caption_images(images: list[PIL.Image.Image], *, prompt=DEFAULT_PROMPT, max_new_tokens=1024, num_beams=3) -> list[str]`
  - Batched via `processor(text=[prompt]*len(images), images=images, return_tensors="pt", padding=True)` (Florence-2 supports batching but not all HF examples show this).
  - `model.generate(input_ids=..., pixel_values=..., max_new_tokens=..., num_beams=...)`.
  - Decode + `processor.post_process_generation(...)` per item.
- Fail-soft: if `_get_florence2()` raises (offline / OOM), log + return `""` for each image so the orchestrator can keep going.

### `src/uir_pipeline/pipeline.py` (integrate after `tables`, before `chunk`)
Insert between the existing "tables" stage and the "chunk" stage (~line 134 today):
1. **Collect figure regions** from `DoclingResult.pictures` (already on 0-1000).
2. **Render crops** with PyMuPDF (Tier 2 dependency; add to `requirements.txt` if not present).
3. **Caption** via `caption_images(pils, prompt=prompt)`.
4. **Build figure chunks** with the caption as `text` + base64 PNG in `modal_features.image_b64`.
5. Add a per-progress step `image_caption` at ~60% (between `tables:55` and `chunk:70`).
6. Fail-soft: if caption stage raises or returns empty, log + continue without the figure chunks. Tier 1+2 don't regress.

### `src/uir_pipeline/device.py`
Verify `pick_device()` returns `"cuda"` on ROCm (PyTorch abstracts ROCm through the CUDA API on MI300X — `torch.cuda.is_available() == True` + `torch.version.hip is not None`). Honor an `UIR_FORCE_DEVICE` env override for the test matrix.

---

## Cross-platform matrix

| Device | Gate | `pick_device()` | `pick_dtype()` | Cold-load | Per-figure latency | Notes |
|---|---|---|---|---|---|---|
| **M4 Pro (MPS)** | macOS Apple Silicon | `"mps"` | `float16` (fallback `float32` on NaN — see risk 4) | ~2 s | ~0.6 s | Single-streamed inference. Batch = cap at 4 images to avoid swap. |
| **MI300X (ROCm)** | `rocm/pytorch:rocm6.0_…` Docker | `"cuda"` | `float16` | ~3 s (cold cache) | ~0.05 s batched (8 fig/page) | `torch.compile(model)` gives +15% throughput on ROCm Triton. |
| **Acceleration fallback** | env without MPS + ROCm | `"cpu"` | `float32` (FP16 unstable on x86-CPU for Florence's custom ops) | ~6 s | ~2-3 s | CI-only smoke; not a default UX path. |

All three paths go through the same `caption_images()` function. No `#ifdef`.

---

## Test strategy

### Pure unit (`tests/test_caption.py`)
- `caption_images_with_stubbed_model`: monkeypatch `caption._get_florence2` to return a `(_FakeProcessor, _FakeModel)` pair producing canned `<MORE_DETAILED_CAPTION>` outputs. Assert the function returns the expected list and **never touches the real model**.
- `caption_images_empty_list_returns_empty`: edge case.
- `caption_image_filter_too_small_drops_tiny_crops`: assert regions smaller than 50×50 px are skipped (risk 1 mitigation).
- `fail_soft_on_model_load_failure`: monkeypatch `_get_florence2` to raise `OSError("offline")`. Assert `caption_images()` returns `[""]` for each input (no exception bubbles).

### MPS smoke (`@pytest.mark.mps`, `tests/integration/test_caption_mps.py`)
- Skip unless `torch.backends.mps.is_available()`.
- Load the real Florence-2 once (slow, ~30 s cold). Caption a 256×256 synthetic PIL image filled with a colored geometric pattern. Assert output is a non-empty string ending in punctuation.

### ROCm smoke (`@pytest.mark.rocm`, `tests/integration/test_caption_rocm.py`)
- Skip unless `torch.cuda.is_available()` *and* `torch.version.hip is not None`.
- Same shape as the MPS smoke but against ROCm-claimed CUDA. Assert `torch.cuda.get_device_name(0)` contains `"MI300X"` or `"MI250"` (warn if not).

### End-to-end (`@pytest.mark.slow`, `tests/integration/test_pipeline_tier3.py`)
- Run full pipeline on `tests/fixtures/sample_pdfs/flat_text.pdf` (current fixture set has no figure-rich PDF — see §"Fixtures" below).
- Stub Florence-2 with the same canned-output fake as the unit tests (use a module-level pytest fixture that gates on `os.environ.get("UIR_TIER3_E2E") == "live"`).
- Assert at least one `ChunkNode` has `modal_features.image_b64` set *and* `text` non-empty.

### `scripts/check_caption_env.py` *(new)*
Standalone diagnostic. Prints:
- `device`, `dtype`, `torch.cuda.is_available()`, `torch.version.hip`, `torch.backends.mps.is_available()`.
- One synthetic forward pass + wall-clock cold-load time.
- Empty `caption_images` round-trip on a 128×128 PIL: prints "OK" or reports failure mode.

This script is what CI runs on the Azure ML MI300X box to validate the runner without paying for the pytest collection cost.

---

## Fixtures

Current `tests/fixtures/sample_pdfs/{flat_text,dense_table,multi_column}.pdf` are **text-only**. Tier 3 needs a figure-rich PDF.

- **Generate** `tests/fixtures/sample_pdfs/arxiv_figure.pdf` via `reportlab`: a single-page doc with a colored bar chart rendered as a vector figure plus a caption "Figure 1: Quarterly revenue by region." This guarantees deterministic bbox geometry for the caption test.
- **Plus** a real public-domain figure-heavy PDF (Wikipedia print-to-PDF, an FCC public report) — manual download, one-shot, into `tests/fixtures/sample_pdfs/real_figures.pdf`.
- Generator: extend `scripts/generate_fixtures.py` with a `figure_rich` profile.

---

## Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | Florence-2 hallucinates on tiny/empty image regions (logos, decorative dots). | Drop crops < 50x50 px (configurable); cap per-page figure count at 16. |
| 2 | `max_new_tokens=1024` overflows on dense diagrams (rare). | Truncate response at sentence boundary; log dropped suffix. |
| 3 | `trust_remote_code=True` is an infosec flag in production. | First-run review (manual): read `microsoft/Florence-2-base`'s modeling file, snapshot weights under a known SHA. Future: vendor a stripped `florence2_local/` package. Phase 2 / Tier 4 candidate. |
| 4 | `float16` produces NaNs on some MPS attention kernels (M-series pre-M4 has known issues). | Override default to `float32` on MPS if NaN detected in a calibration run; `scripts/check_caption_env.py` does this check. |
| 5 | MSI / batch inference forces OCR off the critical path. | Captioned at 60% in pipeline progress; user-visible progress says `"captioning: page 3 of 12"`. Batch 4 images per call on MPS, 8 on ROCm. |
| 6 | Weight download flake on first run (HF rate-limit / offline). | Lazy `_get_florence2` retries 3× with exponential backoff. Fail-soft path returns caption-less figures; README's "first-run downloads" table extended to include Florence-2 size + cache dir. |
| 7 | ROCm 6 wheel pinning: MI300X needs ROCm 6.2+ for full graph capture; older ROCm 5.x silently deoptimizes. | Pin Docker base image tag in `scripts/devcontainer.json` (`rocm6.2_…`). Production deployment doc explicitly warns against ROCm 5.x. |
| 8 | Caption model license drift. | `microsoft/Florence-2-base` is MIT; pin `MODEL_ID` to a specific commit SHA in `caption.py` so future re-pulls fail loudly. |

---

## Build order (within Tier 3)

1. **Scaffold `caption.py`** with the lazy singleton + stub-mode `_FakeModel` fixture pattern (mirrors `tests/test_web.py`'s `fake_run`).
2. **Unit tests** (`tests/test_caption.py`) for: stub round-trip, empty list, tiny-crop filter, fail-soft on load failure.
3. **Pipeline integration** (`pipeline.py`): insert figure-collect → crop → caption → `ChunkNode` assembly between `tables` and `chunk` stages. Per-page progress callback extended to `"caption"` at 60%.
4. **Fixtures**: extend `scripts/generate_fixtures.py` with `figure_rich` profile; commit `arxiv_figure.pdf`.
5. **`scripts/check_caption_env.py`**: standalone env diagnostic. Documents the matrix in README.
6. **MPS smoke test** (`tests/integration/test_caption_mps.py`): real load, 256×256 PIL, assert non-empty.
7. **ROCm smoke test** (`tests/integration/test_caption_rocm.py`): same shape on AMD runner.
8. **End-to-end pipeline integration test** (`tests/integration/test_pipeline_tier3.py`): real PDF, stubbed Florence-2, assert figure chunks with `modal_features.image_b64`.
9. **Docs**: extend README quickstart with the "Run on MI300X" section (Docker base image, `HIP_VISIBLE_DEVICES=0`, expected cold-load latency).

Each step lands as a single PR-style commit. Steps 1–4 deliver an **internally-testable** caption path; steps 5–9 are pure ops / cross-platform validation.

---

## What "done" looks like for Tier 3

- `pytest` defaults stay fast (no Florence-2 weight download) — still 246+ unit tests passing.
- `pytest -m mps` on the M4 Pro dev box: 1 new slow test.
- `pytest -m rocm` on the MI300X box: 1 new slow test.
- `pytest -m slow` (e2e): 1 new integration test asserts a `ChunkNode` carries both `image_b64` and a non-empty caption text.
- `scripts/check_caption_env.py` returns `OK` on both target devices.
- README §"Hardware" extended with the apple-silicon-vs-ROCm matrix; `scripts/check_caption_env.py` documented as the first-run validator.

---

## Out-of-scope, but worth tracking

- Vendor Florence-2 weights + modeling files locally (security hardening for the AMD cloud deployment).
- `<OCR_WITH_REGION>` mode to extract figure cells instead of captions — relevant for financial tables.
- Florence-2-large weights (~3 GB) for higher quality — quick follow-up once base is stable.
- CLIP-based multimodal index for figure-only similarity queries.
