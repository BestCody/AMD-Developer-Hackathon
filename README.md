# Phase 1 PDF → UIR Pipeline

Pipeline that ingests PDF documents and emits **Universal Intermediate
Representation (UIR v1.0)** JSON plus chunk-level BGE-small embeddings
into a Weaviate vector index.

> **Status**: ✅ Phase 1 MVP complete (Phases B–N shipped). The CLI runs
> end-to-end on real PDFs, **246** unit tests + **2** slow integration
> tests pass; an in-browser web UI is shipped for LAN testing. See
> [`PLAN.md`](./PLAN.md) for the per-phase exit criteria.

This is the MVP for an AMD-hackathon project (Phase 1, multimodal
unification system). The plan lives in [`PLAN.md`](./PLAN.md); the original
spec sits at [`INSTRUCTIONS.md`](./INSTRUCTIONS.md).

---

## What it does, end to end

```
PDF (file or directory)
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│  ingest  → IBM Docling transformer ── DoclingDocument ── typed regions │
│           → LayoutClassifier (heuristic, 4 labels)       │
│           → tables (Docling) → markdown                 │
│           → chunks (BGE tokenizer, ~256 tokens, overlap) │
│           → enrich (spaCy NER + co-occurrence)           │
│           → embed  (BAAI/bge-small-en-v1.5, 384-d)       │
│           → assemble UIRV1 (pydantic) → JSON             │
│           → upsert (Weaviate: UIRChunks_v1 + ParentDoc) │
└──────────────────────────────────────────────────────────┘
          │
          ▼
out/{doc_id}.uir.json   +   Weaviate: chunks + parent doc
```

The orchestrator's text-extraction path uses IBM Docling -- a
of real OCR; one-line swap behind `_get_page_text()` if you want EasyOCR
primary / Tesseract fallback.

---

## Quickstart (macOS Apple Silicon dev box)

Requires Python 3.10–3.13 (Python 3.14 wheels lag for some ML libraries
— see "Python compatibility" below), Docker Desktop (≥ 4 GB RAM), and
the `tesseract` CLI binary on `PATH`.

```bash
# 1. System deps
brew install tesseract

# 2. venv + install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm

# 3. Weaviate (docker compose; cold pull ~150 MB, warm restart ~30s)
docker compose up -d
sleep 30 && curl -fsS http://localhost:18080/v1/meta | head -20

# 4. Drop a PDF and run the pipeline (skip Weaviate to keep it fast)
python pipeline.py tests/fixtures/sample_pdfs/flat_text.pdf \
        --output-data data/output/ --skip-weaviate
ls data/output/   # -> doc_<uuid>.uir.json

# 5. Or sweep a directory of PDFs
python pipeline.py tests/fixtures/sample_pdfs/ --output-data data/output/ --skip-weaviate

# 6. Run the Phase N web UI (defaults to LAN-visible on :5050)
python web.py                                 # http://192.168.0.X:5050
HOST=127.0.0.1 python web.py                  # loopback-only
PORT=8080 python web.py                       # custom port
```

The web UI serves a single-page upload form with progress polling and
an in-browser UIR JSON viewer. Default port is **5050** (not 5000) to
avoid macOS AirPlay Receiver's hold on `:5000`.

---

## Modules (`src/uir_pipeline/`)

| Module | Role | Notes |
|---|---|---|
| `uir_schema.py` | Pydantic models — `UIRV1`, `ChunkNode`, `Entity`, `Relationship`, `Provenance`. | Backbone of every other module. |
| `weaviate_store.py` | Weaviate client wrapper; ensures `UIRChunks_v1` and `UIRParentDoc_v1` collections exist; upserts on `run()`. | Optional path; CLI / web skip by default. |
| `device.py` | Picks MPS / CUDA(ROCm) / CPU and dtype via env IR defaults. | Single source of truth for compute backend. |
| `ingest.py` | `DocumentInput`: file URI, sha256, page count, title. | The pipeline's only I/O surface. |
| `ocr.py` | `EasyOCRReader` (primary), Tesseract fallback, `DetectedWord` / `OCRPage`. | Fast path bypasses this with synthetic DetectedWords. |
| `layout.py` | Heuristic `LayoutClassifier` (paragraph / heading / list_item / caption); lazy `LayoutLMv3Backend`. | Heuristic version is the default. |
| `tables.py` | `extract_tables()` — pdfplumber → list of `TableDraft` with markdown + bbox. | Handles missing cells, page-clipped, truncated rows. |
| `chunk.py` | `chunk_text()` — sentence-aware, BGE-tokenized, target ~256 tokens w/ overlap; recursive halving for oversize segments. | Tested with both stub and live BGE tokenizer. |
| `enrich.py` | `enrich_chunks()` — spaCy NER + co-occurrence relationships + topics. | All optional; off-by-default in tests. |
| `embed.py` | BGE-small singleton tokenizer + embedder; `mean_pool_vectors`; `upsert_chunks` / `upsert_parent_doc`. | Same model as the BGE chunker. |
| `pipeline.py` | **Public orchestrator** — `run(input_path, output_dir, *, skip_weaviate, dry_run, with_embeddings, page_numbers, on_progress) -> PipelineResult`. | The only entry point CLI and web call. |
| `web.py` | Flask front-end: `POST /api/run`, `GET /api/status/<id>`, `GET /api/result/<id>`, `GET /api/download/<id>`, `GET /api/health`. | Threaded job runner; per-job status dict; JSON streaming for in-browser display. |
| `utils.py` | `deterministic_node_id(prefix, *seeds)` (URL-`uuid5`), `bbox_from_pixel`, `get_bge_tokenizer` lazy singleton. | Shared helpers; everything else threads through here. |
| `logging_config.py` | JSON-line formatter + `configure()` + `attach_doc_log` / `detach_doc_log`. | Idempotent; honors `LOG_LEVEL` / `LOG_FORMAT` env vars. |
| `__init__.py` | Module docstring only — heavy modules (`ocr`, `layout`) are **opt-in** so light callers don't pay the torch-cost. | Documentation, not code. |

---

## Pipeline scripts

```text
phase1-pipeline/
├── pipeline.py        ← CLI entrypoint (argparse: input / --output-data / --skip-weaviate / --dry-run / --no-embeddings / --log-{level,format})
├── web.py             ← root launcher for the Phase N Flask web app (defaults: HOST=0.0.0.0, PORT=5050)
├── scripts/
│   ├── spike_layoutlmv3.py          ← Phase A.5 de-risking: prove LayoutLMv3 forward pass works on MPS
│   ├── generate_fixtures.py         ← emits tests/fixtures/sample_pdfs/{flat_text,dense_table,multi_column}.pdf
│   └── export_uir_json_schema.py    ← dump UIRV1 pydantic schema to JSON Schema / TS / OpenAPI
└── src/uir_pipeline/   ← the package itself
```

---

## Tests

```bash
.venv/bin/python -m pytest                          # default: unit + smoke, no model downloads
.venv/bin/python -m pytest -m "not slow"            # same — explicit marker filter
.venv/bin/python -m pytest -m slow                  # only the integration tests (real PDFs + heavy deps)
.venv/bin/python -m pytest tests/test_web.py -v    # just the Flask front-end
.venv/bin/python -m pytest tests/integration/       # only the E2E pipeline smoke
```

**Counts (current)**:
- 250+ passing tests (see the latest `pytest -q` tail).
- 4 environmental skips (require live Weaviate container).
- 2 slow integration tests included in the default run; 2 more under `tests/integration/` are tier3-marked and excluded by default. Gate them off with `pytest -m "not slow"`; run only the Tier 3 image-aware path with `pytest -m tier3`.

Pytest markers are registered in `pytest.ini` (`slow`); tests under
`tests/integration/` are auto-marked `slow` by `tests/conftest.py`.

**Fixtures**:
| File | Profile | Purpose |
|---|---|---|
| `tests/fixtures/sample_pdfs/flat_text.pdf` | All body text, single column | Smoke test the text path. |
| `tests/fixtures/sample_pdfs/dense_table.pdf` | Heavy table content | Smoke test `tables.py` markdown rendering. |
| `tests/fixtures/sample_pdfs/multi_column.pdf` | Two-column body | Smoke test rough reading-order. |
| `tests/fixtures/sample_pdfs/figure_rich.pdf` | 1 colored bar-chart raster + caption | Tier 3 (Florence-2) e2e: asserts `ChunkNode.modal_features.figure.image_b64` is set with a non-empty caption text. |

Generated by `python scripts/generate_fixtures.py` (reportlab, fully
deterministic).

---

## First-run model downloads

The first run downloads ~700 MB of weights to local caches:

| Model | Used by | Size | Cache dir |
|---|---|---|---|
| `microsoft/layoutlmv3-base` | `layout.LayoutLMv3Backend` (lazy) | ~500 MB | `~/.cache/huggingface/` |
| EasyOCR English (`english_g2`) | `ocr.EasyOCRReader` (lazy) | ~100 MB | `~/.EasyOCR/model/` |
| `BAAI/bge-small-en-v1.5` | `chunk` + `embed` | ~133 MB | `~/.cache/huggingface/` |
| spaCy `en_core_web_sm` | `enrich.Enricher` (lazy) | ~40 MB | spacy default |

All loads are lazy / on-demand. The CLI's fast path skips OCR and LayoutLMv3
until you enable them, so a cold run only spends 173 MB (BGE-small + spaCy).

---

## Hardware

- **Development (MVP)**: Apple M4 Pro (macOS Apple Silicon). MPS preferred
  for transformers / sentence-transformers; CPU fallback when MPS kernels
  are missing.
- **Production**: AMD cloud GPUs (MI250X / MI300X, ROCm). PyTorch's
  compatibility layer makes ROCm appear as CUDA via the standard API, so
  `device.pick_device()` returns `"cuda"` and the existing code runs
  unchanged after switching to a `rocm/pytorch` Docker base image and
  setting `HIP_VISIBLE_DEVICES=0`. **No end-to-end ROCm validation has
  run yet on AMD hardware** — the M-series dev box is the only tested
  environment.
- **Tier 3 image-aware (Florence-2)**: cross-platform matrix per
  `PLAN_TIER3.md §Cross-platform matrix` — M4 Pro / MPS runs the caption
  stage lazily; MI300X / ROCm accelerates to sub-second captioning on
  the same Python path.

### Tier 3 — first-run validator

```bash
.venv/bin/python scripts/check_caption_env.py        # env snapshot + weight probe
.venv/bin/python scripts/check_caption_env.py --full # + Florence-2 round-trip
.venv/bin/python scripts/check_caption_env.py --json # machine-readable for CI
```

- Exit `0` — Florence-2 loadable; pipeline will run captions by default.
- Exit `1` — degraded (weights not cached); pipeline will **fail-soft** and
  emit figures **without** captions (no UIR schema break).
- Exit `2` — broken (`torch` missing); caption stage cannot run.

### Tier 3 — figure-rich fixture

A 4th deterministic fixture exercises the image-aware path:

| File | Profile | Purpose |
|---|---|---|
| `tests/fixtures/sample_pdfs/figure_rich.pdf` | 1 colored bar-chart raster (220x140 px) + Figure caption text | E2E test that `pipeline.run` wires `modal_features.figure.image_b64` onto a `ChunkNode` (Tier 3 integration test, marked `slow` + `tier3`). |

Generated by `python scripts/generate_fixtures.py figure_rich`.

---

## Repo layout

```text
phase1-pipeline/
├── PLAN.md                       ← per-phase plan + decision log (read first)
├── INSTRUCTIONS.md               ← upstream spec (read-only)
├── README.md                     ← you are here
├── pipeline.py                   ← CLI entrypoint
├── web.py                        ← Flask web launcher
├── requirements.txt
├── pytest.ini                    ← registers the 'slow' marker
├── docker-compose.yml            ← Weaviate stack
├── data/
│   ├── input/                    ← drop PDFs here for CLI runs
│   └── output/                   ← where {doc_id}.uir.json lands
├── scripts/
│   ├── spike_layoutlmv3.py       ← Phase A.5 de-risking
│   ├── generate_fixtures.py      ← emits the 3 sample_pdfs/*.pdf
│   └── export_uir_json_schema.py ← UIRV1 → JSON Schema / TS / OpenAPI dump
├── src/uir_pipeline/             ← package (see "Modules" table above)
└── tests/
    ├── conftest.py               ← slow-marker auto-tagger, tmp dirs
    ├── test_utils.py
    ├── test_ingest.py
    ├── test_ocr.py
    ├── test_layout.py
    ├── test_tables.py
    ├── test_chunk.py
    ├── test_enrich.py
    ├── test_embed.py
    ├── test_weaviate_store.py
    ├── test_web.py               ← Flask front-end via test_client
    ├── fixtures/sample_pdfs/     ← reportlab-generated PDFs
    └── integration/
        └── test_pipeline_smoke.py    ← full E2E on real PDFs (slow)
```

---

## Known risks / limitations (current MVP)

- **Synthetic DetectedWord bboxes in fast path.** When the orchestrator
  uses pdfplumber text extraction (the default), the per-word bboxes are
  full-canvas `(0,0,1000,1000)` placeholders. The LayoutClassifier's
  y-clustering still works because the synthetic bboxes preserve word
  order, but you lose the ability to attach words to specific page
  regions. **Swap to `pdfplumber.extract_words()` for real geometry**
  (next Phase O).
- **easyocr + numpy 2.x.** Some internal easyocr codepaths assume pre-2.x
  APIs. If you see `np.float was removed` at runtime, pin `numpy<2.0`
  in `requirements.txt`.
- **LayoutLMv3 multimodal branch is out of scope.** The lazy
  `LayoutLMv3Backend` runs the text+bbox branch (`visual_embed=False`);
  Detectron2 is deliberately disabled (PLAN §9 Phase A.5).
- **Web UI is single-process.** Phase N uses the Flask dev server on a
  single thread; swap to `gunicorn` / `waitress` for LAN production.
  No auth, no rate-limit — anyone on the LAN can upload.
- **OCR is fast-path-only.** Real EasyOCR primary + Tesseract fallback is
  implemented in `ocr.py` but the orchestrator doesn't use it yet.
- **AMD ROCm not yet validated.** `device.py`'s abstractions are designed
  to work on ROCm (PyTorch's API is identical through CUDA), but no
  end-to-end run has been done on MI300X hardware.

---

## License

TBD.
