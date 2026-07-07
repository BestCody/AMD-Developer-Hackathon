# Phase 1 PDF → UIR Pipeline — MVP Plan

> Status: **in progress** - Repository initialized and pushed to GitHub: BestCody/AMD-Developer-Hackathon. Owner: AMA (AMD Hackathon, Phase 1, multimodal unification system)
> Upstream spec: [`INSTRUCTIONS.md`](./INSTRUCTIONS.md)

---

## 1. TL;DR

Build a production-shape Python pipeline that ingests PDF documents and emits **Universal Intermediate Representation (UIR v1.0)** JSON, plus chunk-level vector embeddings into a local Weaviate index.

**MVP scope**: end-to-end on **one PDF** through a local CLI, fully validated. Multi-doc batch plumbing exists in the CLI but is not the acceptance bar.

**Target hardware — now**: Apple M4 Pro (macOS, Docker Desktop, MPS GPU).
**Target hardware — Phase 2**: AMD cloud GPUs (MI250X / MI300X on ROCm). Stack choices pre-baked for portability; switch is `device=cuda` + image swap.

---

## 2. Scope

### In scope (MVP acceptance bar)

1. All eight pipeline stages (ingest → OCR → layout → tables → chunk → enrich → embed → assemble) run sequentially on 1 PDF.
2. Output written to `/output/{doc_id}.uir.json` and validated against the exact UIR v1.0 schema from INSTRUCTIONS.md.
3. Chunk-level embeddings upserted into a local Weaviate collection, with the UIR metadata attached.
4. CLI:
   - `python pipeline.py <file.pdf> --output-data <dir>`
   - `python pipeline.py <dir/> --output-data <dir/> --batch-size N`
   - Flags: `--dry-run`, `--skip-weaviate`, `--log-level`.
5. **Unit tests** per module (heavy deps mocked) + **1 end-to-end integration test** on one real sample PDF.
6. **3–5 sample PDFs** in `tests/fixtures/sample_pdfs/`.
7. **README quickstart** + **UIR schema reference**.

### Out of scope for MVP (explicit non-goals)

| Non-goal | Status | Why |
|---|---|---|
| S3 ingest (`s3://...` source) | Stubbed in CLI parser — runtime no-op | No S3 creds available; deferred. |
| 100-PDF benchmark suite | Deferred to Phase 2 | MVP acceptance = 1 PDF. |
| Production `Dockerfile` | Not in MVP | Dev runs locally. |
| Redis OCR cache | Not in MVP | Premature optimization. |
| Async multi-doc batching | Not in MVP (CLI accepts dir but processes serially) | Throughput target deferred. |
| Generic `topics` field via LDA | Stub returns `[]` | gensim dep + tuning cost; defer to Phase 2. |
| `<500MB` Docker image target | **Dropped from MVP** | Incompatible with current model stack (will revisit post-ONNX export). |

---

## 3. Spec adjustments from INSTRUCTIONS.md

These reflect explicit deviations from the original spec, with rationale.

| Spec said | We're doing | Why |
|---|---|---|
| PaddleOCR (primary) | **EasyOCR** (primary) | AMD ROCm portability — PaddlePaddle has no production ROCm build. |
| Tesseract (fallback) | Tesseract (fallback) | Unchanged. |
| `camelot` or `pdfplumber` for tables | **`pdfplumber`** | Lighter install on macOS; both OK on ROCm. |
| `text-embedding-3-small` (OpenAI) or `bge-small` | **`BAAI/bge-small-en-v1.5`** via `sentence-transformers` | No OpenAI key available; PyTorch-native = ROCm-friendly. |
| N/A | **`device.py` helper** selects `cuda > mps > cpu` | One env-line AMD switch. |
| Topics via LDA or embedding clustering | **Stub: empty list** for MVP; TODO Phase 2 | Avoids gensim dep; not an acceptance gate. |
| Docker image <500MB | **Dropped** | Re-baselined after model stack is finalized. |

The UIR JSON output still matches the spec byte-for-byte; only the internals changed.

---

## 4. AMD migration plan (Phase 2 — informational, pre-baked in MVP)

Reading constraints today means the AMD cloud switch is **a port, not a redesign**:

| Component | Mac (MVP) | AMD (Phase 2) | Notes |
|---|---|---|---|
| Python / pip | macOS Python 3.10+ | Linux + ROCm 6.2+ | Use AMD's `rocm/pytorch:latest` Docker image (verified production-ready for these models). |
| LayoutLMv3 | `device="mps"`, `float32`, cpu fallback | `device="cuda"` (ROCm compatible API), `float16` | Same `transformers` code path; ~5–10× throughput. |
| EasyOCR | `torch`, `device="mps"` or "cpu" | `torch`, `device="cuda"` | Same PyTorch op graph; runs on ROCm. |
| bge-small | `torch` + MPS | `torch` + ROCm | Identical. |
| pdfplumber | CPU | CPU | No change. |
| spaCy | CPU NER | CPU NER | No change. |
| Weaviate | Docker Desktop, single-node | Either: managed Weaviate Cloud Services OR AMI-based self-host | Client API unchanged. |
| Migrating toggle | Local dev env | `CUDA_VISIBLE_DEVICES=0` on AMD host | Single env-var drive. |

**Pre-baked checks** (already done at planning time):
- ✅ Verified PaddlePaddle is **not** in the stack (removed via the OCR swap).
- ✅ Verified PyTorch+ROCm is **production-ready** for LayoutLMv3, EasyOCR, and bge-small.
- ✅ Confirmed `device.py` is the single point of hardware abstraction.

**One known AMD-portability risk**: Weaviate's default HNSW index has unbounded memory at large scale. **Mitigation in Phase 2**: configure `vectorIndexConfig` with `quantizer` enabled; benchmark before 1k-docs/day deployment.

**Optional 1-hour CUDA smoke test on the 2020 Linux+NVIDIA laptop**: not on the critical path. Recommended AFTER Phase N (after MVP is green) as a Phase-2 prep step. Does not slow MVP dev.

---

## 5. Architecture

```
PDF (.pdf, filesystem)
    │
    ▼
[ ingest ]   → validate, sha256, pypdf metadata                  → DocumentInput
    │
    ▼
[ ocr ]      → EasyOCR per page → text + bbox + confidence       → OCRResult
    │             └─ fallback: pytesseract
    ▼
[ layout ]   → LayoutLMv3 (mps|cpu) → region classification      → LayoutRegion[]
    │
    ▼
[ tables ]   → pdfplumber on detected table regions → markdown   → TableNode[]
    │
    ▼
[ chunk ]    → growing window 256-512 tokens, 10-20% overlap     → Chunk[]
    │
    ▼
[ enrich ]   → spaCy NER + co-occurrence                          → Entities[], Relationships[]
    │
    ▼
[ embed ]    → bge-small-en-v1.5 (mps|cuda|cpu) → 384-d vectors   → Embeddings
    │
    ▼
[ assemble ] → Pydantic UIR v1.0 + provenance block               → UIR JSON
    │             └─ Weaviate upsert (chunk-level)
    ▼
{output_dir}/{doc_id}.uir.json
{logs_dir}/{doc_id}.log
weaviate://UIRChunks_v1
```

Per-stage outputs are immutable `@dataclass(frozen=True)` value objects so they pass cleanly across the orchestrator and into tests without exotic mocking. Confidence scores are propagated through the pipeline and surfaced in `modal_features.quality` per chunk.

---

## 6. Tech stack & rationale

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | Strict type hints, Pydantic v2. |
| OCR (primary) | **EasyOCR** | PyTorch-native, AMD-portable. English: `en` model only at MVP. |
| OCR (fallback) | pytesseract | CPU, failsafe. |
| Layout | LayoutLMv3 base (`microsoft/layoutlmv3-base`) | Standard `transformers` load. |
| Tables | pdfplumber | Markdown output + bbox preserved. |
| Chunking | Custom growing-window | `AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")` for token counting (NOT tiktoken); sentence boundaries honored when feasible. |
| Entities | spaCy `en_core_web_sm` | Standard NER. |
| Relationships | Co-occurrence heuristic (within chunk, within distance k) | Transformer RE deferred to Phase 2. |
| Topics | Stub: `[]` | LDA deferred. |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | 384-dim, 133 MB, ~1k sent/sec on M-series MPS. |
| Vector store | Weaviate v1.26.4, Docker Desktop, ARM64 | `vectorizer: none` (BYO vectors). |
| Schema | Pydantic v2 | JSON Schema exported to `docs/uir.schema.json`. |
| Tests | pytest + pytest-mock | Heavy deps mocked; one real-PDF integration test. |
| Logging | `logging` + `python-json-logger` | Per-doc JSON line logs + stdout. |

---

## 7. Project layout

```
phase1-pipeline/
├── PLAN.md                       ← you are here
├── README.md
├── INSTRUCTIONS.md               ← upstream spec (do not modify)
├── docker-compose.yml            ← Weaviate (ARM64 image)
├── requirements.txt              ← pinned versions
├── .env.example                  ← sample env vars
├── pipeline.py                   ← CLI entrypoint (argparse wrapper around src.uir_pipeline.pipeline)
├── src/
│   └── uir_pipeline/
│       ├── __init__.py
│       ├── uir_schema.py         ← Pydantic models matching INSTRUCTIONS.md schema exactly
│       ├── device.py             ← cuda > mps > cpu selector (single source of truth)
│       ├── ingest.py
│       ├── ocr.py
│       ├── layout.py
│       ├── tables.py
│       ├── chunk.py
│       ├── enrich.py
│       ├── embed.py
│       ├── weaviate_store.py
│       ├── pipeline.py           ← programmatic orchestrator
│       ├── logging_config.py
│       └── utils.py              ← uuid5 helpers, bbox, token counting, time helpers
├── tests/
│   ├── conftest.py
│   ├── test_uir_schema.py
│   ├── test_ingest.py
│   ├── test_ocr.py
│   ├── test_layout.py
│   ├── test_tables.py
│   ├── test_chunk.py
│   ├── test_enrich.py
│   ├── test_embed.py
│   ├── test_pipeline_integration.py
│   └── fixtures/
│       └── sample_pdfs/          ← 3–5 public-domain PDFs
└── data/
    ├── input/                    ← drop PDFs here for dev runs
    ├── output/                   ← UIR JSON outputs
    └── logs/                     ← per-doc structured logs
```

---

## 8. UIR schema mapping (Pydantic → spec)

The Pydantic models in `uir_schema.py` mirror the spec **exactly**. Highlights:

- **IDs**: deterministic strings of the form `"<prefix>_<uuid>"` where prefix is one of `doc`, `section`, `chunk`, `entity`, generated via `uuid5(NAMESPACE_URL, f"{source.uri}|{page}|{region}|chunk_index")`. Typed as **`str` with a regex validator** (`^(doc|section|chunk|entity)_[0-9a-f-]{36}$`) — **NOT** `pydantic.UUID5`, which would reject the prefix.
- **Bounding boxes**: `[x1, y1, x2, y2]` rectangle — 4 ints — validated for length plus `x1 ≤ x2, y1 ≤ y2`. **EasyOCR returns 4-point polygons** (`[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`), so a `polygon_to_bbox()` helper sits upstream of any consumer that expects rectangles.
- **Confidence scores**: `confloat(ge=0.0, le=1.0)` across the tree.
- **ISO8601 timestamps**: `datetime` with timezone-aware parser; serialized to `Zulu` form in JSON.
- **`modal_features`**: free-form `dict[str, dict[str, Any]]` so we can drop in new modality fields in Phase 2 (image, audio) without schema churn.
- **`provenance`**: `model: "LayoutLMv3"`, `version: "1.2.0"` for extraction; `"1.0"` for normalization. Hardcoded constants in `uir_schema.py`.
- **TOC of nested types**:
  - `UIRV1` (root)
  - `Source`, `Metadata`, `Structure`, `StructureNode`, `ChunkNode`, `ModalFeatures`
  - `Entity`, `Relationship`, `Semantics`
  - `Provenance`

The frozen Pydantic models ARE the source of truth for the UIR JSON contract — drift from `INSTRUCTIONS.md` is a failing test, not a code review nit.

---

## 9. Implementation phases (with exit criteria)

Each phase has a 🔍 checkpoint — pause and review before proceeding.

### Phase A — Bootstrap
🔍 exit: directory tree exists, `pip install -r requirements.txt` succeeds cleanly on M4 Pro, README skeleton present, `.env.example` documents `WEAVIATE_URL=http://localhost:8080`.

### Phase A.5 — LayoutLMv3 de-risking spike (BEFORE building schema or OCR around it)
🔍 exit: 30-line spike script that loads `microsoft/layoutlmv3-base` from `transformers`, sends a single OCR'd page (synthetic OK), and confirms inference runs on MPS with `float32`. If it fails: switch to the **text+bbox-only branch** (`--no-visual` on `AutoModelForTokenClassification`); if THAT fails: pivot to **`LayoutParser + PubLayNet` (Detectron2)** or **pure PDF heuristics**. **STOP and reshape the plan if this fails** — LayoutLMv3 is too central to discover at Phase G.

### Phase B — UIR Pydantic schema
🔍 exit: schema loads; `pydantic.UIRV1.model_validate_json(spec_example)` succeeds with the example from INSTRUCTIONS.md; `pytest tests/test_uir_schema.py` is green.

### Phase C — Weaviate via docker-compose
🔍 exit: `docker compose up -d` brings Weaviate; `curl http://localhost:18080/v1/meta` returns 200; image is `cr.weaviate.io/semitechnologies/weaviate:1.27.0` (ARM64 verified). Host port mapping is **18080→8080** to dodge the common `:8080` collision on macOS dev machines.

### Phase D — `device.py` hardware selector
🔍 exit: `device.py` selects correctly on M-series (returns `"mps"`), reports `"cpu"` on CPU-only Linux, exposes `torch_dtype` (`fp32` on mps, `fp16` on cuda-cuda-rocm). Tested with 3 unit cases.

### Phase E — `ingest.py`
🔍 exit: unit tests for sha256 (known vector), mime detection (PDF magic bytes), `pypdf` metadata extraction (title, author, created, page_count). Returns `DocumentInput` dataclass.

### Phase F — `ocr.py` (EasyOCR + Tesseract fallback)
🔍 exit: EasyOCR runs on a fixture page successfully; confidence per word; command-line flag to switch to Tesseract; per-page timeout. Auto-fallback when EasyOCR raises. Mocks deep in unit tests.

> **Bbox shape** — EasyOCR emits 4-point polygons (`[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`); the UIR schema wants 4-int rectangles `[x1, y1, x2, y2]`. Add a `polygon_to_bbox()` normalization step in `ocr.py` so downstream consumers (LayoutLMv3, UIR chunks) see a consistent shape. `tables.py` reuses the same helper.

### Phase G — `layout.py` (LayoutLMv3)
🔍 exit: loads model once (cached), runs inference per page, returns `LayoutRegion[]` with labels from set `{heading, paragraph, table, list, figure, caption, header, footer}`. MPS auto-falls-back to CPU on `NotImplementedError`.

> **Layout-branch choice** — `layout.py` uses the **text+bbox-only branch** by setting `LayoutLMv3Config.visual_embed=False` (then `from_pretrained(model_id, config=...)`). Rationale (validated empirically by Phase A.5 spike 2026-07-07): trim model size and per-page latency to stay within the MVP's `<10s/doc` budget. **The historical worry about Detectron2 conflict on macOS MPS / AMD ROCm does NOT apply in `transformers 5.x`** — visual-branch submodules are plain `nn.Conv2d` patch embeddings, not Detectron2 (verified by reading the top of `transformers/models/layoutlmv3/modeling_layoutlmv3.py`). Path A (multimodal, `visual_embed=True`) is therefore also viable but adds ~250 MB of weights and ~30–50% per-page slowdown for ~2 pp accuracy gain on text-heavy PDFs. Default to Path B; expose a `LAYOUTLMV3_USE_VISUAL` env flag for the A↔B switch.

> **Coordinate normalization** — LayoutLMv3 expects bboxes on a `0–1000` normalized scale (the document-image convention). `layout.py` normalizes raw pixel bboxes relative to the page's pixel dimensions before model input. **UIR output keeps pixel-coordinate bboxes** for round-trip fidelity from PDF rendering.

### Phase H — `tables.py` (pdfplumber)
🔍 exit: detects tables on a fixture PDF; converts each to markdown with header row preserved; emits `TableNode` chunks. Falls back to "no tables detected" gracefully.

### Phase I — `chunk.py`
🔍 exit: chunking produces 256–512 token chunks with ~10–20% overlap; preserves sentence boundaries where feasible, attaches page + bbox + confidence + modal_features (text quality, layout type + reading_order).

> **Tokenizer alignment** — token counting must use **`AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")`**, NOT `tiktoken` or whitespace heuristics. The chunker's hard ceiling must respect BGE's strict 512-token input limit; chunks above that are recursively halved (preserving overlap). A `tiktoken`/BERT mismatch causes silent embedding overflows or truncation that destroys retrieval signals.

### Phase J — `enrich.py`
🔍 exit: spaCy NER ≥ 1 entity on a fixture with known entities; co-occurrence relationships within chunks. Topics stub returns empty list with a TODO in `embed.py`.

### Phase K — `embed.py` + `weaviate_store.py`
🔍 exit: 384-d chunk embeddings; Weaviate collection `UIRChunks_v1` created/verified on first run; per-chunk upsert with UIR metadata blob. Document-level aggregate (mean pool) computed and stored on `UIRParentDoc_v1` collection.

> **Weaviate ID mapping** — UIR IDs are strings like `chunk_<uuid>` (and `section_`, `doc_`, `entity_`). Weaviate's primary ID field requires a bare UUID with no prefix. We **strip the prefix** to form the Weaviate ID; the **full prefixed UIR id** is stored as a BM25-indexed `uir_id` property so the UIR-to-vec link survives round-trips. Identical convention for the parent doc collection.

### Phase L — `pipeline.py` programmatic orchestrator
🔍 exit: chain A→K works on one fixture PDF; provenance block populated with model name, version, and ISO timestamp; emits `UIRV1` JSON. Serial processing is fine for MVP.

### Phase M — CLI + structured logging
🔍 exit: `argparse` flags work; `pipeline.py` runs on a real PDF; per-doc JSON log written; CLI logs to stdout at the configured level.

### Phase N — Sample PDFs + integration tests (3 fixtures, profile-based)
🔍 exit: **3 fixed-profile fixtures** in `tests/fixtures/sample_pdfs/`:
  - `flat_text.pdf` — single column, plain prose (validates OCR + chunking end-to-end)
  - `dense_table.pdf` — ≥ 1 well-formed table spanning 3+ rows × 3+ columns (validates `tables.py` output + layout's `table` label)
  - `multi_column.pdf` — ≥ 2 column layout (validates layout reading-order + bbox post-processing)

Three matching integration tests (`test_integration_flat_text`, `test_integration_dense_table`, `test_integration_multi_column`), each asserting:
  1. `UIRV1.model_validate_json(...)` succeeds.
  2. `structure.root.children` is non-empty.
  3. ≥ 1 entity.
  4. ≥ 1 chunk has a 384-dim embedding.
  5. Weaviate collection contains the doc ID.
  6. **Profile-specific assertion**: `dense_table` → table node present in UIR; `multi_column` → ≥ 2 distinct `reading_order` values; `flat_text` → first section label = `"paragraph"`.
  7. Each marked `@pytest.mark.slow` so CI excludes by default.

### Phase O — Review, polish, docs
🔍 exit: README quickstart validated end-to-end; UIR schema reference committed; `code-reviewer-minimax-m3` review fixes ARE PR-ready; `pytest` green; lint clean.

---

## 10. CLI surface (planned)

```bash
# Single PDF
python pipeline.py data/input/example.pdf --output-data data/output/

# Directory batch (serial processing; --batch-size controls concurrency later)
python pipeline.py data/input/ --output-data data/output/ --batch-size 5

# Skip Weaviate (just JSON to disk)
python pipeline.py data/input/*.pdf --output-data data/output/ --skip-weaviate

# Dry run (no disk writes, no Weaviate)
python pipeline.py data/input/example.pdf --output-data data/output/ --dry-run

# Verbose
python pipeline.py data/input/example.pdf --output-data data/output/ --log-level DEBUG
```

Exit codes: `0` on success, `1` on validation/ingest failure, `2` on partial failure with retryable errors logged.

---

## 11. Performance — reality vs spec

| Spec target | Reality on M4 Pro | Plan |
|---|---|---|
| >95% text accuracy | EasyOCR ≈ PaddleOCR for clean text | Achievable for clean text PDFs; scan-quality degraded (logged). |
| <15% RAG hallucination | Downstream concern | Out of MVP scope; benchmark in Phase 2. |
| >30% token savings | Semantic chunking vs raw page text | Achievable; measure after Phase I. |
| <10s per doc | Mostly OCR-bound | Achievable for ≤20-page PDFs. Median target ~5s. |
| <2GB memory per doc | EasyOCR peak ≈1.2 GB; LayoutLMv3 ≈700 MB; Weaviate negligible | Achievable. |
| <1% error rate | Tesseract fallback chain | Achievable. |
| 1000 PDFs/day | Single-thread ≈500/day on M-series | **Deferred to Phase 2.** |
| Docker image <500MB | Incompatible with current stack | **Dropped from MVP.** |

M4 Pro cost we accept: LayoutLMv3 per-page inference runs at ~150–300 ms/page on MPS — within budget. EasyOCR per-page ~600–900 ms on M-series CPU — within budget for ≤10-page fixture PDFs.

---

## 12. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LayoutLMv3 multimodal branch footprint exceeds MVP latency budget | Low | Med | Use Path B (`visual_embed=False`) by default; flip to Path A only when accuracy drops materially on real fixtures. |
| LayoutLMv3 on MPS hits missing-kernel error | Med | Med | `device.py` tries MPS → falls back to CPU on `NotImplementedError`; log once & continue. |
| EasyOCR download takes minutes on first run | High | Low | README callout; pre-download in CI; model cached in `~/.EasyOCR/model/`. |
| spaCy `en_core_web_sm` download is slow | Low | Low | Same pattern — README callout + CI pre-download. |
| Weaviate Docker under-resourced | Low | Med | README mandates ≥4 GB RAM in Docker Desktop. |
| pdfplumber misses complex tables | Med | Low | Tables emitted with `confidence` + warning; raw text from OCR still used. |
| EasyOCR slower on Mac CPU than PaddleOCR was | Low | Low | Per-page timeout (10s default); Tesseract fallback. |
| AMD porting friction at Phase 2 | Med | High | **Eliminated by today's EasyOCR swap** — full PyTorch-native stack. |
| UIR schema drift from INSTRUCTIONS.md | Low | High | Pydantic schema is source of truth; JSON Schema exported; example-file conformance test in CI. |

---

## 13. Testing strategy

- **Unit tests** — fake every heavy dep (EasyOCR, LayoutLMv3, sentence-transformers, weaviate-client) using pytest fixtures. Each module's outputs validated against a frozen `expected.json` snapshot.
- **Integration test** — one PDF in fixtures runs the entire pipeline. Asserts:
  1. Output file exists at `{output_dir}/{doc_id}.uir.json`.
  2. JSON validates against `UIRV1` model.
  3. `structure.root.children` is non-empty.
  4. ≥ 1 chunk has a 384-dim embedding attached.
  5. If Weaviate is up, `UIRChunks_v1` collection has the doc ID.
  6. Total runtime on fixture < 30 s.
- **Coverage target** — ≥ 80% on the orchestrator and schema; ≥ 60% on the model wrappers (the heavy-dep integration is one path).
- **CI gate** — unit tests + integration test; smoke test on `docker compose up` for Weaviate.

---

## 14. Open questions (please confirm before Phase A starts)

1. **Sample PDFs** — do we have any company-specific PDFs to use, or should we pull from public-domain sources (arXiv preprint, IRS form, government report)? Recommend public-domain for legal hygiene in the open-source repo.
2. **Confidence thresholds** — spec says OCR >0.85, Layout >0.90. For MVP, propose: keep all chunks in the UIR but surface a per-chunk `warnings: ["low_confidence"]` field. **Drop** is more aggressive but loses retrievability. Recommendation: keep + warn.
3. **Topics field** — leave as `[]` for MVP with a `TODO` comment? Confirm OK.
4. **Weaviate collection naming** — `UIRChunks_v1` and `UIRParentDoc_v1`? Or do you have a different naming convention?
5. **Provenance `version` strings** — I'm using the spec's `"1.2.0"` for LayoutLMv3 extraction and `"1.0"` for normalization. Confirm OK to hardcode these in `uir_schema.py` for MVP.

---

## 15. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-07-07 | Scope = MVP end-to-end on 1 PDF | User-confirmed in planning conversation. |
| 2026-07-07 | Local Weaviate via Docker Desktop (no S3, no OpenAI key) | Infrastructure available; constraint from user. |
| 2026-07-07 | AMD cloud = Phase 2 target; Linux+NVIDIA laptop = optional CUDA smoke-test, not primary dev loop | User-confirmed. |
| 2026-07-07 | **PaddleOCR → EasyOCR** (AMD ROCm portability) | Research-confirmed: Paddle lacks production ROCm, PyTorch ecosystem is mature. |
| 2026-07-07 | `camelot` → `pdfplumber` | Cleaner Mac install; equivalent output. |
| 2026-07-07 | `text-embedding-3-small` → `BAAI/bge-small-en-v1.5` | No OpenAI key; PyTorch-native = ROCm-friendly. |
| 2026-07-07 | New `device.py` (`cuda > mps > cpu`) | Single source of truth for hardware; one-line AMD switch. |
| 2026-07-07 | Topics stub returns `[]` (Phase 2 deferred) | Avoid gensim dep; not an MVP acceptance gate. |
| 2026-07-07 | `<500MB` Docker image target **dropped** for MVP | Incompatible with current stack; revisit post-ONNX export. |
| 2026-07-07 | Async multi-doc batching **deferred** | Throughput target is Phase 2. |
| 2026-07-07 | Production `Dockerfile` **deferred** | Local dev containers only for MVP. |
| 2026-07-07 | Confidence policy: keep + warn (not drop) | Spec thresholds satisfied by warnings surfaced in `modal_features.quality` flags. |
| 2026-07-07 | LayoutLMv3: text+bbox-only branch (`visual_embed=False`) | Trade <2 pp accuracy for ~30–50% lower per-page latency and ~250 MB model savings. Detectron2-hostility was a stale 2019-era fear; verified `transformers 5.13` does not pull Detectron2 (visual branch is plain `nn.Conv2d`). |
| 2026-07-07 | Weaviate server 1.26.4 → **1.27.0** | `weaviate-client` 4.22.0 (PyPI-resolved mid-2026) hard-rejects server 1.26.x with `WeaviateStartUpError`. 1.27.0 is the minimum client 4.22 supports. Host port 18080 (not 8080) to dodge dev-machine `:8080` collisions. |
| 2026-07-07 | Phase A.5 spike added before schema/OCR work | LayoutLMv3 is the central risk; surface it BEFORE building around it. |
| 2026-07-07 | Tokenizer = `transformers.AutoTokenizer` for `BAAI/bge-small-en-v1.5`, NOT tiktoken | Embedder expects BERT tokenizer; mismatch → silent overflow / truncation. |
| 2026-07-07 | OCR bbox normalization: `polygon_to_bbox()` step | EasyOCR = 4-point polygons; UIR + LayoutLMv3 want `[x1,y1,x2,y2]` rectangles. |
| 2026-07-07 | UIR IDs typed as `str` regex-validated, not `UUID5` | Native `pydantic.UUID5` rejects the `<prefix>_` segment. |
| 2026-07-07 | LayoutLMv3 input expects `0–1000` normalized coords | Add pixel→normalized step before model input; UIR output keeps pixel coords. |
| 2026-07-07 | Weaviate ID = stripped UUID; full prefixed UIR id stored in `uir_id` BM25 property | Weaviate's primary ID requires a plain UUID. |
| 2026-07-07 | Test fixtures: 3 distinct profiles; one integration test each | One-fixture integration tests have blind spots in layout + table paths. |

---

## 16. References

- Upstream spec: [`INSTRUCTIONS.md`](./INSTRUCTIONS.md) (read-only)
- EasyOCR: <https://github.com/JaidedAI/EasyOCR> (PyTorch-native, runs on MPS and ROCm)
- LayoutLMv3: <https://huggingface.co/microsoft/layoutlmv3-base>
- BGE-small: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- pdfplumber: <https://github.com/jsvine/pdfplumber>
- Weaviate: <https://weaviate.io/developers/weaviate>
- AMD ROCm PyTorch wheels: `rocm/pytorch:latest` Docker image (production-ready as of 2026)

