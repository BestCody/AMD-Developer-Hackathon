**Phase 1 Pipeline Prompt for AI Coding Agent**

---

**🎯 Objective**
Build a **production-ready Python pipeline** that converts **PDF documents** into **Universal Intermediate Representation (UIR)** format. This is Phase 1 of a multimodal unification system.

---

**📦 Scope**
- **Input**: Raw PDF files (text-heavy, forms, tables, slides)
- **Output**: UIR JSON (schema defined below)
- **Modalities**: **PDFs only** (Phase 1)
- **Volume**: Handle 1,000+ documents/day
- **Latency**: <10s per document

---

**⚙️ Technical Requirements**

**Core Stack**
- **Language**: Python 3.10+
- **OCR**: PaddleOCR (primary) + Tesseract (fallback)
- **Layout Understanding**: LayoutLMv3 (`microsoft/layoutlmv3-base`)
- **Table Extraction**: `camelot` or `pdfplumber`
- **Chunking**: Semantic (growing window, 256-512 tokens)
- **Embeddings**: `text-embedding-3-small` (OpenAI) or `bge-small`
- **Storage**: JSON files + Weaviate vector index
- **Dependencies**: `pypdf`, `paddleocr`, `transformers`, `sentence-transformers`, `weaviate-client`

**UIR Schema (Strict)**
```json
{
  "uiR_version": "1.0",
  "id": "uuid5",
  "modal_type": "document",
  "source": {
    "uri": "s3://bucket/file.pdf",
    "format": "PDF",
    "mime_type": "application/pdf",
    "size_bytes": 2450000,
    "checksum": "sha256:...",
    "timestamp": "2026-07-07T00:00:00Z"
  },
  "metadata": {
    "title": "string",
    "author": "string|null",
    "created": "ISO8601|null",
    "modified": "ISO8601|null",
    "page_count": 10,
    "language": "en",
    "domain": "infer_or_null"
  },
  "structure": {
    "type": "hierarchical",
    "root": {
      "id": "doc_<uuid>",
      "type": "document",
      "title": "string",
      "children": [
        {
          "id": "section_<uuid>",
          "type": "section|table|figure|list",
          "title": "string|null",
          "page": 1,
          "bounding_box": [x1, y1, x2, y2],
          "children": [
            {
              "id": "chunk_<uuid>",
              "type": "chunk",
              "text": "string",
              "token_count": 256,
              "page": 1,
              "bounding_box": [x1, y1, x2, y2],
              "confidence": 0.95,
              "modal_features": {
                "text": {"quality": 0.98},
                "layout": {"type": "paragraph", "reading_order": 1}
              }
            }
          ]
        }
      ]
    }
  },
  "semantics": {
    "entities": [{"text": "revenue", "type": "financial_metric", "confidence": 0.92}],
    "relationships": [{"from": "entity_id", "to": "entity_id", "type": "string", "confidence": 0.88}],
    "topics": ["string"]
  },
  "provenance": {
    "extraction": {"model": "LayoutLMv3", "version": "1.2.0", "timestamp": "ISO8601"},
    "normalization": {"version": "1.0", "timestamp": "ISO8601"}
  }
}
```

---
---
**📥 Input/Output Contract**

| **Input** | **Output** |
|-----------|------------|
| `/input/*.pdf` | `/output/{doc_id}.uir.json` |
| S3 bucket path | Weaviate vector index |
| Raw binary | UIR JSON + embeddings |

---
---
**🏗️ Pipeline Architecture**

```
PDF Input
│
├── 1. Ingestion
│   ├── Validate file (size, type, checksum)
│   ├── Extract metadata (title, author, pages)
│   └── Store raw in `/input/{doc_id}.pdf`
│
├── 2. Text Extraction
│   ├── PaddleOCR → raw text + layout
│   ├── Fallback: Tesseract if OCR fails
│   └── Output: `{"text": "...", "pages": [{"text": "...", "bbox": [...]}]`
│
├── 3. Layout Understanding
│   ├── LayoutLMv3 → semantic structure
│   ├── Classify: headings, paragraphs, tables, lists
│   └── Output: `{"structure": [...], "entities": [...]}` (spaCy NER)
│
├── 4. Table Extraction
│   ├── camelot → structured tables
│   ├── Convert to markdown/HTML
│   └── Embed as `type: "table"` in UIR
│
├── 5. Chunking
│   ├── Growing window (256-512 tokens)
│   ├── Preserve: sentences, lists, tables, code blocks
│   └── Output: `structure.children[].children[]`
│
├── 6. Semantic Enrichment
│   ├── Extract entities (spaCy)
│   ├── Infer relationships (co-occurrence)
│   ├── Assign topics (LDA or embedding clustering)
│   └── Output: `semantics.*`
│
├── 7. Embedding
│   ├── Chunk-level: `text-embedding-3-small`
│   ├── Document-level: aggregate
│   └── Store in Weaviate with UIR metadata
│
└── 8. Output
    ├── Save UIR JSON to `/output/{doc_id}.uir.json`
    └── Log to `/logs/{doc_id}.log`
```

---
---
**📋 Implementation Tasks**

1. **Setup**
   - Dockerize pipeline (`Dockerfile` + `docker-compose.yml`)
   - Environment vars: `OPENAI_API_KEY`, `WEAVIATE_URL`, `S3_BUCKET`
   - Dependency lock: `requirements.txt`

2. **Core Modules**
   - `ingest.py` – File validation + metadata extraction
   - `ocr.py` – PaddleOCR wrapper + Tesseract fallback
   - `layout.py` – LayoutLMv3 inference + structure parsing
   - `tables.py` – Table detection + extraction
   - `chunk.py` – Semantic chunking (growing window)
   - `enrich.py` – Entity/relationship extraction
   - `embed.py` – Weaviate client + embedding logic
   - `pipeline.py` – Orchestrator

3. **CLI**
   ```bash
   # Process single file
   python pipeline.py /input/doc.pdf --output /output/

   # Process directory
   python pipeline.py /input/ --output /output/ --batch-size 10

   # Process S3 bucket
   python pipeline.py s3://bucket/ --output /output/
   ```

4. **Validation**
   - UIR schema validation (Pydantic)
   - Confidence thresholds (OCR >0.85, LayoutLM >0.90)
   - Fallback to raw text if structured parsing fails

---
---
**⚡ Performance Targets**
| Metric | Target |
|--------|--------|
| Accuracy (text extraction) | >95% |
| Hallucination rate (RAG) | <15% |
| Token savings | >30% |
| Processing time | <10s/doc |
| Memory usage | <2GB/doc |
| Error rate | <1% |

---
---
**📦 Deliverables**
1. **Code**: Modular, tested, Dockerized
2. **Tests**: Unit + integration (pytest)
3. **Docs**: README, API spec, UIR schema reference
4. **Benchmark**: 100 sample PDFs + metrics
5. **Deployment**: `docker-compose.yml` for local dev

---
---
**🚀 Success Criteria**
✅ Pipeline processes 1,000 PDFs without errors
✅ UIR output validates against schema
✅ <15% hallucination rate in downstream RAG
✅ <10s average processing time
✅ Docker container <500MB

---
---
**💡 Optimization Notes**
- Cache OCR results (Redis) for repeated documents
- Batch LayoutLMv3 inference for multi-page docs
- Parallelize: OCR + layout analysis
- Retry failed extractions with alternative methods
- Log warnings for low-confidence chunks

---
---
**📌 Constraints**
- **No**: Hardcoded paths, API keys, or credentials
- **No**: External service calls without retries/timeouts
- **Yes**: Async I/O for network-bound ops
- **Yes**: Type hints + Pydantic models
- **Yes**: Logging (DEBUG, INFO, WARNING, ERROR)

---
**Start Command**
```bash
# Clone, setup, run
git clone <repo>
cd phase1-pipeline
docker-compose up --build
python pipeline.py /data/input/ --output /data/output/
```
