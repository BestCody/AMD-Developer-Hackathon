"""tests/test_umr.py -- Universal Markdown Representation contract.

These tests pin the UMR renderer's behavior so a regression in
``src/uir_pipeline/umr.py`` surfaces here, not at the web UI / agent
consumption layer. Each test isolates one design goal from PLAN §17:

    * Empty / abnormal inputs render a safe header instead of crashing.
    * Section / chunk hierarchy maps cleanly to ``##`` headings and
      blockquote anchors with page + bbox + token count + kind.
    * Tabular data (markdown tables inside chunk text) renders verbatim.
    * Figure captions surface the ``caption`` kind badge so an agent can
      distinguish raw figure regions from captioned ones.
    * Intent-filter subset view collapses non-matching chunks AND
      drops empty headings (no dead TOC anchors going forward).
    * Determinism: byte-identical output across re-runs of the same
      input dict.
    * Missing ``modal_features`` is graceful (``unknown`` kind / ``?``
      page) so a partial UIR doesn't take down the renderer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uir_pipeline.umr import build_umr


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _section(
    sid: str,
    title: str,
    children: list[dict] | None = None,
    path: str | None = None,
) -> dict:
    """Build a minimal section dict for unit tests."""
    sec: dict = {
        "id": sid,
        "type": "section",
        "title": title,
        "page": 1,
        "children": list(children or []),
    }
    # The orchestrator stores the section path on EACH child chunk's
    # ``modal_features.section.path``; section nodes themselves don't
    # carry a path field. Tests' chunks below set it directly.
    return sec


def _chunk(
    cid: str,
    text: str,
    *,
    page: int = 1,
    bbox: list[int] | None = None,
    tokens: int = 100,
    kind: str | None = None,
    section_path: str | None = None,
) -> dict:
    """Build a minimal chunk dict for unit tests."""
    chunk: dict = {
        "id": cid,
        "type": "chunk",
        "text": text,
        "page": page,
        "token_count": tokens,
        "bounding_box": bbox if bbox is not None else [12, 80, 990, 210],
        "confidence": 0.95,
        "modal_features": {},
    }
    if section_path:
        chunk["modal_features"]["section"] = {"path": section_path}
    if kind:
        chunk["modal_features"]["intent"] = {"region_kind": kind}
    return chunk


def _uir(
    *,
    title: str = "Sample Report",
    author: str | None = "Alice Smith",
    pages: int = 7,
    domain: str | None = None,
    language: str = "en",
    children: list[dict] | None = None,
) -> dict:
    """Build a minimal UIR document dict."""
    return {
        "uiR_version": "1.0",
        "id": "doc_test",
        "modal_type": "document",
        "metadata": {
            "title": title,
            "author": author,
            "page_count": pages,
            "language": language,
            "domain": domain,
        },
        "structure": {
            "type": "hierarchical",
            "root": {
                "id": "doc_test",
                "type": "document",
                "title": title,
                "page": 1,
                "children": list(children or []),
            },
        },
        "semantics": {
            "entities": [],
            "relationships": [],
            "topics": [],
        },
        "provenance": {
            "extraction": {"model": "test", "version": "1.0",
                           "timestamp": "2026-01-01T00:00:00Z"},
            "normalization": {"version": "1.0",
                              "timestamp": "2026-01-01T00:00:00Z"},
        },
    }


# ----------------------------------------------------------------------------
# Caseness tests
# ----------------------------------------------------------------------------

def test_empty_document_renders_safe_header():
    """An empty UIR (no chunks, no sections) emits H1 + a one-line marker."""
    out = build_umr(_uir(title="Empty Doc", children=[]))
    assert out.startswith("# Empty Doc\n")
    assert "No content extracted or matched." in out
    assert "Contents:" not in out  # No sections => no TOC
    # Always ends with a trailing newline (byte-stable contract).
    assert out.endswith("\n")


def test_root_level_chunks_render_before_any_section():
    """Root-level (unsection'd) chunks emit BEFORE any ``## Section`` heading."""
    out = build_umr(_uir(children=[
        _chunk("c_pre", "Preamble text."),
    ]))
    # Single chunk, only one anchor + body; no section heading because
    # the chunk is not nested in a section node.
    assert "## " not in out
    assert "> **[c_pre" in out
    assert "Preamble text." in out


def test_multi_section_hierarchy_sequential_h2():
    """Nested sections emit sequential ``## `` with chunks under each header."""
    out = build_umr(_uir(children=[
        _section("sec_a", "Abstract", [
            _chunk("c_a", "We measured lithium abundance.",
                   kind="paragraph", section_path="Abstract"),
        ], path="Abstract"),
        _section("sec_b", "Methodology", [
            _chunk("c_b", "We used Keck telescope.",
                   kind="paragraph", section_path="Methodology"),
        ], path="Methodology"),
    ]))
    # Two section headings, in source order.
    assert "## Abstract\n" in out
    assert "## Methodology\n" in out
    # Chunk anchor fires once per chunk.
    assert "> **[c_a" in out
    assert "> **[c_b" in out
    # The methods chunk follows its parent ``## Methodology`` heading.
    methodology_index = out.index("## Methodology")
    chunk_b_index = out.index("> **[c_b")
    assert methodology_index < chunk_b_index


def test_intent_filter_subset_only_keeps_matched_chunks():
    """An intent_filter dict narrows the rendered subtree to matched ids."""
    out = build_umr(_uir(children=[
        _section("sec_a", "Abstract", [
            _chunk("c_a", "We measured lithium abundance.",
                   kind="paragraph", section_path="Abstract"),
            _chunk("c_b", "We used Keck telescope.",
                   kind="paragraph", section_path="Abstract"),
        ]),
    ]),
        intent_filter={
            "intent": "lithium abundance",
            "keywords": ["lithium"],
            "matches": [
                {"chunk_id": "c_a", "score": 0.83, "score_kind": "cosine"},
                {"chunk_id": "c_b", "score": 0.21, "score_kind": "cosine"},
            ],
        },
    )
    # Filtered view banner must appear.
    assert "Filtered view" in out
    assert 'lithium abundance' in out
    # Both matched chunks render.
    assert "> **[c_a" in out
    assert "> **[c_b" in out


def test_intent_filter_drops_empty_section_heading():
    """A section whose chunks DO NOT match disappears entirely (no header leak)."""
    out = build_umr(_uir(children=[
        _section("sec_a", "Irrelevant Section", [
            _chunk("c_drop", "Unrelated text.",
                   kind="paragraph"),
        ]),
        _section("sec_b", "Relevant Section", [
            _chunk("c_hit", "Lithium abundance is high.",
                   kind="paragraph"),
        ]),
    ]),
        intent_filter={
            "intent": "lithium",
            "keywords": ["lithium"],
            "matches": [{"chunk_id": "c_hit", "score": 0.9,
                          "score_kind": "cosine"}],
        },
    )
    assert "## Relevant Section" in out
    assert "## Irrelevant Section" not in out
    assert "## " not in [line for line in out.split("\n")
                          if line.startswith("##") and "Relevant" not in line]


def test_table_chunk_preserves_markdown_table():
    """A chunk whose text already contains markdown table pipes renders verbatim."""
    md_table = (
        "| column_a | column_b |\n"
        "|----------|----------|\n"
        "| 1.0      | 2.0      |\n"
    )
    out = build_umr(_uir(children=[
        _section("sec_t", "Results", [
            _chunk("c_t", md_table, kind="table"),
        ]),
    ]))
    assert "| column_a | column_b |" in out
    assert "> **[c_t" in out
    # Region-kind badge in the anchor.
    assert "table" in out


def test_caption_chunk_kind_badge():
    """A caption chunk renders the ``caption`` kind badge so agents can route."""
    out = build_umr(_uir(children=[
        _section("sec_fig", "Results", [
            _chunk("c_cap", "Figure 1: lithium abundance by mass.",
                   kind="caption"),
        ]),
    ]))
    assert "> **[c_cap" in out
    assert "caption" in out


def test_deterministic_byte_identical_rerun():
    """Two consecutive calls with the same input produce byte-identical output."""
    uir = _uir(children=[
        _section("sec_a", "Methodology", [
            _chunk("c_a", "Subsection A.1." * 40,
                   kind="paragraph", section_path="Methodology"),
            _chunk("c_b", "Subsection A.2." * 40,
                   kind="paragraph", section_path="Methodology"),
        ]),
    ])
    out1 = build_umr(uir)
    out2 = build_umr(uir)
    assert out1 == out2


# ----------------------------------------------------------------------------
# Defensive coverage
# ----------------------------------------------------------------------------

def test_missing_modal_features_falls_back_gracefully():
    """A chunk with no modal_features renders an ``unknown`` kind badge."""
    chunk = _chunk("c_x", "Body text.")
    chunk.pop("modal_features", None)
    out = build_umr(_uir(children=[
        _section("sec_x", "Results", [chunk]),
    ]))
    assert "> **[c_x" in out
    assert "unknown" in out


def test_top_level_metadata_eyebrow_only_when_present():
    """Eyebrow line is suppressed when no relevant metadata fields are set."""
    out = build_umr(_uir(title="Bare Doc",
                          author=None, pages=0,
                          domain=None, language=""))
    # Empty author + 0 pages + empty language => no eyebrow rendered.
    eyebrow_lines = [line for line in out.split("\n")
                     if line.startswith("*") and " · " in line]
    assert eyebrow_lines == []


def test_page_zero_chunk_does_not_break_renderer():
    """A chunk with ``page=0`` (degenerate UIR) still renders without errors."""
    chunk = _chunk("c_zero", "Some body text.", page=0)
    out = build_umr(_uir(children=[
        _section("sec_zero", "Bad", [chunk]),
    ]))
    assert "> **[c_zero" in out
    assert "page 0" in out  # 0 is still valid display


def test_unknown_node_type_emits_diagnostic_comment():
    """An unknown ``type`` field emits an ``<!-- unknown --->`` HTML comment."""
    weird = {"id": "weird_1", "type": "mystery_node",
             "text": "Should be skipped."}
    out = build_umr(_uir(children=[
        _section("sec_a", "Results", [weird]),
    ]))
    assert "<!-- unknown node type: mystery_node -->" in out


def test_end_to_end_pipeline_emit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sanity: invoking the orchestrator writes a non-empty .umr.md file.

    Lightweight -- patches the heavy-dep imports so the test runs without
    BGE / spaCy / pdfplumber. We don't go through Tier 3 caption stage.
    """
    import importlib.machinery
    import sys
    import types

    # Stub out the heavy deps that the orchestrator eagerly imports.
    #
    # `__spec__` is load-bearing, not decoration: docling's import chain calls
    # `importlib.util.find_spec("spacy")`, which raises
    # `ValueError: spacy.__spec__ is None` on a bare ModuleType. That surfaced
    # as `DoclingUnavailable: docling package not importable`, so this test
    # only passed when an earlier test had already imported the real spacy.
    #
    # `monkeypatch.setitem`, not a raw assignment: the stub used to persist in
    # `sys.modules` for every later test in the session.
    spacy_stub = types.ModuleType("spacy")
    spacy_stub.__spec__ = importlib.machinery.ModuleSpec("spacy", loader=None)

    class _FakeNLP:
        def __call__(self, _txt):
            from types import SimpleNamespace
            return SimpleNamespace(ents=[])

    spacy_stub.load = lambda _id: _FakeNLP()
    monkeypatch.setitem(sys.modules, "spacy", spacy_stub)
    # Use the test fixtures' pdfplumber shim from conftest.
    pdf = Path("tests/fixtures/sample_pdfs/flat_text.pdf")
    if not pdf.is_file():
        pytest.skip("flat_text.pdf fixture missing; end-to-end optional")
    from uir_pipeline.pipeline import run
    out_dir = tmp_path / "out"
    result = run(
        pdf, output_dir=out_dir,
        skip_weaviate=True, with_embeddings=False,
        page_numbers=[1], include_semantics=False,
    )
    # UMR path is on the result.
    assert result.umr_path.is_file()
    txt = result.umr_path.read_text()
    assert txt.startswith("# ")
    assert len(txt) > 200  # not a sentinel
    # Default-OFF semantics: JSON does NOT carry relationships.
    uir = json.loads(result.out_path.read_text())
    assert uir["semantics"]["entities"] == []
    assert uir["semantics"]["relationships"] == []


# ---------------------------------------------------------------------------
# Empty-section rollback under an intent filter
# ---------------------------------------------------------------------------

def _render(node, intent_filter):
    from uir_pipeline import umr as umr_mod

    lines: list[str] = []
    n = umr_mod._render_children_recursive(
        node, lines, intent_filter=intent_filter, recursion_depth=0
    )
    return n, lines


def test_filtered_out_section_leaves_no_dangling_heading():
    """A section with no surviving chunks must not render its heading.

    The rollback used to pop trailing blanks and then one `## ` line. An
    `<!-- unknown node type -->` comment inside the section stopped the blank
    pop, the heading check failed, and `## Empty Section` survived with no
    body -- a heading the agent would try to read under.
    """
    node = {"type": "root", "children": [
        {"type": "section", "id": "sec_1", "title": "Empty Section", "children": [
            {"type": "chunk", "id": "chunk_zzz", "text": "filtered out", "page": 1},
            {"type": "mystery"},
        ]},
    ]}
    rendered, lines = _render(node, {"matches": [{"chunk_id": "chunk_keep"}]})
    assert rendered == 0
    assert lines == [], f"section left residue: {lines}"


def test_section_with_a_surviving_chunk_keeps_its_heading():
    node = {"type": "root", "children": [
        {"type": "section", "id": "sec_1", "title": "Kept Section", "children": [
            {"type": "chunk", "id": "chunk_keep", "text": "survives", "page": 1},
        ]},
    ]}
    rendered, lines = _render(node, {"matches": [{"chunk_id": "chunk_keep"}]})
    assert rendered == 1
    assert any(line.startswith("## ") for line in lines)
    assert any("survives" in line for line in lines)


def test_rollback_does_not_eat_a_preceding_sibling_section():
    """Truncation must remove only what the empty section appended."""
    node = {"type": "root", "children": [
        {"type": "section", "id": "sec_1", "title": "Kept", "children": [
            {"type": "chunk", "id": "chunk_keep", "text": "survives", "page": 1},
        ]},
        {"type": "section", "id": "sec_2", "title": "Dropped", "children": [
            {"type": "chunk", "id": "chunk_gone", "text": "filtered", "page": 2},
        ]},
    ]}
    rendered, lines = _render(node, {"matches": [{"chunk_id": "chunk_keep"}]})
    assert rendered == 1
    text = "\n".join(lines)
    assert "Kept" in text and "survives" in text
    assert "Dropped" not in text


def test_no_filter_renders_every_section():
    """intent_filter=None is the full-document view; nothing is rolled back."""
    node = {"type": "root", "children": [
        {"type": "section", "id": "sec_1", "title": "Anything", "children": [
            {"type": "chunk", "id": "chunk_a", "text": "body", "page": 1},
        ]},
    ]}
    rendered, lines = _render(node, None)
    assert rendered == 1
    assert any(line.startswith("## ") for line in lines)


def test_nested_empty_sections_collapse_entirely():
    node = {"type": "root", "children": [
        {"type": "section", "id": "sec_1", "title": "Outer", "children": [
            {"type": "section", "id": "sec_2", "title": "Inner", "children": [
                {"type": "chunk", "id": "chunk_gone", "text": "filtered", "page": 1},
            ]},
        ]},
    ]}
    rendered, lines = _render(node, {"matches": [{"chunk_id": "chunk_keep"}]})
    assert rendered == 0
    assert lines == [], f"nested sections left residue: {lines}"
