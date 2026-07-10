"""umr -- Universal Markdown Representation.

Renders a UIR v1.0 document into a clean Markdown blob sized for an LLM
agent's context window. The UIR JSON's ``semantics`` block (411+
entities + 736+ relationships on a real arXiv doc) makes the JSON super
long and useless for agents that just want the document's content + a
citation anchor per chunk. UMR emits ONLY what an agent needs:

    1. Title + metadata eyebrow (one line).
    2. Section headings rendered from ``modal_features.section.path`` OR
       ``StructureNode.title``.
    3. Per-chunk citation anchor + the chunk text inline (no modal_features
       vector blob; no entities; no relationships; no topics).

When an intent_filter view is requested, UMR composes the filtered
subtree so agents see only the chunks that matched their query instead
of the full document + a manual filter step.

The renderer is intentionally pure-Python / zero-dep so the web UI's
``/api/umr/<job_id>`` endpoint can stream it without spinning up a
markdown library, and so a CLI ``pipeline.py`` invocation can produce a
shell-pipelineable stream.

Design goals (work in progress -- iteration-1, see PLAN.md §17):

    * Median-sized arXiv doc emits ~5-30 KB of markdown.
    * Deterministic: identical UIR in -> byte-identical UMR out (used
      for caching + diffing when the same PDF is re-uploaded).
    * Robust: missing fields render as ``?`` / ``unknown`` rather than
      crashing so a truncated UIR still produces a usable preview.
    * Intent-filter-aware: pass an ``intent_filter`` dict (with
      ``matches``) to render the narrowed subtree -- saves downstream
      token cost.

Not in scope (yet):

    * Markdown -> HTML rendering. The web UI does that client-side
      with a vendored ``marked.min.js`` follow-up (not this iteration).
    * Schema validation. UMR operates on parsed dicts so a malformed
      UIR raises during pydantic validation upstream and never reaches
      here.
"""
from __future__ import annotations

from typing import Any

# Module-level constants carried as ``Final`` for stable test-snapshot
# output. ``DEFAULT_AUTHOR_LITERAL`` keeps the eyebrow gracefully empty
# when the UIR's metadata.author is missing (most PDFs don't ship one).
_MAX_TABLE_OF_CONTENTS_ENTRIES: int = 50
_MAX_BBOX_DISPLAY: tuple[int, int, int, int] = (0, 0, 999, 999)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _coerce_int(v: Any, default: int = 0) -> int:
    """Int-coerce ``v`` defensively; return ``default`` on null/bad input."""
    try:
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_str(v: Any, default: str = "") -> str:
    """String-coerce ``v`` defensively; return ``default`` when None."""
    return default if v is None else str(v)


def _bbox_str(bbox: Any) -> str:
    """Render a UIR bbox as ``(x1,y1,x2,y2)`` with defensive coercion."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "(bbox ?)"
    try:
        parts = tuple(int(b) for b in bbox)
    except (TypeError, ValueError):
        return "(bbox ?)"
    # Clamp display to canvas so a bad upstream value never blows the
    # line width unexpectedly. Tests pin the literal at "(0,0,999,999)"
    # for the worst-case fallback.
    parts = tuple(max(0, min(999, p)) for p in parts)
    return f"({parts[0]},{parts[1]},{parts[2]},{parts[3]})"


def _kind_label(chunk: dict[str, Any]) -> str:
    """Return a one-word kind badge for ``chunk`` from modal_features.

    ``modal_features.intent.region_kind`` is the canonical source -- the
    orchestrator already canonicalizes against ``LayoutLabel`` (e.g.
    ``caption`` / ``table`` / ``paragraph``). Unknown / missing falls
    back to ``unknown`` so the agent can still cite by page+bbox+id.
    """
    mf = chunk.get("modal_features") or {}
    intent = mf.get("intent") or {}
    kind = intent.get("region_kind")
    if not kind:
        return "unknown"
    return str(kind).strip().lower()


def _section_path(chunk: dict[str, Any]) -> str:
    """Return ``modal_features.section.path`` (empty string when absent)."""
    mf = chunk.get("modal_features") or {}
    sec = mf.get("section") or {}
    return _coerce_str(sec.get("path"), default="").strip()


def _section_path_for_node(node: dict[str, Any]) -> str:
    """A section node's path: prefer ``title`` (already chosen by orchestrator).

    Falls back to the chunk-derived path if a section is empty (rare --
    only when the orchestrator set title="" without a path).
    """
    title = _coerce_str(node.get("title"), default="").strip().rstrip(".")
    return title


def _is_filtered_match(
    chunk_id: str,
    *,
    intent_filter: dict[str, Any] | None,
) -> bool:
    """Test if ``chunk_id`` is in the intent_filter's match set, including
    synthetic no-match fallback. ``intent_filter=None`` means "include all
    chunks" (full-document view)."""
    if not intent_filter:
        return True
    matches = intent_filter.get("matches") or []
    if not matches:
        return True
    for m in matches:
        if m.get("chunk_id") == chunk_id:
            return True
    return False


def _chunk_neighbours(chunk: dict[str, Any]) -> list[str]:
    """Return the chunk ids of the immediate preceding/following siblings.

    Mirrors the UIR convention ``modal_features.preceding_chunk_id`` /
    ``modal_features.following_chunk_id`` (set in pipeline.Stage 9).
    """
    mf = chunk.get("modal_features") or {}
    out: list[str] = []
    pre = (mf.get("preceding_chunk_id") or {}).get("chunk_id")
    post = (mf.get("following_chunk_id") or {}).get("chunk_id")
    if pre:
        out.append(str(pre))
    if post:
        out.append(str(post))
    return out


def _sanitize_text(text: str) -> str:
    """Trim + collapse so chunk text renders cleanly inside markdown body.

    We never strip backticks / hashes -- that would mangle code blocks.
    We DO trim trailing whitespace per line so a Docling artifact
    (``"\\n \\n"``) doesn't blow the markdown source into a visible blank
    line on render. Tests pin the \"trailing whitespace stripped\" rule.
    """
    if not text:
        return ""
    out_lines = [ln.rstrip() for ln in text.splitlines()]
    # Trim trailing blank lines so the markdown source ends each chunk
    # cleanly. Leading blank lines are kept -- they're rare but valid
    # (e.g. an author-note line offset).
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    return "\n".join(out_lines).strip()


# ----------------------------------------------------------------------------
# Table of contents (deterministic; depth-first walk)
# ----------------------------------------------------------------------------

def _walk_sections(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat list of ``section`` nodes in source order (depth-first)."""
    out: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = list(root.get("children") or [])
    while stack:
        n = stack.pop(0)
        if n.get("type") == "section":
            out.append(n)
            stack = list(n.get("children") or []) + stack
      # chunks and unknown types are skipped at this layer
    return out


# ----------------------------------------------------------------------------
# Core renderer
# ----------------------------------------------------------------------------

def build_umr(
    uir_doc: dict[str, Any],
    *,
    intent_filter: dict[str, Any] | None = None,
) -> str:
    """Render ``uir_doc`` (parsed JSON dict) as Universal Markdown.

    ``intent_filter`` -- when present, carries the same shape as
    :func:`uir_pipeline.intent_filter.filter_uirstream_by_intent`'s
    return value (with ``matches[].chunk_id``). UMR composes a
    *subset* view: only chunks whose id appears in ``matches`` are
    rendered, with their enclosing sections kept (so an agent retains
    the section's header for citation context).

    Empty / abnormal inputs render a safe header + a single-line
    "No content" marker instead of raising -- the rendered output is
    always a markdown document an agent can read.
    """
    meta = (uir_doc or {}).get("metadata") or {}
    structure = (uir_doc or {}).get("structure") or {}
    root = structure.get("root") or {}

    lines: list[str] = []
    title = _coerce_str(meta.get("title"), default="Untitled document").strip() or \
        "Untitled document"
    lines.append(f"# {title}")
    lines.append("")

    # Metadata eyebrow: only emit when at least one field is set. Avoids
    # a lonely *| empty* line for documents with no author / pages / etc.
    eyebrow_parts: list[str] = []
    author = _coerce_str(meta.get("author"), default="").strip()
    if author:
        eyebrow_parts.append(f"author: {author}")
    pages = _coerce_int(meta.get("page_count"), default=0)
    if pages > 0:
        eyebrow_parts.append(f"pages: {pages}")
    domain = _coerce_str(meta.get("domain"), default="").strip()
    if domain:
        eyebrow_parts.append(f"domain: {domain}")
    lang = _coerce_str(meta.get("language"), default="").strip()
    if lang:
        eyebrow_parts.append(f"language: {lang}")
    if eyebrow_parts:
        lines.append("*" + " · ".join(eyebrow_parts) + "*")
        lines.append("")

    # Filtered-view banner: a single italic line so an agent reading the
    # UMR knows the document is narrowed.
    # Test `intent_filter` directly in the guard (not a precomputed bool) so
    # the body is narrowed to a non-None dict. `is_filtered` is still needed
    # below, at the _render_children_recursive call.
    is_filtered = bool(intent_filter and intent_filter.get("matches"))
    if intent_filter and is_filtered:
        intent_label = _coerce_str(
            intent_filter.get("intent"), default="(intent)",
        ).strip() or "(intent)"
        kw = intent_filter.get("keywords") or []
        kw_part = (
            f"; keywords: {', '.join(str(k) for k in kw)}"
            if kw else ""
        )
        lines.append(
            f"*Filtered view: matches for intent \"{intent_label}\"{kw_part}.*"
        )
        lines.append("")

    # Table of contents: list each section's heading in source order. We
    # cap the TOC depth (top ``_MAX_TABLE_OF_CONTENTS_ENTRIES``) so a
    # pathological UIR with 1000 sections doesn't emit a 200 KB TOC.
    sections = _walk_sections(root)
    if sections:
        toc_lines: list[str] = []
        for s in sections:
            path = _section_path_for_node(s)
            heading_label = _heading_render(path, s)
            toc_lines.append(f"- {heading_label}")
        if toc_lines:
            lines.append("**Contents:**")
            lines.append("")
            for tl in toc_lines[:_MAX_TABLE_OF_CONTENTS_ENTRIES]:
                lines.append(tl)
            if len(toc_lines) > _MAX_TABLE_OF_CONTENTS_ENTRIES:
                lines.append(
                    f"- …({len(toc_lines) - _MAX_TABLE_OF_CONTENTS_ENTRIES} more sections)"
                )
            lines.append("---")
            lines.append("")

    # Body: walk root.children, render each as either a section heading
    # or a root-level chunk. Root-level chunks render immediately, even
    # in filtered mode (matches-first pass below).
    body_chunks_rendered = _render_root_children(
        root, lines,
        intent_filter=intent_filter if is_filtered else None,
        recursion_depth=0,
    )

    if not body_chunks_rendered:
        # Fall-through: document has no chunks that survive the filter
        # (or is genuinely empty). Emit a single-line marker so the agent
        # can route correctly.
        lines.append(suffix_marker(intent_filter=intent_filter))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _heading_render(path: str, node: dict[str, Any]) -> str:
    """Compose a section heading string.

    Convention decided by the design review (PLAN §17):
    ``"<path_or_title>"`` (no leading numeric prefix when not present).
    Yields e.g. ``§3.2 Multi-Head Attention`` -> ``3.2 Multi-Head Attention``
    when the path is numeric + matching title; falls through to either
    ``path`` alone, ``title`` alone, or `(unnamed section)` for empty
    cases.
    """
    title = _coerce_str(node.get("title"), default="").strip().rstrip(".")
    if path and title and title != path:
        # Path and title both set -- prefer the explicit title, since
        # the orchestrator chose it. Numeric path is preserved as a
        # scope anchor.
        return f"{path} {title}"
    if title:
        return title
    if path:
        return path
    return "(unnamed section)"


def _render_root_children(
    node: dict[str, Any],
    lines: list[str],
    *,
    intent_filter: dict[str, Any] | None,
    recursion_depth: int,
) -> int:
    """Render ``node.children`` (root or nested section). Return chunk count.

    When ``intent_filter`` is set, only chunks whose id is in the match
    set keep emitting AND sections whose name/title matches a keyword
    keep emitting. Sections with NO surviving chunks drop out entirely
    (TOC entries for filtered-out sections become dead links, which is
    worse than not listing them).
    """
    n_rendered = 0
    for child in (node.get("children") or []):
        ctype = child.get("type")
        if ctype == "chunk":
            if _is_filtered_match(_coerce_str(child.get("id")), intent_filter=intent_filter):
                _render_chunk(child, lines)
                n_rendered += 1
        elif ctype == "section":
            # Render the section only if it (a) is not filtered OR
            # (b) has at least one chunk that survived the filter.
            sub_filtered = (
                intent_filter if intent_filter else None
            )
            pre_lines_len = len(lines)
            heading = f"## {_heading_render(
                _section_path_for_node(child), child
            )}"
            lines.append(heading)
            lines.append("")
            # Recurse; we re-decide heading skip after seeing child output.
            sub_rendered = _render_children_recursive(
                child, lines,
                intent_filter=sub_filtered,
                recursion_depth=recursion_depth + 1,
            )
            if sub_rendered == 0 and intent_filter:
                # No surviving chunks in this section under filter -> drop
                # everything this section appended, so the agent never sees a
                # dangling ``## foo`` with no body.
                #
                # Truncating to the recorded length is exact. Popping trailing
                # blanks and then one ``## `` line was not: when the recursion
                # emitted an ``<!-- unknown node type -->`` comment, the blank
                # pop stopped on it, the heading check failed, and the empty
                # heading survived anyway.
                del lines[pre_lines_len:]
            else:
                n_rendered += sub_rendered
        else:
            # Unknown node type -- pass through JSON-as-block so the
            # agent sees it but can ignore it.
            lines.append(
                f"<!-- unknown node type: {_coerce_str(ctype, default='?')} -->"
            )
            lines.append("")
    return n_rendered


def _render_children_recursive(
    node: dict[str, Any],
    lines: list[str],
    *,
    intent_filter: dict[str, Any] | None,
    recursion_depth: int,
) -> int:
    """Render children of a section (recursively). Same return contract."""
    return _render_root_children(
        node, lines,
        intent_filter=intent_filter,
        recursion_depth=recursion_depth,
    )


def _render_chunk(chunk: dict[str, Any], lines: list[str]) -> None:
    """Emit one chunk's blockquote-anchor + body text into ``lines``."""
    cid = _coerce_str(chunk.get("id"), default="(no-id)")
    page = _coerce_int(chunk.get("page"), default=0)
    bbox = chunk.get("bounding_box")
    bbox_s = _bbox_str(bbox)
    tokens = _coerce_int(chunk.get("token_count"), default=0)
    kind = _kind_label(chunk)
    spath = _section_path(chunk)
    spath_part = f" · §{spath}" if spath else ""

    # Anchor line: a blockquote (>) so markdown renderers treat it as a
    # distinct citation block. Includes chunk id (so the agent can cite),
    # page (1-based), bbox (canvas rect), token count, region_kind, and
    # optional section path. The chunk body follows on subsequent lines
    # WITHOUT the blockquote prefix so it renders as plain text.
    lines.append(
        f"> **[{cid} · page {page} · bbox {bbox_s} · "
        f"{tokens} tok · {kind}]{spath_part}**"
    )
    body = _sanitize_text(_coerce_str(chunk.get("text"), default=""))
    if body:
        lines.append(body)
    lines.append("")


def suffix_marker(*, intent_filter: dict[str, Any] | None) -> str:
    """One-line marker when no chunks survive (filter or empty doc)."""
    if intent_filter and intent_filter.get("matches"):
        return (
            "_No chunks matched this intent. Adjust the query or clear it "
            "to view the full document._"
        )
    return "_No content extracted or matched._"


__all__ = ["build_umr"]
