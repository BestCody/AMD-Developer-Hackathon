"""tests/test_tables.py -- Phase H tables module tests.

We mock ``pdfplumber`` so the tests don't need a real PDF. The stub
fixture intercepts ``pdfplumber.open`` so even a non-existent path
returns canned pages (the production pre-check was removed to give
tests full control over the call path).
"""
from __future__ import annotations

from typing import Any
import pytest

from uir_pipeline.tables import TableDraft, _render_markdown, _table_confidence, extract_tables


def test_render_markdown_single_row():
    rows = [["Name", "Age"]]
    md = _render_markdown(rows)
    assert "Name" in md and "Age" in md
    assert "---" in md and md.count("|") >= 4


def test_render_markdown_two_rows_produces_three_lines():
    """A 2-row table (1 header + 1 body) -> 3 lines / 2 newlines."""
    rows = [["Col A", "Col B"], ["1", "2"]]
    md = _render_markdown(rows)
    assert md.count("\n") == 2
    assert len(md.split("\n")) == 3
    assert "Col A" in md and "1" in md


def test_render_markdown_three_rows_produces_four_lines():
    """A 3-row table (1 header + 2 body) -> 4 lines / 3 newlines."""
    rows = [["Col A", "Col B"], ["1", "2"], ["3", "4"]]
    md = _render_markdown(rows)
    assert md.count("\n") == 3
    assert len(md.split("\n")) == 4
    assert "Col A" in md and "1" in md and "3" in md


def test_render_markdown_escapes_pipes():
    rows = [["a|b", "c"], ["x|y", "z"]]
    md = _render_markdown(rows)
    assert r"a\|b" in md and r"x\|y" in md


def test_render_markdown_pads_short_rows():
    """3-col header + 2-col row -> 3-cell padded row ``(4 '|' chars)``."""
    rows = [["A", "B", "C"], ["1", "2"]]
    md = _render_markdown(rows)
    lines = md.split("\n")
    assert len(lines) == 3
    assert lines[2] == "| 1 | 2 |  |"
    assert lines[2].count("|") == 4


def test_render_markdown_truncates_long_rows():
    rows = [["A", "B"], ["1", "2", "3", "4"]]  # body has more cells than header
    md = _render_markdown(rows)
    assert md.count("\n") == 2
    # Body row truncated to header width -> 2 cells = 4 '|' chars.
    assert md.split("\n")[2] == "| 1 | 2 |"
    assert md.split("\n")[2].count("|") == 3


def test_render_markdown_empty_rows_returns_empty_string():
    assert _render_markdown([]) == ""


def test_confidence_empty_is_zero():
    assert _table_confidence([], 2) == 0.0


def test_confidence_single_col_is_zero():
    assert _table_confidence([["x"], ["y"]], 1) == 0.0


def test_confidence_full_table_is_high():
    rows = [["a", "b"], ["c", "d"], ["e", "f"]]
    assert _table_confidence(rows, 2) >= 0.9


# ----------------------------------------------------------------------------
# extract_tables with mocked pdfplumber (stub intercepts even missing files)
# ----------------------------------------------------------------------------

class _FakeTableObj:
    def __init__(self, bbox, rows):
        self.bbox = bbox  # (x0, top, x1, bot)
        self._rows = rows
    def extract(self):
        return list(self._rows)


class _FakePage:
    def __init__(self, *, width=612, height=792, tables=None):
        self.width = width
        self.height = height
        self._tables = tables or []
    def find_tables(self):
        return list(self._tables)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


@pytest.fixture
def stub_pdfplumber(monkeypatch):
    """Patch ``pdfplumber.open`` so ``extract_tables`` sees canned pages."""
    import sys
    import types

    mod = types.ModuleType("pdfplumber")
    mod_state: dict[str, Any] = {"next_pages": []}
    def _open(path):
        return _FakePdf(mod_state.get("next_pages") or [])
    mod.open = _open  # type: ignore[attr-defined]
    orig_pdfplumber = sys.modules.get("pdfplumber")
    sys.modules["pdfplumber"] = mod
    yield mod_state
    if orig_pdfplumber is not None:
        sys.modules["pdfplumber"] = orig_pdfplumber
    else:
        sys.modules.pop("pdfplumber", None)


def test_extract_tables_returns_empty_when_no_tables(stub_pdfplumber):
    stub_pdfplumber["next_pages"] = [_FakePage(width=612, height=792, tables=[])]
    result = extract_tables("/tmp/does-not-matter.pdf", page_numbers=[1])
    assert result == []


def test_extract_tables_renders_table_to_markdown(stub_pdfplumber):
    page = _FakePage(
        width=612, height=792,
        tables=[_FakeTableObj(bbox=(10, 50, 500, 200), rows=[
            ["Item", "Qty"], ["Apple", "3"], ["Banana", "5"],
        ])],
    )
    stub_pdfplumber["next_pages"] = [page]
    result = extract_tables("/tmp/does-not-matter.pdf")
    assert len(result) == 1
    draft = result[0]
    assert isinstance(draft, TableDraft)
    assert draft.page_number == 1
    assert "Item" in draft.markdown and "Apple" in draft.markdown
    assert draft.row_count == 3 and draft.col_count == 2
    # Bbox coords must be on the 0-1000 virtual canvas.
    for coord in draft.bbox:
        assert 0 <= coord <= 1000


def test_extract_tables_filters_none_cells(stub_pdfplumber):
    page = _FakePage(
        width=612, height=792,
        tables=[_FakeTableObj(bbox=(50, 100, 400, 300), rows=[
            ["A", "B", "C"], ["1", None, "3"],
        ])],
    )
    stub_pdfplumber["next_pages"] = [page]
    [draft] = extract_tables("/tmp/does-not-matter.pdf")
    assert draft.col_count == 3
    assert "|" in draft.markdown


def test_extract_tables_handles_table_obj_with_no_rows(stub_pdfplumber):
    page = _FakePage(
        width=612, height=792,
        tables=[_FakeTableObj(bbox=(10, 10, 100, 100), rows=[])],
    )
    stub_pdfplumber["next_pages"] = [page]
    assert extract_tables("/tmp/does-not-matter.pdf") == []


def test_extract_tables_with_page_numbers_filters_pages(stub_pdfplumber):
    pages = [
        _FakePage(width=612, height=792, tables=[]),
        _FakePage(width=612, height=792, tables=[
            _FakeTableObj(bbox=(10, 50, 500, 200), rows=[
                ["A"], ["1"],
            ]),
        ]),
        _FakePage(width=612, height=792, tables=[]),
    ]
    stub_pdfplumber["next_pages"] = pages
    result = extract_tables("/tmp/x.pdf", page_numbers=[2])
    assert len(result) == 1
    assert result[0].page_number == 2
