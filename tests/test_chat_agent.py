"""test_chat_agent.py -- the agentic tool-calling loop in chat.answer().

The model call is stubbed via ``chat._complete`` (no Fireworks key needed):
the first call returns a ``tool_calls`` message, the second a final answer.
The tool execution is stubbed via ``chat._run_tool``. Asserts the loop runs
the tool, records ``tool_steps``, gathers the tool's passages, and maps the
final citation back to the gathered passage.
"""
from __future__ import annotations

import json
import os

import pytest

from uir_pipeline import chat


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")


def _tool_call(name="search", query="invoices", cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps({"query": query, "top_k": 4})}}


def test_agent_loop_runs_tool_then_answers(monkeypatch):
    calls = {"complete": 0, "tool": 0}

    def fake_complete(messages, **kw):
        calls["complete"] += 1
        if calls["complete"] == 1:
            return ({"role": "assistant", "content": None, "tool_calls": [_tool_call()]}, {"p": 1})
        return ({"role": "assistant", "content": "The total is $42 [1]."}, {"p": 2})

    def fake_run_tool(name, args, docs, q):
        calls["tool"] += 1
        assert args.get("query") == "invoices"
        return [{"job_id": "j1", "doc_id": "d1", "doc_title": "Invoices Q3",
                 "chunk_id": "c1", "page": 1, "text": "total 42", "score": 0.9,
                 "title_match": True}]

    monkeypatch.setattr(chat, "_complete", fake_complete)
    monkeypatch.setattr(chat, "_run_tool", fake_run_tool)

    docs = [{"job_id": "j1", "uir_path": "/tmp/x.uir.json", "filename": "invoices-q3.pdf"}]
    out = chat.answer("what is the total?", [], history=[], docs=docs)

    assert out["success"] is True
    assert out["answer"] == "The total is $42 [1]."
    assert out["tool_steps"] == [{"tool": "search", "query": "invoices", "n_results": 1}]
    # The tool's passage was gathered and is citation [1].
    assert out["cited"] == [1]
    assert len(out["citations"]) == 1
    assert out["citations"][0]["doc_title"] == "Invoices Q3"
    assert out["grounded"] is True
    assert calls["complete"] == 2 and calls["tool"] == 1


def test_agent_loop_caps_and_forces_final_answer(monkeypatch):
    """If the model keeps calling tools, the cap forces a tool_choice='none' call."""
    calls = {"complete": 0}

    def fake_complete(messages, **kw):
        calls["complete"] += 1
        # Always request a tool until forced (tool_choice='none' on the cap call).
        if kw.get("tool_choice") == "none":
            return ({"role": "assistant", "content": "Done [1]."}, {"p": 9})
        return ({"role": "assistant", "content": None, "tool_calls": [_tool_call(cid=f"c{calls['complete']}")]}, {"p": 1})

    monkeypatch.setattr(chat, "_run_tool", lambda name, args, docs, q: [
        {"job_id": "j1", "doc_id": "d", "doc_title": "T", "chunk_id": "c", "page": 1,
         "text": "x", "score": 0.5, "title_match": False}])
    monkeypatch.setattr(chat, "_complete", fake_complete)

    docs = [{"job_id": "j1", "uir_path": "/tmp/x.uir.json", "filename": "x.pdf"}]
    out = chat.answer("q", [], docs=docs, max_iterations=2)

    assert out["success"] is True
    assert out["answer"] == "Done [1]."
    # 2 tool-calling iterations + 1 forced final call = 3 complete calls.
    assert calls["complete"] == 3
    # Both tool calls recorded.
    assert len(out["tool_steps"]) == 2


def test_agent_loop_without_tools_answers_directly(monkeypatch):
    """When the model returns no tool_calls, it answers in one call."""
    def fake_complete(messages, **kw):
        return ({"role": "assistant", "content": "It is 42 [1]."}, {"p": 1})
    monkeypatch.setattr(chat, "_complete", fake_complete)
    monkeypatch.setattr(chat, "_run_tool", lambda *a, **k: pytest.fail("tool should not run"))

    docs = [{"job_id": "j1", "uir_path": "/tmp/x.uir.json", "filename": "x.pdf"}]
    out = chat.answer("q", [{"doc_id": "d", "doc_title": "T", "chunk_id": "c1",
                             "page": 1, "text": "p", "score": 0.9}], docs=docs)
    assert out["success"] is True
    assert out["tool_steps"] == []
    assert out["cited"] == [1]


def test_single_shot_path_preserved_when_no_docs(monkeypatch):
    """answer(query, contexts) with no docs stays single-shot (test_chat_citations contract)."""
    def fake_complete(messages, **kw):
        # Single-shot must NOT pass tools.
        assert "tools" not in kw or kw.get("tools") is None
        return ({"role": "assistant", "content": "Heads: 8 [1]."}, {})
    monkeypatch.setattr(chat, "_complete", fake_complete)
    out = chat.answer("q", [{"doc_id": "d", "doc_title": "T", "chunk_id": "c1",
                             "page": 1, "text": "p", "score": 0.9}])
    assert out["success"] is True
    assert out["cited"] == [1]
    assert out["tool_steps"] == []


def test_empty_contexts_and_no_docs_short_circuits():
    out = chat.answer("q", [])
    assert out["grounded"] is False
    assert out["tool_steps"] == []
    assert out["cited"] == [] and out["invalid_citations"] == []
