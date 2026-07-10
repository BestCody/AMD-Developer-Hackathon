"""tests/test_web_conversations.py -- the Chats panel's HTTP surface.

Exercises /api/conversations and the ``gemini:`` command routing end to end
against a signed-in test client, with the model layer stubbed so no network
or API key is needed. Two accounts prove per-user isolation.
"""
from __future__ import annotations

import pytest

from uir_pipeline.web import _gemini_question, create_app


# ----------------------------------------------------------------------------
# The gemini: trigger parser (pure)
# ----------------------------------------------------------------------------

class TestGeminiQuestion:
    def test_plain_text_is_not_a_command(self):
        assert _gemini_question("remember to buy milk") is None

    def test_extracts_the_question(self):
        assert _gemini_question("gemini: what is attention") == "what is attention"

    def test_is_case_insensitive(self):
        assert _gemini_question("GEMINI: hello") == "hello"
        assert _gemini_question("Gemini:  hi ") == "hi"

    def test_leading_whitespace_allowed(self):
        assert _gemini_question("   gemini: x") == "x"

    def test_empty_question_is_empty_string_not_none(self):
        # Distinct from None: it *was* a command, just an empty one.
        assert _gemini_question("gemini:") == ""
        assert _gemini_question("gemini:    ") == ""

    def test_gemini_not_at_start_is_a_note(self):
        assert _gemini_question("ask gemini: later") is None
        assert _gemini_question("regemini: no") is None


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    application = create_app(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        execution="thread",
    )
    application.config["TESTING"] = True
    return application


def _signup(app, email):
    c = app.test_client()
    resp = c.post("/api/auth/signup",
                  json={"email": email, "password": "test-password-123", "name": email[:3]})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return c


@pytest.fixture()
def client(app):
    return _signup(app, "alice@example.com")


# ----------------------------------------------------------------------------
# CRUD + ownership
# ----------------------------------------------------------------------------

class TestConversationRoutes:
    def test_requires_auth(self, app):
        anon = app.test_client()
        assert anon.get("/api/conversations").status_code == 401

    def test_create_list_get_empty(self, client):
        assert client.get("/api/conversations").get_json()["conversations"] == []
        created = client.post("/api/conversations", json={"title": "Notes"})
        assert created.status_code == 201
        cid = created.get_json()["conversation"]["id"]
        listed = client.get("/api/conversations").get_json()["conversations"]
        assert [c["id"] for c in listed] == [cid]
        msgs = client.get(f"/api/conversations/{cid}/messages")
        assert msgs.status_code == 200
        assert msgs.get_json()["messages"] == []

    def test_delete(self, client):
        cid = client.post("/api/conversations", json={}).get_json()["conversation"]["id"]
        assert client.delete(f"/api/conversations/{cid}").status_code == 200
        assert client.get(f"/api/conversations/{cid}/messages").status_code == 404

    def test_another_user_cannot_see_or_touch_a_thread(self, app):
        alice = _signup(app, "alice@example.com")
        bob = _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"title": "secret"}).get_json()["conversation"]["id"]
        # Bob's list is empty and his access is 404 (not 403 -- no oracle).
        assert bob.get("/api/conversations").get_json()["conversations"] == []
        assert bob.get(f"/api/conversations/{cid}/messages").status_code == 404
        assert bob.delete(f"/api/conversations/{cid}").status_code == 404
        assert bob.post(f"/api/conversations/{cid}/messages",
                        json={"text": "hi"}).status_code == 404
        # Alice still has it.
        assert alice.get(f"/api/conversations/{cid}/messages").status_code == 200


# ----------------------------------------------------------------------------
# Messages: plain notes vs gemini: commands
# ----------------------------------------------------------------------------

class TestSendMessage:
    def _new(self, client):
        return client.post("/api/conversations", json={}).get_json()["conversation"]["id"]

    def test_empty_text_rejected(self, client):
        cid = self._new(client)
        assert client.post(f"/api/conversations/{cid}/messages", json={"text": "  "}).status_code == 400

    def test_plain_note_persists_with_no_reply(self, client):
        cid = self._new(client)
        resp = client.post(f"/api/conversations/{cid}/messages", json={"text": "check page 3"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["reply"] is None
        assert body["user_message"]["role"] == "user"
        msgs = client.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [m["content"] for m in msgs] == ["check page 3"]

    def test_first_message_autotitles_the_thread(self, client):
        cid = self._new(client)
        client.post(f"/api/conversations/{cid}/messages", json={"text": "Summarize the deck"})
        row = client.get("/api/conversations").get_json()["conversations"][0]
        assert row["title"] == "Summarize the deck"

    def test_gemini_empty_question_rejected(self, client):
        cid = self._new(client)
        resp = client.post(f"/api/conversations/{cid}/messages", json={"text": "gemini:   "})
        assert resp.status_code == 400
        # Nothing was stored.
        assert client.get(f"/api/conversations/{cid}/messages").get_json()["messages"] == []

    def test_gemini_command_calls_the_model_and_stores_reply(self, client, monkeypatch):
        """A gemini: command runs the grounded answer and persists the reply.

        The model is stubbed: we assert the *routing* (prefix stripped, answer
        stored as an assistant message with citations), not the model itself.
        """
        import uir_pipeline.chat as chat_mod
        seen = {}

        def fake_retrieve(paths, message):
            seen["message"] = message
            return [{"doc_title": "paper", "page": 1, "text": "...", "score": 0.9}]

        def fake_answer(message, contexts, history=None):
            seen["history"] = history
            return {
                "success": True, "answer": "Attention weighs tokens.",
                "citations": contexts, "cited": contexts,
                "invalid_citations": [], "grounded": True, "model": "stub",
            }

        # A done job must exist for retrieval to run (else the no-docs branch).
        _seed_done_job(client)
        monkeypatch.setattr(chat_mod, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_mod, "answer", fake_answer)

        cid = self._new(client)
        resp = client.post(f"/api/conversations/{cid}/messages",
                           json={"text": "gemini: what is attention"})
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        # Prefix stripped before the model saw it.
        assert seen["message"] == "what is attention"
        assert body["reply"]["role"] == "assistant"
        assert body["reply"]["content"] == "Attention weighs tokens."
        assert body["reply"]["grounded"] is True
        # Both messages persisted, in order.
        msgs = client.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "gemini: what is attention"

    def test_gemini_model_failure_persists_user_message_but_no_reply(self, client, monkeypatch):
        """When the model fails, the user message survives and no reply is faked."""
        import uir_pipeline.chat as chat_mod
        _seed_done_job(client)
        monkeypatch.setattr(chat_mod, "retrieve", lambda paths, m: [{"text": "x"}])
        monkeypatch.setattr(chat_mod, "answer",
                            lambda m, c, history=None: {"success": False, "error": "no FIREWORKS_API_KEY"})

        cid = self._new(client)
        resp = client.post(f"/api/conversations/{cid}/messages",
                           json={"text": "gemini: anything"})
        assert resp.status_code == 502
        assert "FIREWORKS" in resp.get_json()["error"]
        # The user's question is not lost; there is no assistant message.
        msgs = client.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [m["role"] for m in msgs] == ["user"]
        assert msgs[0]["content"] == "gemini: anything"

    def test_gemini_with_no_documents_answers_from_the_no_docs_branch(self, client):
        """No stubbing, no documents: the grounded helper returns its no-docs
        message (status 200), which is stored as the assistant reply."""
        cid = self._new(client)
        resp = client.post(f"/api/conversations/{cid}/messages",
                           json={"text": "gemini: what is X"})
        assert resp.status_code == 200
        assert "haven't converted any documents" in resp.get_json()["reply"]["content"]


def _seed_done_job(client):
    """Insert a completed job into the app registry so retrieval runs.

    The chat routes gather ``uir_path`` from the caller's ``done`` jobs; with
    none, the no-docs branch short-circuits before the model. ``app.config
    ["JOBS"]`` is the same dict the routes read. ``uir_path`` need not exist on
    disk here because the tests that call this stub ``chat.retrieve``.
    """
    import pathlib

    from uir_pipeline.web import JOB_DONE, Job

    jobs = client.application.config["JOBS"]
    uid = client.get("/api/auth/me").get_json()["user"]["id"]
    job = Job(job_id="seededjob", user_id=uid, filename="d.pdf",
              upload_path=pathlib.Path("d.pdf"))
    job.status = JOB_DONE
    job.uir_path = pathlib.Path("seeded.uir.json")
    jobs[job.job_id] = job
