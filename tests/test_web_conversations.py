"""tests/test_web_conversations.py -- the Chats panel's HTTP surface.

Exercises /api/conversations as a multi-user messaging surface: two accounts
share a thread, a third cannot see it, and the ``gemini:`` command answers
from the sender's documents. The model layer is stubbed so no network or API
key is needed.
"""
from __future__ import annotations

import pytest

from uir_pipeline.web import _gemini_question, create_app


# ----------------------------------------------------------------------------
# The gemini: trigger parser (pure)
# ----------------------------------------------------------------------------

class TestGeminiQuestion:
    def test_plain_text_is_not_a_command(self):
        assert _gemini_question("hey, did you see the deck?") is None

    def test_extracts_the_question(self):
        assert _gemini_question("gemini: what is attention") == "what is attention"

    def test_is_case_insensitive(self):
        assert _gemini_question("GEMINI: hello") == "hello"
        assert _gemini_question("Gemini:  hi ") == "hi"

    def test_leading_whitespace_allowed(self):
        assert _gemini_question("   gemini: x") == "x"

    def test_empty_question_is_empty_string_not_none(self):
        assert _gemini_question("gemini:") == ""
        assert _gemini_question("gemini:    ") == ""

    def test_gemini_not_at_start_is_a_message(self):
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
                  json={"email": email, "password": "test-password-123", "name": email.split("@")[0]})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return c


@pytest.fixture()
def alice(app):
    return _signup(app, "alice@example.com")


# ----------------------------------------------------------------------------
# Creating / listing / membership
# ----------------------------------------------------------------------------

class TestConversationRoutes:
    def test_requires_auth(self, app):
        assert app.test_client().get("/api/conversations").status_code == 401

    def test_create_rejects_invalid_email(self, alice):
        resp = alice.post("/api/conversations", json={"peer_email": "nope"})
        assert resp.status_code == 400

    def test_create_rejects_self(self, alice):
        resp = alice.post("/api/conversations", json={"peer_email": "alice@example.com"})
        assert resp.status_code == 400
        assert "yourself" in resp.get_json()["error"]

    def test_create_and_reopen_are_idempotent(self, alice):
        first = alice.post("/api/conversations", json={"peer_email": "bob@example.com"})
        assert first.status_code == 201
        cid = first.get_json()["conversation"]["id"]
        again = alice.post("/api/conversations", json={"peer_email": "bob@example.com"})
        assert again.status_code == 200  # reopened, not duplicated
        assert again.get_json()["conversation"]["id"] == cid

    def test_both_parties_see_the_thread(self, app):
        alice = _signup(app, "alice@example.com")
        bob = _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]
        # Alice sees Bob as the peer; Bob sees Alice.
        a_list = alice.get("/api/conversations").get_json()["conversations"]
        b_list = bob.get("/api/conversations").get_json()["conversations"]
        assert [c["peer_email"] for c in a_list] == ["bob@example.com"]
        assert [c["peer_email"] for c in b_list] == ["alice@example.com"]
        assert a_list[0]["id"] == cid == b_list[0]["id"]

    def test_a_stranger_cannot_access(self, app):
        alice = _signup(app, "alice@example.com")
        _signup(app, "bob@example.com")
        carol = _signup(app, "carol@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]
        # Carol is not a member: empty list, 404 on direct access (no oracle).
        assert carol.get("/api/conversations").get_json()["conversations"] == []
        assert carol.get(f"/api/conversations/{cid}/messages").status_code == 404
        assert carol.post(f"/api/conversations/{cid}/messages", json={"text": "hi"}).status_code == 404
        assert carol.delete(f"/api/conversations/{cid}").status_code == 404

    def test_leave(self, app):
        alice = _signup(app, "alice@example.com")
        bob = _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]
        assert alice.delete(f"/api/conversations/{cid}").status_code == 200
        # Alice left; Bob still has it.
        assert alice.get(f"/api/conversations/{cid}/messages").status_code == 404
        assert bob.get(f"/api/conversations/{cid}/messages").status_code == 200


# ----------------------------------------------------------------------------
# Messaging between users + the gemini command
# ----------------------------------------------------------------------------

class TestMessaging:
    def _thread(self, alice):
        return alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]

    def test_empty_text_rejected(self, alice):
        cid = self._thread(alice)
        assert alice.post(f"/api/conversations/{cid}/messages", json={"text": " "}).status_code == 400

    def test_message_from_one_user_is_visible_to_the_other(self, app):
        alice = _signup(app, "alice@example.com")
        bob = _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]
        resp = alice.post(f"/api/conversations/{cid}/messages", json={"text": "did you see the deck?"})
        assert resp.status_code == 200
        assert resp.get_json()["reply"] is None  # not a command
        # Bob sees it, attributed to Alice.
        msgs = bob.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [(m["sender_email"], m["content"]) for m in msgs] == [("alice@example.com", "did you see the deck?")]

    def test_gemini_empty_question_rejected_and_stores_nothing(self, alice):
        cid = self._thread(alice)
        resp = alice.post(f"/api/conversations/{cid}/messages", json={"text": "gemini:   "})
        assert resp.status_code == 400
        assert alice.get(f"/api/conversations/{cid}/messages").get_json()["messages"] == []

    def test_gemini_command_answers_from_sender_docs_and_shares_the_reply(self, app, monkeypatch):
        alice = _signup(app, "alice@example.com")
        bob = _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]

        import uir_pipeline.chat as chat_mod
        seen = {}
        monkeypatch.setattr(chat_mod, "retrieve",
                            lambda paths, message: (seen.update(message=message) or [{"doc_title": "p", "page": 1, "text": "...", "score": 0.9}]))
        monkeypatch.setattr(chat_mod, "answer",
                            lambda message, contexts, history=None: {
                                "success": True, "answer": "Attention weighs tokens.",
                                "citations": contexts, "cited": contexts,
                                "invalid_citations": [], "grounded": True, "model": "stub"})
        _seed_done_job(alice)

        resp = alice.post(f"/api/conversations/{cid}/messages", json={"text": "gemini: what is attention"})
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert seen["message"] == "what is attention"  # prefix stripped
        body = resp.get_json()
        assert body["reply"]["role"] == "assistant"
        assert body["reply"]["sender_email"] is None
        # Both the question (from Alice) and the shared answer are visible to Bob.
        msgs = bob.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["sender_email"] == "alice@example.com"
        assert msgs[1]["content"] == "Attention weighs tokens."

    def test_gemini_model_failure_keeps_the_question_and_fakes_no_reply(self, app, monkeypatch):
        alice = _signup(app, "alice@example.com")
        _signup(app, "bob@example.com")
        cid = alice.post("/api/conversations", json={"peer_email": "bob@example.com"}).get_json()["conversation"]["id"]

        import uir_pipeline.chat as chat_mod
        monkeypatch.setattr(chat_mod, "retrieve", lambda paths, m: [{"text": "x"}])
        monkeypatch.setattr(chat_mod, "answer",
                            lambda m, c, history=None: {"success": False, "error": "no FIREWORKS_API_KEY"})
        _seed_done_job(alice)

        resp = alice.post(f"/api/conversations/{cid}/messages", json={"text": "gemini: anything"})
        assert resp.status_code == 502
        assert "FIREWORKS" in resp.get_json()["error"]
        msgs = alice.get(f"/api/conversations/{cid}/messages").get_json()["messages"]
        assert [m["role"] for m in msgs] == ["user"]  # question kept, no assistant

    def test_gemini_with_no_documents_uses_the_no_docs_branch(self, alice):
        cid = self._thread(alice)
        resp = alice.post(f"/api/conversations/{cid}/messages", json={"text": "gemini: what is X"})
        assert resp.status_code == 200
        assert "haven't converted any documents" in resp.get_json()["reply"]["content"]


def _seed_done_job(client):
    """Insert a completed job for the caller so retrieval runs.

    The chat routes gather ``uir_path`` from the caller's ``done`` jobs; with
    none, the no-docs branch short-circuits before the model. ``app.config
    ["JOBS"]`` is the same dict the routes read.
    """
    import pathlib

    from uir_pipeline.web import JOB_DONE, Job

    jobs = client.application.config["JOBS"]
    uid = client.get("/api/auth/me").get_json()["user"]["id"]
    job = Job(job_id=f"seed-{uid}", user_id=uid, filename="d.pdf",
              upload_path=pathlib.Path("d.pdf"))
    job.status = JOB_DONE
    job.uir_path = pathlib.Path("seeded.uir.json")
    jobs[job.job_id] = job
