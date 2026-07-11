"""test_web_users.py -- user search + peer_registered in conversations.

Covers:
* GET /api/users/search (autocomplete) — prefix matching, min-2-char gate,
  isolation, login requirement.
* peer_registered in conversation responses (list, create, messages).
"""
from __future__ import annotations

import pytest

from uir_pipeline.web import create_app


def _signup(client, email):
    r = client.post("/api/auth/signup", json={
        "email": email, "password": "test-password-123",
        "name": email.split("@")[0],
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    return client


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    application = create_app(
        upload_dir=tmp_path / "up",
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        execution="thread",
    )
    application.config["TESTING"] = True
    return application


class TestUsersSearch:
    def test_requires_auth(self, app):
        assert app.test_client().get("/api/users/search?q=al").status_code == 401

    def test_min_two_chars(self, app):
        c = _signup(app.test_client(), "alice@example.com")
        r = c.get("/api/users/search?q=a")
        assert r.status_code == 200
        assert r.get_json()["users"] == []

    def test_prefix_match(self, app):
        alice = _signup(app.test_client(), "alice@example.com")
        _signup(app.test_client(), "bob@example.com")
        _signup(app.test_client(), "alex@example.com")
        r = alice.get("/api/users/search?q=al")
        assert r.status_code == 200
        emails = [u["email"] for u in r.get_json()["users"]]
        assert "alice@example.com" in emails
        assert "alex@example.com" in emails
        assert "bob@example.com" not in emails

    def test_case_insensitive(self, app):
        c = _signup(app.test_client(), "Alice@Example.com")
        r = c.get("/api/users/search?q=ALI")
        assert r.status_code == 200
        assert any(u["email"] == "alice@example.com" for u in r.get_json()["users"])

    def test_returns_name_and_id(self, app):
        c = _signup(app.test_client(), "dave@example.com")
        r = c.get("/api/users/search?q=da")
        assert r.status_code == 200
        users = r.get_json()["users"]
        assert len(users) == 1
        assert users[0]["email"] == "dave@example.com"
        assert users[0]["name"] == "dave"
        assert isinstance(users[0]["id"], int)

    def test_isolated_does_not_leak_all_users(self, app):
        """Alice only sees users whose email matches her prefix query."""
        alice = _signup(app.test_client(), "alice@example.com")
        _signup(app.test_client(), "secret@example.com")
        r = alice.get("/api/users/search?q=al")
        emails = [u["email"] for u in r.get_json()["users"]]
        assert "secret@example.com" not in emails


class TestPeerRegistered:
    def test_unregistered_peer_is_pending(self, app):
        alice = _signup(app.test_client(), "alice@example.com")
        r = alice.post("/api/conversations", json={"peer_email": "nobody@example.com"})
        assert r.status_code == 201
        convo = r.get_json()["conversation"]
        assert convo["peer_email"] == "nobody@example.com"
        assert convo["peer_registered"] is False

        # List view also shows it
        ls = alice.get("/api/conversations").get_json()["conversations"]
        assert [c["peer_registered"] for c in ls] == [False]

        # Messages endpoint also shows it
        cid = convo["id"]
        ms = alice.get(f"/api/conversations/{cid}/messages").get_json()["conversation"]
        assert ms["peer_registered"] is False

    def test_registered_peer_is_member(self, app):
        alice = _signup(app.test_client(), "alice@example.com")
        bob = _signup(app.test_client(), "bob@example.com")
        r = alice.post("/api/conversations", json={"peer_email": "bob@example.com"})
        assert r.status_code == 201
        convo = r.get_json()["conversation"]
        assert convo["peer_registered"] is True

        ls = alice.get("/api/conversations").get_json()["conversations"]
        assert [c["peer_registered"] for c in ls] == [True]

        ms = alice.get(f"/api/conversations/{convo['id']}/messages").get_json()["conversation"]
        assert ms["peer_registered"] is True

    def test_bob_also_sees_peer_registered(self, app):
        alice = _signup(app.test_client(), "alice@example.com")
        _signup(app.test_client(), "bob@example.com")
        alice.post("/api/conversations", json={"peer_email": "bob@example.com"})
        bob = app.test_client()
        bob.post("/api/auth/login", json={"email": "bob@example.com", "password": "test-password-123"})
        ls = bob.get("/api/conversations").get_json()["conversations"]
        assert len(ls) == 1
        assert ls[0]["peer_registered"] is True
        assert ls[0]["peer_email"] == "alice@example.com"
