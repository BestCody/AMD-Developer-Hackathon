"""tests/test_conversations.py -- ConversationStore unit tests (no web layer).

Covers persistence, per-user ownership scoping, message roles/citations, and
the auto-title-on-first-message behaviour. No Flask, no model, no network.
"""
from __future__ import annotations

import pytest

from uir_pipeline.conversations import (
    MAX_MESSAGE_LEN,
    MAX_TITLE_LEN,
    ConversationStore,
)


@pytest.fixture()
def store(tmp_path):
    return ConversationStore(tmp_path / "monadlabs.db")


class TestConversations:
    def test_create_and_list(self, store):
        c = store.create(1, "First")
        assert c["title"] == "First"
        assert c["message_count"] == 0
        listed = store.list_for_user(1)
        assert [x["id"] for x in listed] == [c["id"]]

    def test_blank_title_falls_back_to_default(self, store):
        c = store.create(1, "   ")
        assert c["title"] == "New conversation"

    def test_title_is_truncated(self, store):
        c = store.create(1, "x" * 500)
        assert len(c["title"]) == MAX_TITLE_LEN

    def test_list_is_scoped_per_user(self, store):
        store.create(1, "alice-thread")
        store.create(2, "bob-thread")
        assert [c["title"] for c in store.list_for_user(1)] == ["alice-thread"]
        assert [c["title"] for c in store.list_for_user(2)] == ["bob-thread"]

    def test_get_enforces_ownership(self, store):
        c = store.create(1, "mine")
        assert store.get(1, c["id"]) is not None
        # Another user cannot fetch it even with the right id.
        assert store.get(2, c["id"]) is None

    def test_get_missing_returns_none(self, store):
        assert store.get(1, 999999) is None

    def test_delete_enforces_ownership(self, store):
        c = store.create(1, "mine")
        # Wrong user cannot delete.
        assert store.delete(2, c["id"]) is False
        assert store.get(1, c["id"]) is not None
        # Owner can.
        assert store.delete(1, c["id"]) is True
        assert store.get(1, c["id"]) is None

    def test_delete_cascades_to_messages(self, store):
        c = store.create(1, "mine")
        store.add_message(c["id"], "user", "hello")
        store.add_message(c["id"], "assistant", "hi", grounded=True)
        assert len(store.list_messages(c["id"])) == 2
        store.delete(1, c["id"])
        # Messages must not survive their conversation.
        assert store.list_messages(c["id"]) == []

    def test_list_orders_by_recent_activity(self, store):
        a = store.create(1, "a")
        b = store.create(1, "b")
        # Touch `a` after `b` by adding a message -> a should sort first.
        store.add_message(a["id"], "user", "bump")
        order = [c["id"] for c in store.list_for_user(1)]
        assert order[0] == a["id"] and order[1] == b["id"]


class TestMessages:
    def test_add_and_list_messages(self, store):
        c = store.create(1)
        store.add_message(c["id"], "user", "gemini: what is X")
        store.add_message(
            c["id"], "assistant", "X is Y",
            citations=[{"doc_title": "paper", "page": 3}], grounded=True,
        )
        msgs = store.list_messages(c["id"])
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["citations"] == [{"doc_title": "paper", "page": 3}]
        assert msgs[1]["grounded"] is True

    def test_plain_note_has_no_citations_or_grounded(self, store):
        c = store.create(1)
        m = store.add_message(c["id"], "user", "just a note")
        assert m["citations"] == []
        assert m["grounded"] is None

    def test_invalid_role_raises(self, store):
        c = store.create(1)
        with pytest.raises(ValueError, match="role"):
            store.add_message(c["id"], "system", "nope")

    def test_message_content_is_capped(self, store):
        c = store.create(1)
        m = store.add_message(c["id"], "user", "y" * (MAX_MESSAGE_LEN + 1000))
        assert len(m["content"]) == MAX_MESSAGE_LEN

    def test_adding_a_message_bumps_updated_at(self, store):
        c = store.create(1)
        before = store.get(1, c["id"])["updated_at"]
        store.add_message(c["id"], "user", "hello")
        after = store.get(1, c["id"])["updated_at"]
        assert after >= before

    def test_list_preview_reflects_last_message(self, store):
        c = store.create(1)
        store.add_message(c["id"], "user", "first")
        store.add_message(c["id"], "assistant", "second answer", grounded=True)
        row = store.list_for_user(1)[0]
        assert row["preview"] == "second answer"
        assert row["last_role"] == "assistant"
        assert row["message_count"] == 2


class TestAutoTitle:
    def test_autotitle_sets_from_first_message_when_default(self, store):
        c = store.create(1)  # default title
        store.autotitle_if_default(c["id"], "Summarize the Q3 report")
        assert store.get(1, c["id"])["title"] == "Summarize the Q3 report"

    def test_autotitle_does_not_overwrite_a_custom_title(self, store):
        c = store.create(1, "My named thread")
        store.autotitle_if_default(c["id"], "something else")
        assert store.get(1, c["id"])["title"] == "My named thread"

    def test_autotitle_ignores_blank(self, store):
        c = store.create(1)
        store.autotitle_if_default(c["id"], "   ")
        assert store.get(1, c["id"])["title"] == "New conversation"

    def test_autotitle_truncates(self, store):
        c = store.create(1)
        store.autotitle_if_default(c["id"], "z" * 500)
        assert len(store.get(1, c["id"])["title"]) == MAX_TITLE_LEN


def test_two_stores_share_the_tables(tmp_path):
    """A second store on the same file sees the first's data (one DB file)."""
    db = tmp_path / "monadlabs.db"
    s1 = ConversationStore(db)
    c = s1.create(7, "persisted")
    s2 = ConversationStore(db)
    assert [x["id"] for x in s2.list_for_user(7)] == [c["id"]]
