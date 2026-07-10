"""tests/test_conversations.py -- ConversationStore unit tests (no web layer).

Covers the multi-user model: threads defined by email membership, per-member
access, reuse of an existing 1:1 thread, message senders and roles, and
leave-then-cascade. No Flask, no model, no network.
"""
from __future__ import annotations

import pytest

from uir_pipeline.conversations import (
    MAX_MESSAGE_LEN,
    ConversationError,
    ConversationStore,
)

ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"


@pytest.fixture()
def store(tmp_path):
    return ConversationStore(tmp_path / "monadlabs.db")


class TestCreate:
    def test_create_makes_a_two_member_thread(self, store):
        convo, created = store.create_with(ALICE, BOB)
        assert created is True
        assert set(convo["members"]) == {ALICE, BOB}
        # peer_email is relative to the viewer passed in.
        assert convo["peer_email"] == BOB

    def test_create_is_case_insensitive_and_reuses(self, store):
        first, c1 = store.create_with(ALICE, BOB)
        second, c2 = store.create_with(ALICE, "BOB@EXAMPLE.COM")
        assert c1 is True and c2 is False
        assert first["id"] == second["id"], "same pair must not spawn a second thread"

    def test_either_side_reuses_the_same_thread(self, store):
        first, _ = store.create_with(ALICE, BOB)
        # Bob starting a chat with Alice lands in the existing thread.
        second, created = store.create_with(BOB, ALICE)
        assert created is False
        assert second["id"] == first["id"]

    def test_rejects_self_chat(self, store):
        with pytest.raises(ConversationError, match="yourself"):
            store.create_with(ALICE, "ALICE@example.com")

    def test_rejects_invalid_peer(self, store):
        with pytest.raises(ConversationError, match="valid email"):
            store.create_with(ALICE, "not-an-email")


class TestMembershipAccess:
    def test_list_shows_only_your_threads(self, store):
        store.create_with(ALICE, BOB)
        store.create_with(CAROL, BOB)  # Bob is in both; Alice in one
        alice_threads = store.list_for_email(ALICE)
        assert len(alice_threads) == 1
        assert alice_threads[0]["peer_email"] == BOB
        assert len(store.list_for_email(BOB)) == 2

    def test_both_members_see_the_same_thread(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        assert store.get_for_email(ALICE, convo["id"]) is not None
        assert store.get_for_email(BOB, convo["id"]) is not None
        # peer is computed relative to each viewer.
        assert store.get_for_email(ALICE, convo["id"])["peer_email"] == BOB
        assert store.get_for_email(BOB, convo["id"])["peer_email"] == ALICE

    def test_non_member_cannot_access(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        assert store.get_for_email(CAROL, convo["id"]) is None

    def test_leave_removes_only_your_seat(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        assert store.leave(ALICE, convo["id"]) is True
        # Alice is gone; Bob still has it.
        assert store.get_for_email(ALICE, convo["id"]) is None
        assert store.get_for_email(BOB, convo["id"]) is not None

    def test_leave_by_non_member_is_false(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        assert store.leave(CAROL, convo["id"]) is False

    def test_last_leave_deletes_thread_and_messages(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        store.add_message(convo["id"], "user", "hi", sender_email=ALICE)
        store.leave(ALICE, convo["id"])
        store.leave(BOB, convo["id"])
        # Nobody left -> thread and its messages are gone.
        assert store.get_for_email(ALICE, convo["id"]) is None
        assert store.list_messages(convo["id"]) == []


class TestMessages:
    def test_user_message_records_sender(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        m = store.add_message(convo["id"], "user", "hello", sender_email=ALICE)
        assert m["sender_email"] == ALICE
        assert m["role"] == "user"

    def test_assistant_message_has_no_sender(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        m = store.add_message(convo["id"], "assistant", "grounded answer",
                              citations=[{"doc_title": "p", "page": 1}], grounded=True)
        assert m["sender_email"] is None
        assert m["citations"] == [{"doc_title": "p", "page": 1}]
        assert m["grounded"] is True

    def test_user_message_requires_a_sender(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        with pytest.raises(ValueError, match="sender"):
            store.add_message(convo["id"], "user", "hi")

    def test_invalid_role_raises(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        with pytest.raises(ValueError, match="role"):
            store.add_message(convo["id"], "system", "nope", sender_email=ALICE)

    def test_messages_are_ordered_and_visible_to_both(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        store.add_message(convo["id"], "user", "from alice", sender_email=ALICE)
        store.add_message(convo["id"], "user", "from bob", sender_email=BOB)
        msgs = store.list_messages(convo["id"])
        assert [m["content"] for m in msgs] == ["from alice", "from bob"]
        assert [m["sender_email"] for m in msgs] == [ALICE, BOB]

    def test_content_is_capped(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        m = store.add_message(convo["id"], "user", "y" * (MAX_MESSAGE_LEN + 500), sender_email=ALICE)
        assert len(m["content"]) == MAX_MESSAGE_LEN

    def test_list_preview_reflects_last_message(self, store):
        convo, _ = store.create_with(ALICE, BOB)
        store.add_message(convo["id"], "user", "first", sender_email=ALICE)
        store.add_message(convo["id"], "assistant", "the answer", grounded=True)
        row = store.list_for_email(ALICE)[0]
        assert row["preview"] == "the answer"
        assert row["last_role"] == "assistant"
        assert row["message_count"] == 2

    def test_activity_bumps_thread_to_top(self, store):
        a, _ = store.create_with(ALICE, BOB)
        b, _ = store.create_with(ALICE, CAROL)
        store.add_message(a["id"], "user", "bump", sender_email=ALICE)
        order = [c["id"] for c in store.list_for_email(ALICE)]
        assert order[0] == a["id"]


def test_two_stores_share_the_tables(tmp_path):
    db = tmp_path / "monadlabs.db"
    s1 = ConversationStore(db)
    convo, _ = s1.create_with(ALICE, BOB)
    s2 = ConversationStore(db)
    assert [c["id"] for c in s2.list_for_email(ALICE)] == [convo["id"]]


def test_migrates_away_from_the_legacy_single_owner_schema(tmp_path):
    """A DB carrying the first-generation (user_id) tables is rebuilt.

    ``CREATE TABLE IF NOT EXISTS`` would otherwise leave the old
    ``user_id NOT NULL`` table in place and every insert would fail.
    """
    import sqlite3

    db = tmp_path / "monadlabs.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT 'New conversation',
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL
        );
        INSERT INTO conversations (user_id, title, created_at, updated_at)
        VALUES (1, 'old note', 0, 0);
        """
    )
    conn.commit()
    conn.close()

    # Constructing the store must migrate, not raise.
    store = ConversationStore(db)
    convo, created = store.create_with(ALICE, BOB)  # would fail on the old schema
    assert created is True
    assert set(convo["members"]) == {ALICE, BOB}
