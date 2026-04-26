"""Tests for tau.core.state — SessionDB with FTS5."""

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from tau.core.state import SessionDB


@pytest.fixture
def db(tmp_path):
    """Create a SessionDB with a temporary path."""
    return SessionDB(db_path=tmp_path / "test_state.db")


@pytest.fixture
def db_with_data(db):
    """SessionDB pre-populated with test data."""
    sid = db.create_session(
        session_id="test-session-1",
        source="cli",
        model="gpt-4o",
        user_id="user1",
    )
    db.append_message(sid, "user", "Hello, how are you?")
    db.append_message(sid, "assistant", "I'm doing well! How can I help you today?")
    db.append_message(sid, "user", "Write a Python function to sort a list")
    db.append_message(
        sid,
        "assistant",
        "Here's a Python function:\n\ndef sort_list(items):\n    return sorted(items)",
    )

    sid2 = db.create_session(
        session_id="test-session-2",
        source="telegram",
        model="claude-3",
        user_id="user2",
    )
    db.append_message(sid2, "user", "What is the capital of France?")
    db.append_message(sid2, "assistant", "The capital of France is Paris.")
    return db


# ─── Session CRUD ───


class TestSessionCRUD:
    def test_create_session(self, db):
        sid = db.create_session(
            session_id="s1",
            source="cli",
            model="gpt-4o",
        )
        assert sid == "s1"
        session = db.get_session("s1")
        assert session is not None
        assert session["source"] == "cli"
        assert session["model"] == "gpt-4o"

    def test_create_duplicate_ignored(self, db):
        """INSERT OR IGNORE prevents duplicate session IDs."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s1", source="telegram")
        session = db.get_session("s1")
        assert session["source"] == "cli"  # first one wins

    def test_end_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", "user_exit")
        session = db.get_session("s1")
        assert session["ended_at"] is not None
        assert session["end_reason"] == "user_exit"

    def test_get_nonexistent_session(self, db):
        assert db.get_session("nonexistent") is None

    def test_delete_session(self, db_with_data):
        assert db_with_data.delete_session("test-session-1") is True
        assert db_with_data.get_session("test-session-1") is None
        assert db_with_data.message_count("test-session-1") == 0

    def test_delete_nonexistent(self, db):
        assert db.delete_session("nope") is False

    def test_ensure_session(self, db):
        db.ensure_session("new-session", source="discord")
        session = db.get_session("new-session")
        assert session is not None
        assert session["source"] == "discord"

    def test_ensure_idempotent(self, db):
        db.ensure_session("s1", source="cli", model="gpt-4o")
        db.ensure_session("s1", source="telegram", model="claude")
        session = db.get_session("s1")
        assert session["source"] == "cli"  # first one wins


class TestSessionTitle:
    def test_set_title(self, db):
        db.create_session(session_id="s1", source="cli")
        assert db.set_session_title("s1", "My Session") is True
        session = db.get_session("s1")
        assert session["title"] == "My Session"

    def test_title_uniqueness(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.set_session_title("s1", "Unique Title")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("s2", "Unique Title")

    def test_clear_title(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Title")
        db.set_session_title("s1", None)
        session = db.get_session("s1")
        assert session["title"] is None


class TestTokenCounts:
    def test_update_tokens(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=100, output_tokens=50)
        db.update_token_counts("s1", input_tokens=200, output_tokens=75)
        session = db.get_session("s1")
        assert session["input_tokens"] == 300
        assert session["output_tokens"] == 125

    def test_backfill_model(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", model="gpt-4o")
        session = db.get_session("s1")
        assert session["model"] == "gpt-4o"


# ─── Messages ───


class TestMessages:
    def test_append_and_get(self, db):
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", "user", "Hello!")
        assert msg_id > 0

        messages = db.get_messages("s1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"

    def test_message_ordering(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", "First")
        db.append_message("s1", "assistant", "Second")
        db.append_message("s1", "user", "Third")

        messages = db.get_messages("s1")
        assert len(messages) == 3
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    def test_message_count_updates(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", "Msg 1")
        db.append_message("s1", "assistant", "Msg 2")

        session = db.get_session("s1")
        assert session["message_count"] == 2

    def test_tool_call_tracking(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", "assistant", None,
            tool_calls=[{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}],
        )
        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_get_messages_as_conversation(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", "Hi")
        db.append_message("s1", "assistant", "Hello!")

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 2
        assert conv[0] == {"role": "user", "content": "Hi"}
        assert conv[1] == {"role": "assistant", "content": "Hello!"}

    def test_clear_messages(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", "Msg")
        db.clear_messages("s1")
        assert db.message_count("s1") == 0
        session = db.get_session("s1")
        assert session["message_count"] == 0


# ─── FTS5 Search ───


class TestSearch:
    def test_basic_search(self, db_with_data):
        results = db_with_data.search_messages("Python function")
        assert len(results) > 0
        # Should find the message about Python function
        snippets = " ".join(r.get("snippet", "") for r in results)
        assert "Python" in snippets or "function" in snippets

    def test_search_with_source_filter(self, db_with_data):
        results = db_with_data.search_messages("capital", source_filter=["telegram"])
        assert len(results) > 0
        assert all(r["source"] == "telegram" for r in results)

    def test_search_with_role_filter(self, db_with_data):
        results = db_with_data.search_messages("Hello", role_filter=["user"])
        assert len(results) > 0
        assert all(r["role"] == "user" for r in results)

    def test_search_no_results(self, db_with_data):
        results = db_with_data.search_messages("xyzzynonexistent")
        assert len(results) == 0

    def test_search_empty_query(self, db_with_data):
        results = db_with_data.search_messages("")
        assert len(results) == 0

    def test_search_context_included(self, db_with_data):
        results = db_with_data.search_messages("sort")
        assert len(results) > 0
        assert "context" in results[0]

    def test_sanitize_fts5_query(self):
        # These should not crash
        assert SessionDB._sanitize_fts5_query("hello world") == "hello world"
        assert SessionDB._sanitize_fts5_query('"exact phrase"') == '"exact phrase"'
        # Special chars stripped
        assert "(" not in SessionDB._sanitize_fts5_query("func()")
        # Dotted terms quoted
        sanitized = SessionDB._sanitize_fts5_query("os.path.join")
        assert '"os.path.join"' in sanitized


# ─── Listing ───


class TestListing:
    def test_list_sessions_rich(self, db_with_data):
        sessions = db_with_data.list_sessions_rich()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["id"] == "test-session-2"

    def test_list_with_source_filter(self, db_with_data):
        sessions = db_with_data.list_sessions_rich(source="telegram")
        assert len(sessions) == 1
        assert sessions[0]["source"] == "telegram"

    def test_list_with_preview(self, db_with_data):
        sessions = db_with_data.list_sessions_rich()
        for s in sessions:
            assert "preview" in s

    def test_session_count(self, db_with_data):
        assert db_with_data.session_count() == 2
        assert db_with_data.session_count(source="cli") == 1
        assert db_with_data.session_count(source="telegram") == 1

    def test_message_count(self, db_with_data):
        assert db_with_data.message_count() == 6  # 4 + 2
        assert db_with_data.message_count("test-session-1") == 4


# ─── Export ───


class TestExport:
    def test_export_session(self, db_with_data):
        exported = db_with_data.export_session("test-session-1")
        assert exported is not None
        assert exported["id"] == "test-session-1"
        assert len(exported["messages"]) == 4

    def test_export_nonexistent(self, db):
        assert db.export_session("nope") is None


# ─── Pruning ───


class TestPruning:
    def test_prune_old_sessions(self, db):
        # Create and immediately end a session with a fake old timestamp
        db.create_session(session_id="old1", source="cli")
        db.end_session("old1", "done")
        # Manually backdate
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = 'old1'",
                (time.time() - 200 * 86400,),
            )
        )

        db.create_session(session_id="new1", source="cli")

        pruned = db.prune_sessions(older_than_days=100)
        assert pruned == 1
        assert db.get_session("old1") is None
        assert db.get_session("new1") is not None


# ─── Concurrent access ───


class TestConcurrency:
    def test_concurrent_writes(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "concurrent.db")
        db.create_session(session_id="s1", source="cli")

        errors = []

        def writer(thread_id):
            try:
                for i in range(20):
                    db.append_message("s1", "user", f"Thread {thread_id} msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Write errors: {errors}"
        assert db.message_count("s1") == 80  # 4 threads × 20 messages

    def test_wal_mode(self, db):
        """Verify WAL mode is enabled."""
        with db._lock:
            cursor = db._conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"


# ─── Close ───


class TestClose:
    def test_close(self, db):
        db.create_session(session_id="s1", source="cli")
        db.close()
        # After close, connection is None
        assert db._conn is None
