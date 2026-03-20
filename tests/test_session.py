"""Tests for tau.core.session."""

import json
import pytest
from pathlib import Path
from tau.core.session import SessionManager, SessionNotFoundError
from tau.core.types import AgentConfig, ForkInfo


@pytest.fixture()
def sm(tmp_path: Path) -> SessionManager:
    return SessionManager(sessions_dir=tmp_path / "sessions")


def _cfg() -> AgentConfig:
    return AgentConfig(provider="openai", model="gpt-4o")


def _messages(n: int = 6) -> list[dict]:
    """Alternating user/assistant messages (no system)."""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": f"msg {i}"} for i in range(n)]


def test_new_session_creates_file(sm: SessionManager, tmp_path: Path):
    session = sm.new_session(_cfg(), name="test-session")
    assert (tmp_path / "sessions" / f"{session.id}.json").exists()


def test_save_and_load_roundtrip(sm: SessionManager):
    session = sm.new_session(_cfg(), name="roundtrip")
    sm.save(session, messages=[{"role": "user", "content": "hello"}])
    loaded = sm.load(session.id)
    assert loaded.id == session.id
    assert loaded.name == "roundtrip"
    assert loaded.messages[0]["content"] == "hello"


def test_load_by_prefix(sm: SessionManager):
    session = sm.new_session(_cfg())
    prefix = session.id[:8]
    loaded = sm.load(prefix)
    assert loaded.id == session.id


def test_load_missing_raises(sm: SessionManager):
    with pytest.raises(SessionNotFoundError):
        sm.load("nonexistent-id")


def test_list_sessions(sm: SessionManager):
    sm.new_session(_cfg(), name="a")
    sm.new_session(_cfg(), name="b")
    metas = sm.list_sessions()
    assert len(metas) == 2
    names = {m.name for m in metas}
    assert names == {"a", "b"}


def test_delete_session(sm: SessionManager):
    session = sm.new_session(_cfg())
    sm.delete(session.id)
    with pytest.raises(SessionNotFoundError):
        sm.load(session.id)


def test_delete_missing_raises(sm: SessionManager):
    with pytest.raises(SessionNotFoundError):
        sm.delete("ghost-id")


def test_session_meta_display(sm: SessionManager):
    session = sm.new_session(_cfg(), name="display-test")
    display = session.meta.display()
    assert session.id[:8] in display
    assert "display-test" in display
    assert "gpt-4o" in display


class TestFork:
    def test_fork_creates_new_session(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=3)
        assert forked.id != parent.id

    def test_fork_creates_file_on_disk(self, sm: SessionManager, tmp_path: Path):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=3)
        assert (tmp_path / "sessions" / f"{forked.id}.json").exists()

    def test_fork_messages_sliced_at_index(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=3)
        # fork_index=3 means messages[0..3] inclusive → 4 messages
        assert forked.messages == parent.messages[:4]

    def test_fork_at_index_zero(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=0)
        assert len(forked.messages) == 1
        assert forked.messages[0] == parent.messages[0]

    def test_fork_at_last_index(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        msgs = _messages(6)
        sm.save(parent, messages=msgs)
        forked = sm.fork(parent.id, fork_index=5)
        assert forked.messages == msgs

    def test_fork_records_parent_id(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=2)
        assert forked.parent_id == parent.id

    def test_fork_records_fork_index(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=2)
        assert forked.fork_index == 2

    def test_fork_inherits_config(self, sm: SessionManager):
        cfg = AgentConfig(provider="openai", model="gpt-4o-mini", max_tokens=4096)
        parent = sm.new_session(cfg)
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=1)
        assert forked.config.model == "gpt-4o-mini"
        assert forked.config.max_tokens == 4096

    def test_fork_compactions_not_inherited(self, sm: SessionManager):
        from tau.core.types import CompactionEntry
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        sm.append_compaction(parent, CompactionEntry(
            summary="old summary", tokens_before=1000, timestamp="2024-01-01T00:00:00+00:00"
        ))
        forked = sm.fork(parent.id, fork_index=3)
        assert forked.compactions == []

    def test_fork_custom_name(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=1, name="my branch")
        assert forked.name == "my branch"

    def test_fork_default_name_contains_parent(self, sm: SessionManager):
        parent = sm.new_session(_cfg(), name="main")
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=1)
        assert parent.id[:8] in forked.name or "main" in forked.name

    def test_fork_out_of_range_raises(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        with pytest.raises(ValueError, match="out of range"):
            sm.fork(parent.id, fork_index=99)

    def test_fork_negative_index_raises(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        with pytest.raises(ValueError, match="out of range"):
            sm.fork(parent.id, fork_index=-1)

    def test_fork_empty_messages_raises(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        # no messages saved — messages list is empty
        with pytest.raises(ValueError, match="out of range"):
            sm.fork(parent.id, fork_index=0)

    def test_fork_is_loadable(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        forked = sm.fork(parent.id, fork_index=3)
        loaded = sm.load(forked.id)
        assert loaded.parent_id == parent.id
        assert loaded.fork_index == 3

    def test_fork_parent_unchanged(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        original_msgs = _messages(6)
        sm.save(parent, messages=original_msgs)
        sm.fork(parent.id, fork_index=2)
        reloaded_parent = sm.load(parent.id)
        assert reloaded_parent.messages == original_msgs

    def test_fork_of_fork_records_correct_parent(self, sm: SessionManager):
        """Forking a fork should record the immediate parent, not the grandparent."""
        root = sm.new_session(_cfg())
        sm.save(root, messages=_messages(8))
        child = sm.fork(root.id, fork_index=4)
        sm.save(child, messages=child.messages + [{"role": "user", "content": "extra"}])
        grandchild = sm.fork(child.id, fork_index=3)
        assert grandchild.parent_id == child.id

    def test_fork_persists_parent_id_to_disk(self, sm: SessionManager, tmp_path: Path):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=2)
        data = json.loads((tmp_path / "sessions" / f"{forked.id}.json").read_text())
        assert data["parent_id"] == parent.id
        assert data["fork_index"] == 2


class TestGetForkPoints:
    def test_returns_user_messages_only(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))  # roles: user, asst, user, asst, user, asst
        points = sm.get_fork_points(parent.id)
        assert all(isinstance(p, ForkInfo) for p in points)
        # 6 alternating messages → 3 user messages
        assert len(points) == 3

    def test_correct_indices(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        points = sm.get_fork_points(parent.id)
        assert [p.index for p in points] == [0, 2, 4]

    def test_content_preview(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=[{"role": "user", "content": "hello world"}])
        points = sm.get_fork_points(parent.id)
        assert points[0].content == "hello world"

    def test_content_truncated_at_80_chars(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        long_msg = "x" * 200
        sm.save(parent, messages=[{"role": "user", "content": long_msg}])
        points = sm.get_fork_points(parent.id)
        assert len(points[0].content) <= 80

    def test_newlines_replaced_in_preview(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=[{"role": "user", "content": "line1\nline2\nline3"}])
        points = sm.get_fork_points(parent.id)
        assert "\n" not in points[0].content

    def test_empty_session_returns_empty(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        points = sm.get_fork_points(parent.id)
        assert points == []

    def test_no_user_messages_returns_empty(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=[
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "result"},
        ])
        points = sm.get_fork_points(parent.id)
        assert points == []


class TestListBranches:
    def test_returns_direct_forks(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        sm.fork(parent.id, fork_index=1)
        sm.fork(parent.id, fork_index=3)
        branches = sm.list_branches(parent.id)
        assert len(branches) == 2

    def test_returns_empty_for_no_forks(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        assert sm.list_branches(parent.id) == []

    def test_does_not_include_unrelated_sessions(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        other = sm.new_session(_cfg())   # not a fork
        branches = sm.list_branches(parent.id)
        assert all(b.id != other.id for b in branches)

    def test_does_not_include_grandchildren(self, sm: SessionManager):
        """list_branches is shallow — only direct children."""
        root = sm.new_session(_cfg())
        sm.save(root, messages=_messages(8))
        child = sm.fork(root.id, fork_index=3)
        sm.save(child, messages=child.messages + [{"role": "user", "content": "x"}])
        sm.fork(child.id, fork_index=2)          # grandchild of root
        branches = sm.list_branches(root.id)
        assert len(branches) == 1
        assert branches[0].id == child.id

    def test_branch_meta_has_parent_id(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        sm.fork(parent.id, fork_index=1)
        branches = sm.list_branches(parent.id)
        assert branches[0].parent_id == parent.id

    def test_branch_meta_has_fork_index(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        sm.fork(parent.id, fork_index=2)
        branches = sm.list_branches(parent.id)
        assert branches[0].fork_index == 2

    def test_branch_display_shows_fork_marker(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        sm.fork(parent.id, fork_index=2)
        branches = sm.list_branches(parent.id)
        display = branches[0].display()
        assert "⎇" in display or "fork" in display.lower()

    def test_non_fork_session_display_no_marker(self, sm: SessionManager):
        parent = sm.new_session(_cfg(), name="root")
        display = parent.meta.display()
        assert "⎇" not in display

    def test_branches_sorted_by_created_at(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(6))
        b1 = sm.fork(parent.id, fork_index=0)
        b2 = sm.fork(parent.id, fork_index=2)
        b3 = sm.fork(parent.id, fork_index=4)
        branches = sm.list_branches(parent.id)
        ids = [b.id for b in branches]
        assert ids == [b1.id, b2.id, b3.id]


class TestForkFieldsRoundTrip:
    def test_parent_id_round_trips(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=1)
        loaded = sm.load(forked.id)
        assert loaded.parent_id == parent.id

    def test_fork_index_round_trips(self, sm: SessionManager):
        parent = sm.new_session(_cfg())
        sm.save(parent, messages=_messages(4))
        forked = sm.fork(parent.id, fork_index=2)
        loaded = sm.load(forked.id)
        assert loaded.fork_index == 2

    def test_root_session_parent_id_is_none(self, sm: SessionManager):
        session = sm.new_session(_cfg())
        loaded = sm.load(session.id)
        assert loaded.parent_id is None
        assert loaded.fork_index is None

    def test_old_sessions_without_fork_fields_load_fine(self, sm: SessionManager, tmp_path: Path):
        """Sessions saved before branching was added still deserialise cleanly."""
        data = {
            "id": "oldsess1",
            "name": None,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "config": {
                "provider": "openai", "model": "gpt-4o",
                "max_tokens": 8192, "max_turns": 20,
                "system_prompt": "", "trim_strategy": "sliding_window",
                "workspace_root": ".",
            },
            "messages": [],
            "compactions": [],
            # NOTE: no parent_id / fork_index keys
        }
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "oldsess1.json").write_text(json.dumps(data))
        sm2 = SessionManager(sessions_dir=sessions_dir)
        loaded = sm2.load("oldsess1")
        assert loaded.parent_id is None
        assert loaded.fork_index is None
