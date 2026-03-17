"""Tests for tau.core.session."""

import pytest
from pathlib import Path
from tau.core.session import SessionManager, SessionNotFoundError
from tau.core.types import AgentConfig


@pytest.fixture()
def sm(tmp_path: Path) -> SessionManager:
    return SessionManager(sessions_dir=tmp_path / "sessions")


def _cfg() -> AgentConfig:
    return AgentConfig(provider="openai", model="gpt-4o")


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
