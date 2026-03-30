"""Tests for /share command and share_session() helper."""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import tau.cli as _cli_mod
from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import Session, SessionManager, share_session, export_session_markdown
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig, ProviderResponse, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> AgentConfig:
    defaults = dict(
        provider="openai", model="gpt-4o", max_tokens=8192, max_turns=10,
        system_prompt="sys", compaction_enabled=False, retry_enabled=False,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _make_session(messages: list[dict] | None = None) -> Session:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    s = Session(
        id="test-share-session",
        name="Test Share",
        created_at=now,
        updated_at=now,
        config=_cfg(),
        messages=messages or [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    )
    return s


def _make_agent() -> Agent:
    config = _cfg()
    context = ContextManager(config)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.return_value = ProviderResponse(
        content="ok", tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )
    sm = MagicMock(spec=SessionManager)
    sm.save.return_value = None
    session = MagicMock(spec=Session)
    session.id = "test-session"
    session.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    session.compactions = []
    steering = SteeringChannel()
    return Agent(
        config=config, provider=provider, registry=registry,
        context=context, session=session, session_manager=sm,
        steering=steering,
    )


def _mock_urlopen(url: str = "https://paste.rs/abc123"):
    """Return a context manager mock that yields a response reading *url*."""
    resp = MagicMock()
    resp.read.return_value = url.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _call_share(cmd: str, agent: Agent | None = None) -> tuple[bool, list]:
    prints: list = []
    steering = agent._steering if agent else SteeringChannel()
    with patch.object(_cli_mod, "console") as mock_console:
        mock_console.print.side_effect = lambda *a, **kw: prints.append(a[0] if a else None)
        handled = _cli_mod._handle_slash(
            cmd,
            steering=steering,
            ext_registry=None,
            ext_context=None,
            agent=agent,
        )
    return handled, prints


# ===========================================================================
# share_session() unit tests
# ===========================================================================

class TestShareSession:
    def test_returns_url(self):
        session = _make_session()
        expected_url = "https://paste.rs/xyz"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(expected_url)):
            url = share_session(session)
        assert url == expected_url

    def test_default_format_is_markdown(self):
        """Default format should produce markdown content (not JSON)."""
        session = _make_session()
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.data.decode("utf-8"))
            return _mock_urlopen()
        
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            share_session(session)

        assert len(captured) == 1
        # Markdown should contain # Session: header
        assert "# Session:" in captured[0]
        assert "hello" in captured[0]

    def test_json_format(self):
        """--json flag should upload JSON."""
        session = _make_session()
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.data.decode("utf-8"))
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            share_session(session, fmt="json")

        assert len(captured) == 1
        parsed = json.loads(captured[0])
        assert "messages" in parsed
        assert parsed["id"] == "test-share-session"

    def test_posts_to_paste_rs(self):
        """Request must go to paste.rs."""
        session = _make_session()
        captured_reqs = []

        def fake_urlopen(req, timeout=None):
            captured_reqs.append(req)
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            share_session(session)

        assert len(captured_reqs) == 1
        assert "paste.rs" in captured_reqs[0].full_url

    def test_uses_post_method(self):
        session = _make_session()
        captured_reqs = []

        def fake_urlopen(req, timeout=None):
            captured_reqs.append(req)
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            share_session(session)

        assert captured_reqs[0].method == "POST"

    def test_propagates_network_error(self):
        import urllib.error
        session = _make_session()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with pytest.raises(urllib.error.URLError):
                share_session(session)


# ===========================================================================
# /share slash command tests
# ===========================================================================

class TestSlashShare:
    def test_returns_true(self):
        agent = _make_agent()
        with patch("tau.core.session.share_session", return_value="https://paste.rs/ok"):
            with patch("subprocess.run"):
                handled, _ = _call_share("/share", agent)
        assert handled is True

    def test_no_agent(self):
        handled, prints = _call_share("/share", None)
        assert handled is True
        assert any("requires" in str(p) for p in prints)

    def test_prints_url_on_success(self):
        agent = _make_agent()
        with patch("tau.core.session.share_session", return_value="https://paste.rs/abc"):
            with patch("subprocess.run"):
                handled, prints = _call_share("/share", agent)
        assert handled is True
        output = " ".join(str(p) for p in prints)
        assert "paste.rs/abc" in output

    def test_json_flag(self):
        """'/share --json' should call share_session with fmt='json'."""
        agent = _make_agent()
        calls = []

        def fake_share(session, fmt="markdown"):
            calls.append(fmt)
            return "https://paste.rs/json123"

        with patch("tau.core.session.share_session", side_effect=fake_share):
            with patch("subprocess.run"):
                _call_share("/share --json", agent)

        assert calls == ["json"]

    def test_default_is_markdown(self):
        agent = _make_agent()
        calls = []

        def fake_share(session, fmt="markdown"):
            calls.append(fmt)
            return "https://paste.rs/md123"

        with patch("tau.core.session.share_session", side_effect=fake_share):
            with patch("subprocess.run"):
                _call_share("/share", agent)

        assert calls == ["markdown"]

    def test_handles_network_error_gracefully(self):
        agent = _make_agent()
        import urllib.error
        with patch("tau.core.session.share_session",
                   side_effect=urllib.error.URLError("connection refused")):
            handled, prints = _call_share("/share", agent)
        assert handled is True
        output = " ".join(str(p) for p in prints)
        assert "fail" in output.lower() or "share failed" in output.lower()

    def test_share_in_slash_help(self):
        handled, prints = _call_share("/help", None)
        assert handled is True
        # /help renders a Rich Panel; check the renderable content
        panel = prints[0]
        assert "/share" in str(panel.renderable)


    def test_share_in_builtin_commands(self):
        from tau.editor import BUILTIN_SLASH_COMMANDS
        assert "share" in BUILTIN_SLASH_COMMANDS
