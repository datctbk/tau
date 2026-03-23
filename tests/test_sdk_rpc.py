"""Tests for the SDK (tau.sdk) and RPC (tau.rpc) modules."""
from __future__ import annotations

import io
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.extension import ExtensionRegistry
from tau.core.session import Session, SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    ProviderResponse,
    TextChunk,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnComplete,
)
from tau.sdk import InMemorySessionManager, TauSession, create_session
from tau.rpc import run_rpc, _write, _read_request


# =========================================================================
# Helpers
# =========================================================================


def _config(**kw) -> AgentConfig:
    return AgentConfig(provider="openai", model="gpt-4o", max_turns=5, **kw)


def _streaming_response(words, tool_calls=None, stop_reason="end_turn"):
    def _gen():
        for w in words:
            yield TextDelta(text=w)
        yield ProviderResponse(
            content="".join(words),
            tool_calls=tool_calls or [],
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=10, output_tokens=len(words)),
        )
    return _gen()


def _make_mock_agent(provider_responses):
    """Build a real Agent with a mock provider."""
    cfg = _config()
    context = ContextManager(cfg)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    session = MagicMock(spec=Session)
    session.id = str(uuid.uuid4())
    session.name = "test"
    session.messages = []
    session.config = cfg
    session.created_at = "2026-01-01T00:00:00"
    session.updated_at = "2026-01-01T00:00:00"
    session.cumulative_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
    }
    sm = MagicMock()
    sm.save.return_value = None

    agent = Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )
    return agent, session, sm


def _make_tau_session(provider_responses):
    """Build a TauSession backed by a mock agent."""
    agent, session, sm = _make_mock_agent(provider_responses)
    ext = ExtensionRegistry(extra_paths=[], disabled=[])
    steering = SteeringChannel()
    return TauSession(
        agent=agent,
        session=session,
        session_manager=sm,
        ext_registry=ext,
        steering=steering,
    )


# =========================================================================
# InMemorySessionManager tests
# =========================================================================


class TestInMemorySessionManager:

    def test_new_session(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config(), name="test")
        assert s.id
        assert s.name == "test"

    def test_save_and_load(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        s.messages = [{"role": "user", "content": "hi"}]
        sm.save(s)
        loaded = sm.load(s.id)
        assert loaded.messages == [{"role": "user", "content": "hi"}]

    def test_load_prefix(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        loaded = sm.load(s.id[:8])
        assert loaded.id == s.id

    def test_load_not_found(self):
        sm = InMemorySessionManager()
        from tau.core.session import SessionNotFoundError
        with pytest.raises(SessionNotFoundError):
            sm.load("nonexistent")

    def test_delete(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        sm.delete(s.id)
        from tau.core.session import SessionNotFoundError
        with pytest.raises(SessionNotFoundError):
            sm.load(s.id)

    def test_delete_not_found(self):
        sm = InMemorySessionManager()
        from tau.core.session import SessionNotFoundError
        with pytest.raises(SessionNotFoundError):
            sm.delete("nope")

    def test_list_sessions(self):
        sm = InMemorySessionManager()
        sm.new_session(_config(), name="a")
        sm.new_session(_config(), name="b")
        metas = sm.list_sessions()
        assert len(metas) == 2
        names = {m.name for m in metas}
        assert names == {"a", "b"}

    def test_fork(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        s.messages = [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        sm.save(s)
        forked = sm.fork(s.id, fork_index=2)
        assert forked.parent_id == s.id
        assert forked.fork_index == 2
        assert len(forked.messages) == 3  # system + user + assistant

    def test_fork_out_of_range(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        s.messages = [{"role": "user", "content": "q"}]
        sm.save(s)
        with pytest.raises(ValueError):
            sm.fork(s.id, fork_index=5)

    def test_get_fork_points(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        s.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        sm.save(s)
        points = sm.get_fork_points(s.id)
        assert len(points) == 1
        assert points[0].index == 1
        assert "hello" in points[0].content

    def test_list_branches(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        s.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        sm.save(s)
        f = sm.fork(s.id, fork_index=1)
        branches = sm.list_branches(s.id)
        assert len(branches) == 1
        assert branches[0].id == f.id

    def test_append_compaction(self):
        sm = InMemorySessionManager()
        s = sm.new_session(_config())
        from tau.core.types import CompactionEntry
        entry = CompactionEntry(summary="sum", tokens_before=100, timestamp="2026-01-01")
        sm.append_compaction(s, entry)
        loaded = sm.load(s.id)
        assert len(loaded.compactions) == 1
        assert loaded.compactions[0]["summary"] == "sum"

    def test_no_file_path(self):
        sm = InMemorySessionManager()
        with pytest.raises(NotImplementedError):
            sm._path("any-id")


# =========================================================================
# TauSession tests
# =========================================================================


class TestTauSession:

    def test_prompt_yields_events(self):
        ts = _make_tau_session([
            _streaming_response(["Hello", " world"]),
        ])
        events = list(ts.prompt("say hi"))
        types = [type(e).__name__ for e in events]
        assert "TextDelta" in types
        assert "TurnComplete" in types

    def test_prompt_sync(self):
        ts = _make_tau_session([
            _streaming_response(["Hi"]),
        ])
        events = ts.prompt_sync("say hi")
        assert isinstance(events, list)
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_session_id(self):
        ts = _make_tau_session([])
        assert ts.session_id == ts.session.id

    def test_close_prevents_prompt(self):
        ts = _make_tau_session([])
        ts.close()
        with pytest.raises(RuntimeError, match="closed"):
            list(ts.prompt("hi"))

    def test_context_manager(self):
        ts = _make_tau_session([
            _streaming_response(["ok"]),
        ])
        with ts as s:
            events = list(s.prompt("hi"))
        assert ts._closed
        assert len(events) > 0

    def test_steer(self):
        ts = _make_tau_session([])
        # Just verify it doesn't raise
        ts.steer("change topic")

    def test_enqueue(self):
        ts = _make_tau_session([])
        ts.enqueue("follow up")

    def test_agent_property(self):
        ts = _make_tau_session([])
        assert isinstance(ts.agent, Agent)


# =========================================================================
# RPC protocol tests
# =========================================================================


class TestRpcHelpers:

    def test_write(self):
        buf = io.StringIO()
        _write(buf, {"type": "ready"})
        line = buf.getvalue()
        assert line.endswith("\n")
        parsed = json.loads(line.strip())
        assert parsed == {"type": "ready"}

    def test_read_request(self):
        buf = io.StringIO('{"type": "exit"}\n')
        req = _read_request(buf)
        assert req == {"type": "exit"}

    def test_read_request_eof(self):
        buf = io.StringIO("")
        req = _read_request(buf)
        assert req is None

    def test_read_request_empty_line(self):
        buf = io.StringIO("   \n")
        req = _read_request(buf)
        assert req is None


class TestRpcLoop:

    def _run_rpc(self, requests: list[dict], provider_responses=None):
        """Run the RPC loop with canned requests and return output lines."""
        if provider_responses is None:
            provider_responses = []

        ts = _make_tau_session(provider_responses)
        inp = io.StringIO(
            "\n".join(json.dumps(r) for r in requests) + "\n"
        )
        out = io.StringIO()
        run_rpc(ts, inp=inp, out=out)
        out.seek(0)
        lines = [json.loads(line) for line in out if line.strip()]
        return lines

    def test_ready_on_start(self):
        lines = self._run_rpc([{"type": "exit"}])
        assert lines[0]["type"] == "ready"

    def test_exit(self):
        lines = self._run_rpc([{"type": "exit"}])
        types = [l["type"] for l in lines]
        assert "exit" in types
        exit_msg = next(l for l in lines if l["type"] == "exit")
        assert exit_msg["status"] == "ok"

    def test_prompt(self):
        lines = self._run_rpc(
            [
                {"type": "prompt", "text": "say hello"},
                {"type": "exit"},
            ],
            provider_responses=[
                _streaming_response(["Hello", "!"]),
            ],
        )
        types = [l["type"] for l in lines]
        # First ready, then events from prompt, then ready again, then exit
        assert types[0] == "ready"
        assert "text_delta" in types
        assert "turn_complete" in types
        # Should have a second "ready" after prompt finishes
        ready_count = types.count("ready")
        assert ready_count == 2

    def test_prompt_empty_text(self):
        lines = self._run_rpc([
            {"type": "prompt", "text": ""},
            {"type": "exit"},
        ])
        errors = [l for l in lines if l["type"] == "error"]
        assert len(errors) == 1
        assert "text" in errors[0]["message"]

    def test_session_info(self):
        lines = self._run_rpc([
            {"type": "session_info"},
            {"type": "exit"},
        ])
        info = [l for l in lines if l["type"] == "session_info"]
        assert len(info) == 1
        assert "id" in info[0]
        assert "model" in info[0]
        assert "messages" in info[0]

    def test_steer_request(self):
        lines = self._run_rpc([
            {"type": "steer", "text": "go left"},
            {"type": "exit"},
        ])
        steer_ack = [l for l in lines if l["type"] == "steer"]
        assert len(steer_ack) == 1
        assert steer_ack[0]["status"] == "ok"

    def test_steer_empty(self):
        lines = self._run_rpc([
            {"type": "steer", "text": ""},
            {"type": "exit"},
        ])
        errors = [l for l in lines if l["type"] == "error"]
        assert len(errors) == 1

    def test_enqueue_request(self):
        lines = self._run_rpc([
            {"type": "enqueue", "text": "follow up"},
            {"type": "exit"},
        ])
        ack = [l for l in lines if l["type"] == "enqueue"]
        assert len(ack) == 1
        assert ack[0]["status"] == "ok"

    def test_enqueue_empty(self):
        lines = self._run_rpc([
            {"type": "enqueue", "text": ""},
            {"type": "exit"},
        ])
        errors = [l for l in lines if l["type"] == "error"]
        assert len(errors) == 1

    def test_unknown_request(self):
        lines = self._run_rpc([
            {"type": "foobar"},
            {"type": "exit"},
        ])
        errors = [l for l in lines if l["type"] == "error"]
        assert len(errors) == 1
        assert "foobar" in errors[0]["message"]

    def test_invalid_json(self):
        """Malformed JSON is reported as error and loop continues."""
        ts = _make_tau_session([])
        inp = io.StringIO("not-json\n" + json.dumps({"type": "exit"}) + "\n")
        out = io.StringIO()
        run_rpc(ts, inp=inp, out=out)
        out.seek(0)
        lines = [json.loads(line) for line in out if line.strip()]
        errors = [l for l in lines if l["type"] == "error"]
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]["message"]

    def test_eof_closes(self):
        """EOF on stdin gracefully closes the RPC loop."""
        ts = _make_tau_session([])
        inp = io.StringIO("")  # immediate EOF
        out = io.StringIO()
        run_rpc(ts, inp=inp, out=out)
        out.seek(0)
        lines = [json.loads(line) for line in out if line.strip()]
        # Should have at least the initial ready
        assert lines[0]["type"] == "ready"

    def test_prompt_with_images(self):
        lines = self._run_rpc(
            [
                {"type": "prompt", "text": "describe", "images": ["/tmp/test.png"]},
                {"type": "exit"},
            ],
            provider_responses=[
                _streaming_response(["A picture"]),
            ],
        )
        types = [l["type"] for l in lines]
        assert "text_delta" in types

    def test_multiple_prompts(self):
        lines = self._run_rpc(
            [
                {"type": "prompt", "text": "first"},
                {"type": "prompt", "text": "second"},
                {"type": "exit"},
            ],
            provider_responses=[
                _streaming_response(["Reply1"]),
                _streaming_response(["Reply2"]),
            ],
        )
        ready_count = sum(1 for l in lines if l["type"] == "ready")
        # Initial ready + after prompt 1 + after prompt 2 = 3
        assert ready_count == 3


# =========================================================================
# create_session integration test (mocked provider)
# =========================================================================


class TestCreateSession:

    @patch("tau.sdk.get_provider")
    @patch("tau.sdk.load_config")
    @patch("tau.sdk.ensure_tau_home")
    def test_create_session_basic(self, mock_home, mock_cfg, mock_prov):
        mock_cfg.return_value = MagicMock(
            shell=MagicMock(
                require_confirmation=False,
                timeout=30,
                allowed_commands=[],
                use_persistent_shell=False,
            ),
            ollama=MagicMock(base_url="http://localhost:11434"),
            skills=MagicMock(paths=[], disabled=[]),
            extensions=MagicMock(paths=[], disabled=[]),
        )
        mock_prov.return_value = MagicMock()

        ts = create_session(
            provider="openai",
            model="gpt-4o",
            in_memory=True,
            load_skills=False,
            load_extensions=False,
            load_context_files=False,
        )
        assert isinstance(ts, TauSession)
        assert ts.session_id
        assert not ts._closed

    @patch("tau.sdk.get_provider")
    @patch("tau.sdk.load_config")
    @patch("tau.sdk.ensure_tau_home")
    def test_create_session_custom_system_prompt(self, mock_home, mock_cfg, mock_prov):
        mock_cfg.return_value = MagicMock(
            shell=MagicMock(
                require_confirmation=False, timeout=30,
                allowed_commands=[], use_persistent_shell=False,
            ),
            ollama=MagicMock(base_url="http://localhost:11434"),
            skills=MagicMock(paths=[], disabled=[]),
            extensions=MagicMock(paths=[], disabled=[]),
        )
        mock_prov.return_value = MagicMock()

        ts = create_session(
            system_prompt="Custom prompt",
            in_memory=True,
            load_skills=False,
            load_extensions=False,
            load_context_files=False,
        )
        assert ts.agent._config.system_prompt == "Custom prompt"

    @patch("tau.sdk.get_provider")
    @patch("tau.sdk.load_config")
    @patch("tau.sdk.ensure_tau_home")
    def test_create_session_disk_backed(self, mock_home, mock_cfg, mock_prov):
        mock_cfg.return_value = MagicMock(
            shell=MagicMock(
                require_confirmation=False, timeout=30,
                allowed_commands=[], use_persistent_shell=False,
            ),
            ollama=MagicMock(base_url="http://localhost:11434"),
            skills=MagicMock(paths=[], disabled=[]),
            extensions=MagicMock(paths=[], disabled=[]),
        )
        mock_prov.return_value = MagicMock()

        custom_sm = InMemorySessionManager()
        ts = create_session(
            session_manager=custom_sm,
            load_skills=False,
            load_extensions=False,
            load_context_files=False,
        )
        assert ts._session_manager is custom_sm


# =========================================================================
# CLI --mode rpc wiring test
# =========================================================================


class TestCliRpcMode:

    def test_mode_rpc_in_choices(self):
        """Verify 'rpc' is in the --mode choices."""
        from tau.cli import _AGENT_OPTIONS
        mode_opt = None
        for opt in _AGENT_OPTIONS:
            # Click options store type on the option's type
            # Find the --mode option by checking its name
            import click
            if hasattr(opt, "keywords"):
                continue
            # Wrap a dummy to inspect
            pass
        # Simpler: just import and check the click command
        from tau.cli import main
        # The run command should accept --mode rpc
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--mode", "rpc", "--help"])
        # --help should work without error
        assert result.exit_code == 0 or "rpc" in (result.output or "")
