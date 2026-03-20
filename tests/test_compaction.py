"""Tests for auto-compaction (Feature #1)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tau.core.agent import Agent
from tau.core.context import Compactor, ContextManager, _messages_tokens
from tau.core.session import Session, SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    CompactionEntry,
    CompactionEvent,
    ErrorEvent,
    ProviderResponse,
    TextDelta,
    TokenUsage,
    TurnComplete,
    Message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> AgentConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        max_tokens=1000,
        max_turns=5,
        system_prompt="sys",
        compaction_enabled=True,
        compaction_threshold=0.80,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_agent(
    provider_responses: list,
    cfg: AgentConfig | None = None,
    sm: SessionManager | None = None,
) -> Agent:
    config = cfg or _cfg()
    context = ContextManager(config)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    if sm is None:
        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None

    session = MagicMock(spec=Session)
    session.id = "test-session"
    session.messages = []
    session.compactions = []

    return Agent(
        config=config,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )


def _big_message(n: int = 400) -> Message:
    """Return a user message with roughly n tokens worth of content."""
    return Message(role="user", content="word " * n)


def _final_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ===========================================================================
# Compactor unit tests
# ===========================================================================

class TestCompactorShouldCompact:
    def test_disabled_never_compacts(self):
        cfg = _cfg(compaction_enabled=False)
        c = Compactor(cfg)
        msgs = [_big_message(900)]
        assert c.should_compact(msgs) is False

    def test_below_threshold_no_compact(self):
        cfg = _cfg(max_tokens=1000, compaction_threshold=0.80)
        c = Compactor(cfg)
        msgs = [Message(role="user", content="short")]
        assert c.should_compact(msgs) is False

    def test_at_threshold_triggers_compact(self):
        cfg = _cfg(max_tokens=100, compaction_threshold=0.80)
        c = Compactor(cfg)
        # Fill to exactly the threshold (80 tokens)
        msgs = [Message(role="user", content="word " * 80)]
        assert c.should_compact(msgs) is True

    def test_above_threshold_triggers_compact(self):
        cfg = _cfg(max_tokens=100, compaction_threshold=0.50)
        c = Compactor(cfg)
        msgs = [Message(role="user", content="word " * 200)]
        assert c.should_compact(msgs) is True


class TestCompactorIsOverflowError:
    def setup_method(self):
        self.c = Compactor(_cfg())

    @pytest.mark.parametrize("msg", [
        "context_length_exceeded",
        "This model's maximum context length is 8192 tokens",
        "context window full",
        "too many tokens in your request",
        "Please reduce the length of the messages",
        "Please reduce your prompt",
        "The prompt is too long",
    ])
    def test_recognises_overflow_messages(self, msg):
        assert self.c.is_overflow_error(msg) is True

    def test_ignores_non_overflow_errors(self):
        assert self.c.is_overflow_error("network timeout") is False
        assert self.c.is_overflow_error("invalid api key") is False
        assert self.c.is_overflow_error("rate limit exceeded") is False


class TestCompactorCompact:
    def _make_provider(self, summary: str = "summary text") -> MagicMock:
        provider = MagicMock()
        provider.chat.return_value = ProviderResponse(
            content=summary,
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )
        return provider

    def _make_messages(self, n: int) -> list[Message]:
        system = Message(role="system", content="sys")
        non_system = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(n)
        ]
        return [system] + non_system

    def test_returns_new_messages_and_entry(self):
        provider = self._make_provider("nice summary")
        c = Compactor(_cfg())
        msgs = self._make_messages(10)
        new_msgs, entry = c.compact(msgs, provider, tokens_before=500)

        assert isinstance(entry, CompactionEntry)
        assert entry.summary == "nice summary"
        assert entry.tokens_before == 500
        assert entry.timestamp  # non-empty ISO string

    def test_system_prompt_preserved(self):
        provider = self._make_provider()
        c = Compactor(_cfg())
        msgs = self._make_messages(10)
        new_msgs, _ = c.compact(msgs, provider, tokens_before=500)

        assert new_msgs[0].role == "system"

    def test_summary_message_present(self):
        provider = self._make_provider("my summary")
        c = Compactor(_cfg())
        msgs = self._make_messages(10)
        new_msgs, _ = c.compact(msgs, provider, tokens_before=500)

        contents = [m.content for m in new_msgs]
        assert any("my summary" in c for c in contents)

    def test_recent_messages_kept_verbatim(self):
        provider = self._make_provider()
        c = Compactor(_cfg())
        msgs = self._make_messages(12)  # 11 non-system + system
        new_msgs, _ = c.compact(msgs, provider, tokens_before=500)

        # Last 4 non-system messages should appear unchanged
        last_four = [m.content for m in msgs if m.role != "system"][-4:]
        compacted_contents = [m.content for m in new_msgs]
        for text in last_four:
            assert text in compacted_contents

    def test_raises_when_too_few_messages(self):
        provider = self._make_provider()
        c = Compactor(_cfg())
        msgs = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        with pytest.raises(ValueError, match="Not enough messages"):
            c.compact(msgs, provider, tokens_before=100)

    def test_tokens_reduced_after_compact(self):
        provider = self._make_provider("short summary")
        c = Compactor(_cfg())
        msgs = self._make_messages(20)
        tokens_before = _messages_tokens(msgs)
        new_msgs, _ = c.compact(msgs, provider, tokens_before=tokens_before)
        assert _messages_tokens(new_msgs) < tokens_before

    def test_streaming_provider_response_collected(self):
        """Compactor handles a streaming provider (yields TextDelta + ProviderResponse)."""
        def _stream():
            yield TextDelta(text="streamed ")
            yield TextDelta(text="summary")
            yield ProviderResponse(content=None, tool_calls=[], stop_reason="end_turn", usage=TokenUsage())

        provider = MagicMock()
        provider.chat.return_value = _stream()

        c = Compactor(_cfg())
        msgs = [Message(role="system", content="sys")] + [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(10)
        ]
        new_msgs, entry = c.compact(msgs, provider, tokens_before=300)
        assert "streamed summary" in entry.summary

    def test_provider_failure_falls_back_to_transcript(self):
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("api down")
        c = Compactor(_cfg())
        msgs = self._make_messages(10)
        # Should not raise — falls back to truncated transcript
        new_msgs, entry = c.compact(msgs, provider, tokens_before=500)
        assert entry.summary  # non-empty fallback


class TestCompactorOverflowFlags:
    def test_flag_starts_false(self):
        c = Compactor(_cfg())
        assert c.overflow_recovery_attempted() is False

    def test_mark_and_check(self):
        c = Compactor(_cfg())
        c.mark_overflow_recovery_attempted()
        assert c.overflow_recovery_attempted() is True

    def test_reset_clears_flag(self):
        c = Compactor(_cfg())
        c.mark_overflow_recovery_attempted()
        c.reset_overflow_flag()
        assert c.overflow_recovery_attempted() is False


# ===========================================================================
# Agent — threshold-based auto-compaction
# ===========================================================================

class TestAgentThresholdCompaction:
    def _provider_with_summary(self, summary: str = "compacted") -> MagicMock:
        """Provider whose second call returns a compaction summary."""
        p = MagicMock()
        p.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content=summary, tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        return p

    def _agent_near_threshold(self, sm=None) -> Agent:
        """
        Build an agent pre-loaded with enough messages to exceed the threshold.
        max_tokens=10000 so trim() won't destroy history;
        compaction_threshold=0.001 so even a handful of tokens triggers it.
        """
        cfg = _cfg(max_tokens=10000, compaction_threshold=0.001)
        agent = _make_agent([], cfg=cfg, sm=sm)
        for i in range(10):
            agent._context.add_message(
                Message(role="user" if i % 2 == 0 else "assistant", content=f"message {i} " * 5)
            )
        return agent

    def test_compaction_event_start_emitted(self):
        agent = self._agent_near_threshold()
        provider = MagicMock()
        provider.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        agent._provider = provider

        events = list(agent.run("go"))
        start_events = [e for e in events if isinstance(e, CompactionEvent) and e.stage == "start"]
        assert start_events, "Expected at least one CompactionEvent(stage='start')"

    def test_compaction_event_end_emitted(self):
        agent = self._agent_near_threshold()
        provider = MagicMock()
        provider.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        agent._provider = provider

        events = list(agent.run("go"))
        end_events = [e for e in events if isinstance(e, CompactionEvent) and e.stage == "end"]
        assert end_events, "Expected at least one CompactionEvent(stage='end')"

    def test_compaction_end_has_token_counts(self):
        agent = self._agent_near_threshold()
        provider = MagicMock()
        provider.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content="summary text", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        agent._provider = provider

        events = list(agent.run("go"))
        end = next(e for e in events if isinstance(e, CompactionEvent) and e.stage == "end")
        assert end.tokens_before > 0
        assert end.tokens_after >= 0

    def test_compaction_persisted_to_session(self):
        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None

        agent = self._agent_near_threshold(sm=sm)
        provider = MagicMock()
        provider.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        agent._provider = provider

        list(agent.run("go"))
        sm.append_compaction.assert_called_once()
        _, entry = sm.append_compaction.call_args[0]
        assert isinstance(entry, CompactionEntry)
        assert entry.summary == "summary"

    def test_no_compaction_when_disabled(self):
        cfg = _cfg(max_tokens=10000, compaction_threshold=0.001, compaction_enabled=False)
        agent = _make_agent([_final_response("answer")], cfg=cfg)
        for i in range(10):
            agent._context.add_message(
                Message(role="user" if i % 2 == 0 else "assistant", content=f"message {i} " * 5)
            )

        events = list(agent.run("go"))
        compaction_events = [e for e in events if isinstance(e, CompactionEvent)]
        assert not compaction_events

    def test_context_smaller_after_compaction(self):
        agent = self._agent_near_threshold()
        tokens_before_run = agent._context.token_count()

        provider = MagicMock()
        provider.chat.side_effect = [
            _final_response("answer"),
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ]
        agent._provider = provider

        list(agent.run("go"))
        assert agent._context.token_count() < tokens_before_run


# ===========================================================================
# Agent — overflow error recovery
# ===========================================================================

class TestAgentOverflowRecovery:
    def test_overflow_triggers_compaction_and_retry(self):
        """On overflow error, agent compacts and retries → gets a final answer."""
        cfg = _cfg()
        provider = MagicMock()
        provider.chat.side_effect = [
            RuntimeError("context_length_exceeded: too many tokens"),  # 1st: overflow
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),  # compaction
            _final_response("recovered answer"),  # retry
        ]

        agent = _make_agent([], cfg=cfg)
        agent._provider = provider
        # Add enough history for compaction to succeed
        for i in range(10):
            agent._context.add_message(Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"))

        events = list(agent.run("go"))

        compaction_events = [e for e in events if isinstance(e, CompactionEvent)]
        assert any(e.stage == "start" for e in compaction_events)
        assert any(e.stage == "end" for e in compaction_events)
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_overflow_recovery_only_once(self):
        """If overflow recurs after compaction, agent emits ErrorEvent and stops."""
        cfg = _cfg()
        provider = MagicMock()
        provider.chat.side_effect = [
            RuntimeError("context_length_exceeded"),              # 1st: overflow
            ProviderResponse(content="summary", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),  # compaction summary
            RuntimeError("context_length_exceeded"),              # retry still overflows
        ]

        agent = _make_agent([], cfg=cfg)
        agent._provider = provider
        for i in range(10):
            agent._context.add_message(Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"))

        events = list(agent.run("go"))
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert error_events
        assert "compaction already attempted" in error_events[-1].message.lower() or \
               "overflow" in error_events[-1].message.lower()

    def test_non_overflow_error_not_compacted(self):
        """A regular provider error does NOT trigger compaction."""
        cfg = _cfg()
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("network timeout")

        agent = _make_agent([], cfg=cfg)
        agent._provider = provider

        events = list(agent.run("go"))
        compaction_events = [e for e in events if isinstance(e, CompactionEvent)]
        assert not compaction_events
        assert any(isinstance(e, ErrorEvent) and "network timeout" in e.message for e in events)

    def test_overflow_flag_reset_between_runs(self):
        """overflow_recovery_attempted is reset at the start of each run() call."""
        cfg = _cfg()
        agent = _make_agent([_final_response("ok")], cfg=cfg)
        agent._context.compactor.mark_overflow_recovery_attempted()

        list(agent.run("first run"))
        # Flag should have been reset at the start of run()
        assert agent._context.compactor.overflow_recovery_attempted() is False


# ===========================================================================
# Session — compaction persistence
# ===========================================================================

class TestSessionCompactionPersistence:
    def test_append_compaction_stored_in_session(self, tmp_path):
        from tau.core.session import SessionManager
        sm = SessionManager(sessions_dir=tmp_path)
        cfg = _cfg()
        session = sm.new_session(cfg)

        entry = CompactionEntry(
            summary="everything was summarised",
            tokens_before=8000,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        sm.append_compaction(session, entry)

        assert len(session.compactions) == 1
        assert session.compactions[0]["summary"] == "everything was summarised"
        assert session.compactions[0]["tokens_before"] == 8000

    def test_append_compaction_persisted_to_disk(self, tmp_path):
        from tau.core.session import SessionManager
        import json
        sm = SessionManager(sessions_dir=tmp_path)
        cfg = _cfg()
        session = sm.new_session(cfg)

        entry = CompactionEntry(summary="disk summary", tokens_before=5000, timestamp="2024-01-01T00:00:00+00:00")
        sm.append_compaction(session, entry)

        data = json.loads((tmp_path / f"{session.id}.json").read_text())
        assert len(data["compactions"]) == 1
        assert data["compactions"][0]["summary"] == "disk summary"

    def test_multiple_compactions_appended(self, tmp_path):
        from tau.core.session import SessionManager
        sm = SessionManager(sessions_dir=tmp_path)
        session = sm.new_session(_cfg())

        for i in range(3):
            sm.append_compaction(session, CompactionEntry(
                summary=f"summary {i}", tokens_before=1000 * i, timestamp="2024-01-01T00:00:00+00:00"
            ))

        assert len(session.compactions) == 3

    def test_compactions_round_trip_via_load(self, tmp_path):
        from tau.core.session import SessionManager
        sm = SessionManager(sessions_dir=tmp_path)
        session = sm.new_session(_cfg())
        sm.append_compaction(session, CompactionEntry(
            summary="round trip", tokens_before=999, timestamp="2024-01-01T00:00:00+00:00"
        ))

        loaded = sm.load(session.id)
        assert len(loaded.compactions) == 1
        assert loaded.compactions[0]["summary"] == "round trip"

    def test_old_sessions_without_compactions_load_fine(self, tmp_path):
        """Sessions saved before the compactions field was added still load."""
        import json
        session_data = {
            "id": "abc123",
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
            # NOTE: no "compactions" key
        }
        (tmp_path / "abc123.json").write_text(json.dumps(session_data))
        from tau.core.session import SessionManager
        sm = SessionManager(sessions_dir=tmp_path)
        session = sm.load("abc123")
        assert session.compactions == []
