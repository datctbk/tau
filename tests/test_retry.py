"""Tests for auto-retry with exponential backoff (Feature #2)."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from tau.core.agent import Agent, _RETRYABLE_PATTERNS
from tau.core.context import ContextManager
from tau.core.session import Session, SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    ProviderResponse,
    RetryEvent,
    TokenUsage,
    TurnComplete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> AgentConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        max_tokens=8192,
        max_turns=5,
        system_prompt="sys",
        retry_enabled=True,
        retry_max_attempts=3,
        retry_base_delay=0.0,   # zero delay so tests run instantly
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_agent(cfg: AgentConfig, provider_responses: list) -> Agent:
    context = ContextManager(cfg)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    sm = MagicMock(spec=SessionManager)
    sm.save.return_value = None
    sm.append_compaction.return_value = None

    session = MagicMock(spec=Session)
    session.id = "test"
    session.messages = []
    session.compactions = []

    return Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )


def _ok(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    def setup_method(self):
        cfg = _cfg()
        self.agent = _make_agent(cfg, [])

    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "rate_limit_exceeded",
        "too many requests",
        "Error 429",
        "server overloaded",
        "503 service unavailable",
        "502 bad gateway",
        "500 internal error",
        "connection timeout",
        "request timed out",
        "connection error",
        "connection refused",
        "network error",
        "fetch failed",
        "socket hang up",
        "connection reset",
        "temporarily unavailable",
    ])
    def test_retryable_errors(self, msg):
        assert self.agent._is_retryable_error(msg) is True

    @pytest.mark.parametrize("msg", [
        "invalid api key",
        "authentication failed",
        "model not found",
        "bad request",
        "context_length_exceeded",          # overflow → compaction, not retry
        "maximum context length exceeded",  # overflow
    ])
    def test_non_retryable_errors(self, msg):
        assert self.agent._is_retryable_error(msg) is False

    def test_overflow_never_retryable(self):
        """Overflow errors are explicitly excluded even though they may match generic words."""
        assert self.agent._is_retryable_error("context_length_exceeded too many tokens") is False


# ---------------------------------------------------------------------------
# Retry behaviour in the agent loop
# ---------------------------------------------------------------------------

class TestAgentRetry:

    def test_success_on_first_attempt_no_retry_event(self):
        agent = _make_agent(_cfg(), [_ok()])
        events = list(agent.run("hi"))
        retry_events = [e for e in events if isinstance(e, RetryEvent)]
        assert not retry_events
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_retries_on_transient_error_then_succeeds(self):
        agent = _make_agent(_cfg(), [
            RuntimeError("rate limit exceeded"),
            RuntimeError("503 service unavailable"),
            _ok("recovered"),
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        retry_events = [e for e in events if isinstance(e, RetryEvent)]
        assert len(retry_events) == 2
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_retry_event_fields(self):
        agent = _make_agent(_cfg(retry_max_attempts=3, retry_base_delay=2.0), [
            RuntimeError("rate limit exceeded"),
            _ok(),
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        re = next(e for e in events if isinstance(e, RetryEvent))
        assert re.attempt == 2          # about to make attempt 2
        assert re.max_attempts == 3
        assert re.delay == 2.0          # base_delay * 2^0
        assert "rate limit" in re.error.lower()

    def test_exponential_backoff_delays(self):
        """Delays double each attempt: base, base*2, base*4 …"""
        agent = _make_agent(_cfg(retry_max_attempts=4, retry_base_delay=1.0), [
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
            _ok(),
        ])
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            list(agent.run("hi"))

        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_exhausted_retries_yields_error_event(self):
        agent = _make_agent(_cfg(retry_max_attempts=3), [
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert error_events
        assert "rate limit" in error_events[-1].message.lower() or \
               "provider error" in error_events[-1].message.lower()
        assert not any(isinstance(e, TurnComplete) for e in events)

    def test_two_retry_events_before_success(self):
        agent = _make_agent(_cfg(retry_max_attempts=3), [
            RuntimeError("503"),
            RuntimeError("503"),
            _ok(),
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        retry_events = [e for e in events if isinstance(e, RetryEvent)]
        assert len(retry_events) == 2
        assert retry_events[0].attempt == 2
        assert retry_events[1].attempt == 3

    def test_non_retryable_error_no_retry_event(self):
        agent = _make_agent(_cfg(), [RuntimeError("invalid api key")])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        assert not any(isinstance(e, RetryEvent) for e in events)
        assert any(isinstance(e, ErrorEvent) for e in events)

    def test_retry_disabled_no_retry_on_transient(self):
        agent = _make_agent(_cfg(retry_enabled=False), [
            RuntimeError("rate limit exceeded"),
            _ok(),   # never reached
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        assert not any(isinstance(e, RetryEvent) for e in events)
        assert any(isinstance(e, ErrorEvent) for e in events)

    def test_max_attempts_one_no_retry(self):
        agent = _make_agent(_cfg(retry_max_attempts=1), [
            RuntimeError("rate limit"),
            _ok(),
        ])
        with patch("time.sleep"):
            events = list(agent.run("hi"))

        assert not any(isinstance(e, RetryEvent) for e in events)
        assert any(isinstance(e, ErrorEvent) for e in events)

    def test_provider_called_correct_number_of_times(self):
        """Provider.chat should be called once per attempt."""
        cfg = _cfg(retry_max_attempts=3)
        context = ContextManager(cfg)
        registry = ToolRegistry()
        provider = MagicMock()
        provider.chat.side_effect = [
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
            _ok(),
        ]
        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None
        session = MagicMock(spec=Session)
        session.id = "t"
        session.messages = []
        session.compactions = []

        agent = Agent(config=cfg, provider=provider, registry=registry,
                      context=context, session=session, session_manager=sm)

        with patch("time.sleep"):
            list(agent.run("go"))

        assert provider.chat.call_count == 3

    def test_retry_does_not_duplicate_user_message(self):
        """The user message must appear exactly once in context across retries."""
        cfg = _cfg(retry_max_attempts=3)
        context = ContextManager(cfg)
        registry = ToolRegistry()
        provider = MagicMock()
        provider.chat.side_effect = [
            RuntimeError("rate limit"),
            _ok(),
        ]
        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None
        session = MagicMock(spec=Session)
        session.id = "t"
        session.messages = []
        session.compactions = []

        agent = Agent(config=cfg, provider=provider, registry=registry,
                      context=context, session=session, session_manager=sm)

        with patch("time.sleep"):
            list(agent.run("unique input"))

        msgs = context.get_messages()
        user_msgs = [m for m in msgs if m.role == "user" and m.content == "unique input"]
        assert len(user_msgs) == 1

    def test_successful_retry_yields_turn_complete(self):
        agent = _make_agent(_cfg(), [
            RuntimeError("rate limit"),
            _ok("success after retry"),
        ])
        with patch("time.sleep"):
            events = list(agent.run("go"))

        assert any(isinstance(e, TurnComplete) for e in events)
