"""Tests for the agent loop using a mock provider."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import SessionManager
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
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolResultEvent,
    TurnComplete,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _config() -> AgentConfig:
    return AgentConfig(provider="openai", model="gpt-4o", max_turns=5)


def _make_agent(
    provider_responses: list[ProviderResponse],
    extra_tools: list[ToolDefinition] | None = None,
    sm: SessionManager | None = None,
) -> Agent:
    cfg = _config()
    context = ContextManager(cfg)
    registry = ToolRegistry()

    if extra_tools:
        registry.register_many(extra_tools)

    # Mock provider that pops from responses list
    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    # Minimal session + session manager
    if sm is None:
        sm = MagicMock()
        sm.new_session.return_value = MagicMock(id="test-session", messages=[])
        sm.save.return_value = None

    session = MagicMock()
    session.id = "test-session"
    session.messages = []

    return Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )


def _streaming_response(words: list[str], tool_calls=None, stop_reason="end_turn"):
    """Helper: yields TextDelta per word then a final ProviderResponse."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_text_response():
    """Agent yields TextChunk + TurnComplete for a plain text reply."""
    agent = _make_agent([
        ProviderResponse(
            content="Hello, world!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
    ])
    events = list(agent.run("hi"))
    assert any(isinstance(e, TextChunk) and e.text == "Hello, world!" for e in events)
    assert any(isinstance(e, TurnComplete) for e in events)


def test_token_usage_reported():
    agent = _make_agent([
        ProviderResponse(
            content="Done.",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=42, output_tokens=7),
        )
    ])
    events = list(agent.run("go"))
    turn_complete = next(e for e in events if isinstance(e, TurnComplete))
    assert turn_complete.usage.input_tokens == 42
    assert turn_complete.usage.output_tokens == 7


def test_tool_call_dispatch():
    """Agent calls a tool and feeds the result back."""
    call = ToolCall(id="c1", name="say_hi", arguments={"name": "tau"})

    tool = ToolDefinition(
        name="say_hi",
        description="greet",
        parameters={"name": ToolParameter(type="string", description="name")},
        handler=lambda name: f"hi {name}",
    )

    agent = _make_agent(
        provider_responses=[
            # First turn: tool call
            ProviderResponse(
                content=None,
                tool_calls=[call],
                stop_reason="tool_use",
                usage=TokenUsage(),
            ),
            # Second turn: final answer after seeing tool result
            ProviderResponse(
                content="I greeted tau.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=20, output_tokens=5),
            ),
        ],
        extra_tools=[tool],
    )

    events = list(agent.run("greet tau"))

    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    text_events = [e for e in events if isinstance(e, TextChunk)]

    assert len(tool_call_events) == 1
    assert tool_call_events[0].call.name == "say_hi"

    assert len(tool_result_events) == 1
    assert tool_result_events[0].result.content == "hi tau"
    assert not tool_result_events[0].result.is_error

    assert any(e.text == "I greeted tau." for e in text_events)


def test_unknown_tool_returns_error_result():
    """Dispatching an unknown tool yields a ToolResultEvent with is_error=True."""
    call = ToolCall(id="c2", name="does_not_exist", arguments={})

    agent = _make_agent(
        provider_responses=[
            ProviderResponse(
                content=None,
                tool_calls=[call],
                stop_reason="tool_use",
                usage=TokenUsage(),
            ),
            ProviderResponse(
                content="Handled error.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(),
            ),
        ]
    )

    events = list(agent.run("call unknown"))
    error_results = [e for e in events if isinstance(e, ToolResultEvent) and e.result.is_error]
    assert len(error_results) == 1


def test_max_turns_yields_error_event():
    """Agent stops and emits ErrorEvent when max_turns is exceeded."""
    # Always respond with a tool call so the loop never exits naturally
    call = ToolCall(id="cx", name="loop_tool", arguments={})
    tool = ToolDefinition(
        name="loop_tool",
        description="loops",
        parameters={},
        handler=lambda: "looping",
    )

    cfg = AgentConfig(provider="openai", model="gpt-4o", max_turns=3)
    context = ContextManager(cfg)
    registry = ToolRegistry()
    registry.register(tool)

    provider = MagicMock()
    provider.chat.return_value = ProviderResponse(
        content=None,
        tool_calls=[call],
        stop_reason="tool_use",
        usage=TokenUsage(),
    )

    session = MagicMock()
    session.id = "s"
    session.messages = []
    sm = MagicMock()

    agent = Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )

    events = list(agent.run("loop forever"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert error_events
    assert "max_turns" in error_events[-1].message


def test_provider_exception_yields_error_event():
    """A provider crash emits ErrorEvent and stops the loop."""
    provider = MagicMock()
    provider.chat.side_effect = RuntimeError("network failure")

    cfg = _config()
    context = ContextManager(cfg)
    registry = ToolRegistry()
    session = MagicMock()
    session.id = "s"
    session.messages = []
    sm = MagicMock()

    agent = Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
    )

    events = list(agent.run("crash"))
    assert any(isinstance(e, ErrorEvent) and "network failure" in e.message for e in events)


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

def test_streaming_yields_text_deltas():
    """TextDelta events are emitted before TurnComplete, in order."""
    words = ["Hello", ", ", "world", "!"]
    agent = _make_agent([_streaming_response(words)])

    events = list(agent.run("hi"))
    deltas = [e for e in events if isinstance(e, TextDelta)]

    assert len(deltas) == 4
    assert [d.text for d in deltas] == words
    assert any(isinstance(e, TurnComplete) for e in events)


def test_streaming_deltas_arrive_before_turn_complete():
    """All TextDeltas precede TurnComplete in the event sequence."""
    agent = _make_agent([_streaming_response(["a", "b", "c"])])

    events = list(agent.run("go"))
    indices = {type(e).__name__: i for i, e in enumerate(events)}

    last_delta = max(i for i, e in enumerate(events) if isinstance(e, TextDelta))
    turn_complete = next(i for i, e in enumerate(events) if isinstance(e, TurnComplete))

    assert last_delta < turn_complete


def test_streaming_no_text_chunk_emitted():
    """When streaming, no TextChunk is emitted (deltas replace it)."""
    agent = _make_agent([_streaming_response(["streaming", " text"])])
    events = list(agent.run("go"))
    assert not any(isinstance(e, TextChunk) for e in events)


def test_streaming_tool_call_after_stream():
    """Tool call from a streaming provider is dispatched correctly."""
    call = ToolCall(id="s1", name="greet", arguments={"name": "tau"})
    tool = ToolDefinition(
        name="greet",
        description="greet",
        parameters={"name": ToolParameter(type="string", description="name")},
        handler=lambda name: f"hi {name}",
    )

    agent = _make_agent(
        provider_responses=[
            # First turn: streaming response that ends with a tool call
            _streaming_response([], tool_calls=[call], stop_reason="tool_use"),
            # Second turn: final plain response
            ProviderResponse(content="Done.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
    )

    events = list(agent.run("greet tau"))
    assert any(isinstance(e, ToolCallEvent) and e.call.name == "greet" for e in events)
    assert any(isinstance(e, ToolResultEvent) and e.result.content == "hi tau" for e in events)
    assert any(isinstance(e, TurnComplete) for e in events)


def test_streaming_usage_reported():
    """TokenUsage from streaming ProviderResponse is passed through TurnComplete."""
    def _gen():
        yield TextDelta(text="ok")
        yield ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn",
                               usage=TokenUsage(input_tokens=55, output_tokens=3))
    agent = _make_agent([_gen()])

    events = list(agent.run("go"))
    tc = next(e for e in events if isinstance(e, TurnComplete))
    assert tc.usage.input_tokens == 55
    assert tc.usage.output_tokens == 3
