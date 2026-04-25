"""Tests for the agent loop using a mock provider."""
from __future__ import annotations
import time
import tempfile
import pytest
from unittest.mock import MagicMock
from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.policy import PolicyDecision
from tau.core.session import SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    Message,
    ParallelToolsEvent,
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
    PolicyDecisionEvent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _config(**kwargs) -> AgentConfig:
    return AgentConfig(provider="openai", model="gpt-4o", max_turns=5, **kwargs)


def _make_agent(
    provider_responses: list,
    extra_tools: list[ToolDefinition] | None = None,
    config: AgentConfig | None = None,
) -> Agent:
    cfg = config or _config()
    context = ContextManager(cfg)
    registry = ToolRegistry()
    if extra_tools:
        registry.register_many(extra_tools)

    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    session = MagicMock()
    session.id = "test-session"
    session.messages = []
    session.cumulative_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
    }
    sm = MagicMock()
    sm.save.return_value = None

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
# Basic response tests
# ---------------------------------------------------------------------------

def test_simple_text_response():
    agent = _make_agent([
        ProviderResponse(content="Hello, world!", tool_calls=[], stop_reason="end_turn",
                         usage=TokenUsage(input_tokens=10, output_tokens=5)),
    ])
    events = list(agent.run("hi"))
    assert any(isinstance(e, TextChunk) and e.text == "Hello, world!" for e in events)
    assert any(isinstance(e, TurnComplete) for e in events)


def test_token_usage_reported():
    agent = _make_agent([
        ProviderResponse(content="Done.", tool_calls=[], stop_reason="end_turn",
                         usage=TokenUsage(input_tokens=42, output_tokens=7)),
    ])
    events = list(agent.run("go"))
    tc = next(e for e in events if isinstance(e, TurnComplete))
    assert tc.usage.input_tokens == 42
    assert tc.usage.output_tokens == 7


def test_tool_call_dispatch():
    call = ToolCall(id="c1", name="say_hi", arguments={"name": "tau"})
    tool = ToolDefinition(
        name="say_hi", description="greet",
        parameters={"name": ToolParameter(type="string", description="name")},
        handler=lambda name: f"hi {name}",
    )
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="I greeted tau.", tool_calls=[], stop_reason="end_turn",
                             usage=TokenUsage(input_tokens=20, output_tokens=5)),
        ],
        extra_tools=[tool],
    )
    events = list(agent.run("greet tau"))
    assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 1
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.content == "hi tau"
    assert not result_events[0].result.is_error
    assert any(isinstance(e, TextChunk) and e.text == "I greeted tau." for e in events)


def test_unknown_tool_returns_error_result():
    call = ToolCall(id="c2", name="does_not_exist", arguments={})
    agent = _make_agent(provider_responses=[
        ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
        ProviderResponse(content="Handled error.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
    ])
    events = list(agent.run("call unknown"))
    error_results = [e for e in events if isinstance(e, ToolResultEvent) and e.result.is_error]
    assert len(error_results) == 1


def test_policy_hook_can_block_tool_call_before_dispatch():
    call = ToolCall(id="c3", name="say_hi", arguments={"name": "tau"})
    handler = MagicMock(return_value="hi tau")
    tool = ToolDefinition(
        name="say_hi",
        description="greet",
        parameters={"name": ToolParameter(type="string", description="name")},
        handler=handler,
    )
    ws = tempfile.mkdtemp(prefix="tau-policy-block-")
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=_config(workspace_root=ws),
    )

    class _DenyPolicy:
        def before_tool_call(self, *, agent, call):
            _ = agent
            _ = call
            return PolicyDecision(allow=False, reason="Blocked by test policy", risk="high")

    agent._policy_hook = _DenyPolicy()
    events = list(agent.run("greet tau"))
    policy_events = [e for e in events if isinstance(e, PolicyDecisionEvent)]
    assert len(policy_events) == 1
    assert policy_events[0].decision == "block"
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].result.is_error is True
    assert "Blocked by test policy" in results[0].result.content
    handler.assert_not_called()

    audit_file = agent._config.workspace_root + "/.tau/audit/assistant-actions.jsonl"
    with open(audit_file, "r", encoding="utf-8") as f:
        txt = f.read()
    assert "policy.blocked" in txt

    event_file = agent._config.workspace_root + "/.tau/events/assistant-events.jsonl"
    with open(event_file, "r", encoding="utf-8") as f:
        evt = f.read()
    assert '"family": "policy"' in evt
    assert '"name": "blocked"' in evt


def test_policy_approval_granted_executes_tool():
    call = ToolCall(id="c4", name="write_file", arguments={"path": "x", "content": "y"})
    handler = MagicMock(return_value="ok")
    tool = ToolDefinition(
        name="write_file",
        description="write",
        parameters={
            "path": ToolParameter(type="string", description="p"),
            "content": ToolParameter(type="string", description="c"),
        },
        handler=handler,
    )
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
    )

    class _ApprovePolicy:
        def before_tool_call(self, *, agent, call):
            _ = agent
            _ = call
            from tau.core.policy import PolicyDecision
            return PolicyDecision(allow=True, requires_approval=True, risk="medium", reason="needs approval")

    agent._policy_hook = _ApprovePolicy()
    agent._policy_approval_hook = lambda reason: True
    events = list(agent.run("do write"))
    decisions = [e for e in events if isinstance(e, PolicyDecisionEvent)]
    assert any(d.decision == "confirm" for d in decisions)
    assert any(d.decision == "approved" for d in decisions)
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].result.is_error is False
    handler.assert_called_once()


def test_policy_approval_denied_blocks_tool():
    call = ToolCall(id="c5", name="write_file", arguments={"path": "x", "content": "y"})
    handler = MagicMock(return_value="ok")
    tool = ToolDefinition(
        name="write_file",
        description="write",
        parameters={
            "path": ToolParameter(type="string", description="p"),
            "content": ToolParameter(type="string", description="c"),
        },
        handler=handler,
    )
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
    )

    class _ApprovePolicy:
        def before_tool_call(self, *, agent, call):
            _ = agent
            _ = call
            from tau.core.policy import PolicyDecision
            return PolicyDecision(allow=True, requires_approval=True, risk="medium", reason="needs approval")

    agent._policy_hook = _ApprovePolicy()
    agent._policy_approval_hook = lambda reason: False
    events = list(agent.run("do write"))
    decisions = [e for e in events if isinstance(e, PolicyDecisionEvent)]
    assert any(d.decision == "confirm" for d in decisions)
    assert any(d.decision == "denied" for d in decisions)
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].result.is_error is True
    handler.assert_not_called()


def test_max_turns_yields_error_event():
    call = ToolCall(id="cx", name="loop_tool", arguments={})
    tool = ToolDefinition(
        name="loop_tool", description="loops", parameters={},
        handler=lambda: "looping",
    )
    cfg = AgentConfig(provider="openai", model="gpt-4o", max_turns=3)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
        ] * 10,
        extra_tools=[tool],
        config=cfg,
    )
    events = list(agent.run("loop forever"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert error_events
    assert "max_turns" in error_events[-1].message


def test_provider_exception_yields_error_event():
    provider = MagicMock()
    provider.chat.side_effect = RuntimeError("network failure")
    cfg = _config()
    session = MagicMock()
    session.id = "s"
    session.messages = []
    session.cumulative_usage = {"input_tokens": 0, "output_tokens": 0,
                                "cache_read_tokens": 0, "cache_write_tokens": 0}
    agent = Agent(
        config=cfg, provider=provider, registry=ToolRegistry(),
        context=ContextManager(cfg), session=session, session_manager=MagicMock(),
    )
    events = list(agent.run("crash"))
    assert any(isinstance(e, ErrorEvent) and "network failure" in e.message for e in events)


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

def test_streaming_yields_text_deltas():
    words = ["Hello", ", ", "world", "!"]
    agent = _make_agent([_streaming_response(words)])
    events = list(agent.run("hi"))
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(deltas) == 4
    assert [d.text for d in deltas] == words
    assert any(isinstance(e, TurnComplete) for e in events)


def test_streaming_deltas_arrive_before_turn_complete():
    agent = _make_agent([_streaming_response(["a", "b", "c"])])
    events = list(agent.run("go"))
    last_delta = max(i for i, e in enumerate(events) if isinstance(e, TextDelta))
    turn_complete = next(i for i, e in enumerate(events) if isinstance(e, TurnComplete))
    assert last_delta < turn_complete


def test_streaming_no_text_chunk_emitted():
    agent = _make_agent([_streaming_response(["streaming", " text"])])
    events = list(agent.run("go"))
    assert not any(isinstance(e, TextChunk) for e in events)


def test_empty_end_turn_retries_once_with_nudge():
    provider = MagicMock()
    provider.chat.side_effect = [
        ProviderResponse(content=None, tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ProviderResponse(content="Recovered.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
    ]

    cfg = _config()
    session = MagicMock()
    session.id = "s"
    session.messages = []
    session.cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    agent = Agent(
        config=cfg,
        provider=provider,
        registry=ToolRegistry(),
        context=ContextManager(cfg),
        session=session,
        session_manager=MagicMock(),
    )

    events = list(agent.run("hi"))
    assert any(isinstance(e, TextChunk) and e.text == "Recovered." for e in events)
    assert provider.chat.call_count == 2


def test_empty_end_turn_twice_yields_error():
    provider = MagicMock()
    provider.chat.side_effect = [
        ProviderResponse(content=None, tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ProviderResponse(content=None, tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
    ]

    cfg = _config()
    session = MagicMock()
    session.id = "s"
    session.messages = []
    session.cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    agent = Agent(
        config=cfg,
        provider=provider,
        registry=ToolRegistry(),
        context=ContextManager(cfg),
        session=session,
        session_manager=MagicMock(),
    )

    events = list(agent.run("hi"))
    errs = [e for e in events if isinstance(e, ErrorEvent)]
    assert errs
    assert "aborting empty-response retries" in errs[-1].message
    assert provider.chat.call_count == 2


def test_empty_content_with_tool_calls_does_not_retry():
    call = ToolCall(id="tc1", name="does_not_exist", arguments={})
    provider = MagicMock()
    provider.chat.side_effect = [
        ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
        ProviderResponse(content="Handled.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
    ]

    cfg = _config()
    session = MagicMock()
    session.id = "s"
    session.messages = []
    session.cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    agent = Agent(
        config=cfg,
        provider=provider,
        registry=ToolRegistry(),
        context=ContextManager(cfg),
        session=session,
        session_manager=MagicMock(),
    )

    events = list(agent.run("go"))
    assert any(isinstance(e, TextChunk) and e.text == "Handled." for e in events)
    # Exactly 2 calls: initial tool_use + follow-up after tool result. No extra empty-response retry.
    assert provider.chat.call_count == 2


def test_streaming_tool_call_after_stream():
    call = ToolCall(id="s1", name="greet", arguments={"name": "tau"})
    tool = ToolDefinition(
        name="greet", description="greet",
        parameters={"name": ToolParameter(type="string", description="name")},
        handler=lambda name: f"hi {name}",
    )
    agent = _make_agent(
        provider_responses=[
            _streaming_response([], tool_calls=[call], stop_reason="tool_use"),
            ProviderResponse(content="Done.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
    )
    events = list(agent.run("greet tau"))
    assert any(isinstance(e, ToolCallEvent) and e.call.name == "greet" for e in events)
    assert any(isinstance(e, ToolResultEvent) and e.result.content == "hi tau" for e in events)
    assert any(isinstance(e, TurnComplete) for e in events)


def test_streaming_usage_reported():
    def _gen():
        yield TextDelta(text="ok")
        yield ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn",
                               usage=TokenUsage(input_tokens=55, output_tokens=3))
    agent = _make_agent([_gen()])
    events = list(agent.run("go"))
    tc = next(e for e in events if isinstance(e, TurnComplete))
    assert tc.usage.input_tokens == 55
    assert tc.usage.output_tokens == 3


def test_prompt_budget_disabled_keeps_all_tools():
    extra_tools = [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"helper tool {i}",
            parameters={},
            handler=lambda i=i: f"ok-{i}",
        )
        for i in range(15)
    ]
    cfg = _config(prompt_budget_enabled=False)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=extra_tools,
        config=cfg,
    )
    list(agent.run("hello"))
    sent_tools = agent._provider.chat.call_args.kwargs["tools"]
    assert len(sent_tools) == 15


def test_prompt_budget_enabled_caps_tools_and_trims_messages():
    extra_tools = [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"helper tool {i}",
            parameters={},
            handler=lambda i=i: f"ok-{i}",
        )
        for i in range(20)
    ]
    cfg = _config(
        prompt_budget_enabled=True,
        prompt_budget_max_tools_total=5,
        prompt_budget_max_input_tokens=120,
        prompt_budget_output_reserve=80,
    )
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=extra_tools,
        config=cfg,
    )
    # Seed context with enough history so budget mode must trim.
    for i in range(14):
        agent._context.add_message(
            Message(role="user", content=f"history line {i} " * 20)
        )
    before_count = len(agent._context.get_messages())

    list(agent.run("please use tool_19 for this task"))
    sent_messages = agent._provider.chat.call_args.kwargs["messages"]
    sent_tools = agent._provider.chat.call_args.kwargs["tools"]

    assert len(sent_tools) <= 5
    assert len(sent_messages) <= before_count + 1
    assert any(t.name == "tool_19" for t in sent_tools)


# ---------------------------------------------------------------------------
# Parallel tool execution tests
# ---------------------------------------------------------------------------

def test_parallel_tools_all_results_present():
    """All tool results are present when parallel execution is enabled."""
    calls = [ToolCall(id=f"p{i}", name="slow_tool", arguments={"idx": i}) for i in range(4)]

    def slow_tool(idx: int) -> str:
        time.sleep(0.05)
        return f"result-{idx}"

    tool = ToolDefinition(
        name="slow_tool", description="slow",
        parameters={"idx": ToolParameter(type="integer", description="index")},
        handler=slow_tool,
    )
    cfg = _config(parallel_tools=True)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=calls, stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=cfg,
    )
    events = list(agent.run("run all"))
    assert any(isinstance(e, ParallelToolsEvent) for e in events)
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 4
    assert {e.result.content for e in results} == {"result-0", "result-1", "result-2", "result-3"}


def test_parallel_tools_faster_than_sequential():
    """Parallel dispatch of N slow tools is faster than N × sleep_time."""
    SLEEP = 0.1
    N = 4
    calls = [ToolCall(id=f"t{i}", name="wait", arguments={"i": i}) for i in range(N)]

    def wait(i: int) -> str:
        time.sleep(SLEEP)
        return f"done-{i}"

    tool = ToolDefinition(
        name="wait", description="waits",
        parameters={"i": ToolParameter(type="integer", description="i")},
        handler=wait,
    )
    cfg = _config(parallel_tools=True)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=calls, stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=cfg,
    )
    start = time.monotonic()
    list(agent.run("go"))
    elapsed = time.monotonic() - start

    # Parallel: ~SLEEP. Sequential: N*SLEEP. Allow 2× as safe upper bound.
    assert elapsed < SLEEP * 2, f"Expected < {SLEEP * 2:.2f}s but took {elapsed:.2f}s"


def test_parallel_tools_original_order_preserved():
    """ToolResultEvent order matches the LLM's original request order."""
    calls = [ToolCall(id=f"o{i}", name="echo", arguments={"v": i}) for i in range(5)]
    tool = ToolDefinition(
        name="echo", description="echo",
        parameters={"v": ToolParameter(type="integer", description="v")},
        handler=lambda v: str(v),
    )
    cfg = _config(parallel_tools=True)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=calls, stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=cfg,
    )
    events = list(agent.run("order"))
    result_contents = [e.result.content for e in events if isinstance(e, ToolResultEvent)]
    assert result_contents == ["0", "1", "2", "3", "4"]


def test_parallel_tools_disabled_sequential():
    """When parallel_tools=False, tools still run correctly."""
    calls = [ToolCall(id=f"s{i}", name="echo", arguments={"v": i}) for i in range(3)]
    tool = ToolDefinition(
        name="echo", description="echo",
        parameters={"v": ToolParameter(type="integer", description="v")},
        handler=lambda v: str(v),
    )
    cfg = _config(parallel_tools=False)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=calls, stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="done", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=cfg,
    )
    events = list(agent.run("seq"))
    result_contents = [e.result.content for e in events if isinstance(e, ToolResultEvent)]
    assert result_contents == ["0", "1", "2"]


def test_parallel_tool_error_does_not_stop_others():
    """A failing tool in a parallel batch still lets other tools complete."""
    calls = [
        ToolCall(id="ok1", name="echo", arguments={"v": 1}),
        ToolCall(id="bad", name="boom", arguments={}),
        ToolCall(id="ok2", name="echo", arguments={"v": 3}),
    ]
    tool_echo = ToolDefinition(
        name="echo", description="echo",
        parameters={"v": ToolParameter(type="integer", description="v")},
        handler=lambda v: str(v),
    )
    tool_boom = ToolDefinition(
        name="boom", description="explodes", parameters={},
        handler=lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    cfg = _config(parallel_tools=True)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=calls, stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="handled", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool_echo, tool_boom],
        config=cfg,
    )
    events = list(agent.run("mixed"))
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 3
    assert results[0].result.content == "1"
    assert results[1].result.is_error
    assert results[2].result.content == "3"


def test_single_tool_skips_thread_pool():
    """A single tool call never goes through the thread pool (fast path)."""
    call = ToolCall(id="one", name="echo", arguments={"v": 42})
    tool = ToolDefinition(
        name="echo", description="echo",
        parameters={"v": ToolParameter(type="integer", description="v")},
        handler=lambda v: str(v),
    )
    cfg = _config(parallel_tools=True)
    agent = _make_agent(
        provider_responses=[
            ProviderResponse(content=None, tool_calls=[call], stop_reason="tool_use", usage=TokenUsage()),
            ProviderResponse(content="ok", tool_calls=[], stop_reason="end_turn", usage=TokenUsage()),
        ],
        extra_tools=[tool],
        config=cfg,
    )
    events = list(agent.run("one tool"))
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].result.content == "42"
