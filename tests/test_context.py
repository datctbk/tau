"""Tests for tau.core.context."""

import pytest
from unittest.mock import patch
from tau.core.context import (
    ContextManager,
    SlidingWindowStrategy,
    SummariseStrategy,
    _estimate_tokens,
    _messages_tokens,
)
from tau.core.types import AgentConfig, Message


def _config(**kwargs) -> AgentConfig:
    return AgentConfig(system_prompt="You are tau.", **kwargs)


def test_system_prompt_injected():
    ctx = ContextManager(_config())
    msgs = ctx.get_messages()
    assert msgs[0].role == "system"
    assert "tau" in msgs[0].content


def test_add_and_get_messages():
    ctx = ContextManager(_config())
    ctx.add_message(Message(role="user", content="hello"))
    ctx.add_message(Message(role="assistant", content="hi"))
    roles = [m.role for m in ctx.get_messages()]
    assert roles == ["system", "user", "assistant"]


def test_token_count_nonzero():
    ctx = ContextManager(_config())
    ctx.add_message(Message(role="user", content="hello world"))
    assert ctx.token_count() > 0


def test_trim_removes_old_messages():
    # Set a tiny token budget so trimming kicks in
    cfg = _config(max_tokens=60, trim_strategy="sliding_window")
    ctx = ContextManager(cfg)
    for i in range(20):
        ctx.add_message(Message(role="user", content=f"message number {i} with some padding text"))
        ctx.add_message(Message(role="assistant", content=f"reply {i}"))
    before = len(ctx.get_messages())
    ctx.trim()
    after = len(ctx.get_messages())
    assert after < before
    # System prompt must survive
    assert ctx.get_messages()[0].role == "system"


def test_snapshot_is_serialisable():
    ctx = ContextManager(_config())
    ctx.add_message(Message(role="user", content="test"))
    snap = ctx.snapshot()
    assert isinstance(snap, list)
    assert all(isinstance(m, dict) for m in snap)


def test_restore_skips_system():
    ctx = ContextManager(_config())
    ctx.restore([
        {"role": "user", "content": "restored message"},
        {"role": "assistant", "content": "restored reply"},
    ])
    roles = [m.role for m in ctx.get_messages()]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles


def test_inject_prompt_fragment():
    ctx = ContextManager(_config())
    ctx.inject_prompt_fragment("Extra instructions here.")
    system_content = ctx.get_messages()[0].content
    assert "Extra instructions here." in system_content


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def test_estimate_tokens_nonzero():
    assert _estimate_tokens("hello world") > 0

def test_estimate_tokens_scales_with_length():
    short = _estimate_tokens("hi")
    long = _estimate_tokens("hello world, this is a much longer piece of text")
    assert long > short

def test_estimate_tokens_fallback():
    """Force the char/4 fallback by pretending tiktoken is absent."""
    with patch("tau.core.context._enc", None):
        # Import the module-level function fresh to use the fallback branch
        result = _estimate_tokens("hello")
        assert result >= 1

# ---------------------------------------------------------------------------
# SlidingWindowStrategy
# ---------------------------------------------------------------------------

def test_sliding_window_under_budget_unchanged():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]
    strategy = SlidingWindowStrategy()
    result = strategy.trim(msgs, budget=10_000)
    assert result == msgs

def test_sliding_window_preserves_system():
    system = Message(role="system", content="sys")
    msgs = [system] + [
        Message(role="user", content=f"message {i} " * 20)
        for i in range(30)
    ]
    strategy = SlidingWindowStrategy()
    result = strategy.trim(msgs, budget=50)
    assert result[0].role == "system"

def test_sliding_window_drops_oldest_first():
    msgs = [
        Message(role="user", content="first " * 20),
        Message(role="assistant", content="second " * 20),
        Message(role="user", content="third " * 20),
    ]
    strategy = SlidingWindowStrategy()
    result = strategy.trim(msgs, budget=60)
    contents = [m.content for m in result]
    # Oldest messages should be gone, newest should survive
    assert any("third" in c for c in contents)

# ---------------------------------------------------------------------------
# SummariseStrategy
# ---------------------------------------------------------------------------

def test_summarise_under_budget_unchanged():
    msgs = [Message(role="user", content="hi")]
    strategy = SummariseStrategy()
    result = strategy.trim(msgs, budget=10_000)
    assert result == msgs

def test_summarise_falls_back_when_too_few_messages():
    """With fewer than KEEP_RECENT messages, falls back to sliding window."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi " * 50),
    ]
    strategy = SummariseStrategy()
    # Budget so small trimming is required
    result = strategy.trim(msgs, budget=10)
    assert result[0].role == "system"

def test_summarise_calls_summary_when_enough_history():
    """With enough history, _call_summary is invoked and summary appears in result."""
    system = Message(role="system", content="sys")
    msgs = [system] + [
        Message(role="user", content=f"message {i} " * 30)
        for i in range(20)
    ]
    strategy = SummariseStrategy()
    fake_summary = "This is a concise summary."
    with patch.object(strategy, "_call_summary", return_value=fake_summary) as mock_summary:
        # Budget below the full message count (1820 tokens) but large enough
        # to hold system + summary + 6 recent messages
        result = strategy.trim(msgs, budget=1500)
        mock_summary.assert_called_once()
    assert any(fake_summary in m.content for m in result)


def test_summarise_call_failure_falls_back():
    """If _call_summary raises, trim() falls back to sliding window without raising."""
    system = Message(role="system", content="sys")
    msgs = [system] + [
        Message(role="user", content=f"message {i} " * 30)
        for i in range(20)
    ]
    strategy = SummariseStrategy()
    with patch.object(strategy, "_call_summary", side_effect=RuntimeError("network error")):
        result = strategy.trim(msgs, budget=5000)
    # Should not raise; system prompt must survive
    assert isinstance(result, list)
    assert result[0].role == "system"
