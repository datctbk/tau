"""Tests for tau.core.context."""

import pytest
from tau.core.context import ContextManager, _estimate_tokens
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
