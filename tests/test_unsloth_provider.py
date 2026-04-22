from __future__ import annotations

from types import SimpleNamespace

from tau.providers.unsloth_provider import _estimate_prompt_tokens_from_messages, _estimate_tokens
from tau.providers.unsloth_provider import _split_stream_text_for_thinking
from tau.providers.unsloth_provider import UnslothProvider


def test_unsloth_fallback_token_estimators_are_nonzero_for_text():
    messages = [
        {"role": "system", "content": "You are tau."},
        {"role": "user", "content": "Please create a daily brief at 7 AM."},
    ]
    assert _estimate_prompt_tokens_from_messages(messages) > 0
    assert _estimate_tokens("Short assistant response.") > 0


def test_unsloth_stream_throttle_config_maps_to_provider_fields():
    cfg = SimpleNamespace(
        unsloth=SimpleNamespace(
            base_url="http://localhost:8001/v1",
            timeout_seconds=10.0,
            stream_yield_every_chunks=3,
            stream_yield_ms=5.0,
        )
    )
    agent_cfg = SimpleNamespace(model="test-model")
    p = UnslothProvider(cfg, agent_cfg)
    assert p._stream_yield_every_chunks == 3
    assert p._stream_yield_s == 0.005


def test_split_stream_text_for_thinking_tags():
    visible, think, in_think, carry = _split_stream_text_for_thinking(
        "A<think>plan first</think>B",
        in_thinking=False,
    )
    assert visible == "AB"
    assert think == "plan first"
    assert in_think is False
    assert carry == ""


def test_split_stream_text_for_thinking_with_partial_tag_carry():
    visible, think, in_think, carry = _split_stream_text_for_thinking(
        "Hello <thi",
        in_thinking=False,
    )
    assert visible == "Hello "
    assert think == ""
    assert in_think is False
    assert carry == "<thi"
