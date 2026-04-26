"""Tests for tau.core.prompt_caching."""

import copy

from tau.core.prompt_caching import apply_anthropic_cache_control


class TestApplyCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_system_prompt_gets_marker(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_anthropic_cache_control(messages)
        # System prompt converted to list format with cache_control
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_last_three_non_system_get_markers(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Msg 3"},
        ]
        result = apply_anthropic_cache_control(messages)
        # System (idx 0) + last 3 non-system (idx 3, 4, 5) = 4 breakpoints
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0].get("cache_control") is not None
        # Msg 1 (idx 1) should NOT have marker
        assert isinstance(result[1]["content"], str) or (
            isinstance(result[1]["content"], list) and
            "cache_control" not in result[1]["content"][-1]
        )
        # Last 3 should have markers
        for idx in [3, 4, 5]:
            content = result[idx]["content"]
            if isinstance(content, list):
                assert content[-1].get("cache_control") is not None
            else:
                assert "cache_control" in result[idx]

    def test_no_system_prompt(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = apply_anthropic_cache_control(messages)
        # No system = all 4 breakpoints available for non-system
        # With only 3 messages, all get markers
        for msg in result:
            content = msg["content"]
            if isinstance(content, list):
                assert content[-1].get("cache_control") is not None

    def test_1h_ttl(self):
        messages = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(messages, cache_ttl="1h")
        assert result[0]["content"][0]["cache_control"] == {
            "type": "ephemeral", "ttl": "1h",
        }

    def test_does_not_mutate_original(self):
        messages = [{"role": "user", "content": "Original"}]
        original = copy.deepcopy(messages)
        apply_anthropic_cache_control(messages)
        assert messages == original

    def test_list_content_block(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]},
        ]
        result = apply_anthropic_cache_control(messages)
        # Marker goes on the last block
        assert result[0]["content"][-1].get("cache_control") is not None
        assert "cache_control" not in result[0]["content"][0]

    def test_tool_role_native_anthropic(self):
        messages = [
            {"role": "tool", "content": "result"},
        ]
        result = apply_anthropic_cache_control(messages, native_anthropic=True)
        assert result[0].get("cache_control") == {"type": "ephemeral"}

    def test_tool_role_non_native(self):
        messages = [
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "Next"},
        ]
        result = apply_anthropic_cache_control(messages, native_anthropic=False)
        # Tool role should NOT get marker when not native
        assert "cache_control" not in result[0]

    def test_none_content(self):
        messages = [{"role": "assistant", "content": None}]
        result = apply_anthropic_cache_control(messages)
        assert result[0].get("cache_control") == {"type": "ephemeral"}
