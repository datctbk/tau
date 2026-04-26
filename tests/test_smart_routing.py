"""Tests for tau.core.smart_routing."""

from tau.core.smart_routing import (
    choose_cheap_model_route,
    resolve_turn_route,
)


class TestChooseCheapModelRoute:
    ROUTING_CFG = {
        "enabled": True,
        "max_simple_chars": 160,
        "max_simple_words": 28,
        "cheap_model": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    }

    def test_disabled(self):
        cfg = {**self.ROUTING_CFG, "enabled": False}
        assert choose_cheap_model_route("Hello", cfg) is None

    def test_no_config(self):
        assert choose_cheap_model_route("Hello", None) is None
        assert choose_cheap_model_route("Hello", {}) is None

    def test_simple_message(self):
        route = choose_cheap_model_route("What time is it?", self.ROUTING_CFG)
        assert route is not None
        assert route["model"] == "gpt-4o-mini"
        assert route["provider"] == "openai"
        assert route["routing_reason"] == "simple_turn"

    def test_long_message_not_cheap(self):
        long_msg = "X" * 200
        assert choose_cheap_model_route(long_msg, self.ROUTING_CFG) is None

    def test_many_words_not_cheap(self):
        many_words = " ".join(["word"] * 35)
        assert choose_cheap_model_route(many_words, self.ROUTING_CFG) is None

    def test_code_fence_not_cheap(self):
        assert choose_cheap_model_route("```python\nprint('hi')```", self.ROUTING_CFG) is None

    def test_inline_code_not_cheap(self):
        assert choose_cheap_model_route("Fix `main.py`", self.ROUTING_CFG) is None

    def test_url_not_cheap(self):
        assert choose_cheap_model_route("Read https://example.com", self.ROUTING_CFG) is None

    def test_multiline_not_cheap(self):
        msg = "Line 1\nLine 2\nLine 3"
        assert choose_cheap_model_route(msg, self.ROUTING_CFG) is None

    def test_complex_keywords(self):
        for kw in ["debug this", "implement feature", "refactor code", "run pytest"]:
            assert choose_cheap_model_route(kw, self.ROUTING_CFG) is None, f"Should be complex: {kw}"

    def test_empty_message(self):
        assert choose_cheap_model_route("", self.ROUTING_CFG) is None

    def test_greeting(self):
        route = choose_cheap_model_route("Hey, how are you?", self.ROUTING_CFG)
        assert route is not None

    def test_missing_cheap_model(self):
        cfg = {**self.ROUTING_CFG, "cheap_model": {}}
        assert choose_cheap_model_route("Hello", cfg) is None


class TestResolveTurnRoute:
    PRIMARY = {
        "model": "claude-sonnet-4",
        "api_key": "sk-primary",
        "base_url": "https://api.anthropic.com",
        "provider": "anthropic",
    }

    def test_no_routing(self):
        result = resolve_turn_route("Hello", None, self.PRIMARY)
        assert result["model"] == "claude-sonnet-4"
        assert result["label"] is None

    def test_complex_keeps_primary(self):
        result = resolve_turn_route("Debug the failing test", None, self.PRIMARY)
        assert result["model"] == "claude-sonnet-4"

    def test_simple_uses_cheap(self):
        cfg = {
            "enabled": True,
            "cheap_model": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        result = resolve_turn_route("Hi there", cfg, self.PRIMARY)
        assert result["model"] == "gpt-4o-mini"
        assert "smart route" in result["label"]

    def test_signature_tuple(self):
        result = resolve_turn_route("Hello", None, self.PRIMARY)
        assert isinstance(result["signature"], tuple)
        assert result["signature"][0] == "claude-sonnet-4"
