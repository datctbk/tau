"""Tests for tau.core.model_metadata."""

from tau.core.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _strip_provider_prefix,
    get_context_length,
    get_next_probe_tier,
    is_local_endpoint,
    parse_context_limit_from_error,
    _extract_context_length,
    _extract_pricing,
    _infer_provider_from_url,
)


class TestStripProviderPrefix:
    def test_local_prefix(self):
        assert _strip_provider_prefix("local:my-model") == "my-model"

    def test_openai_prefix(self):
        assert _strip_provider_prefix("openai:gpt-4") == "gpt-4"

    def test_no_prefix(self):
        assert _strip_provider_prefix("gpt-4o") == "gpt-4o"

    def test_ollama_tag_preserved(self):
        assert _strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"

    def test_ollama_latest_preserved(self):
        assert _strip_provider_prefix("deepseek:latest") == "deepseek:latest"

    def test_http_url_unchanged(self):
        assert _strip_provider_prefix("http://model:8080") == "http://model:8080"

    def test_unknown_prefix_unchanged(self):
        assert _strip_provider_prefix("unknown:model") == "unknown:model"


class TestGetContextLength:
    def test_exact_claude(self):
        assert get_context_length("claude-sonnet-4.6") == 1000000

    def test_substring_match_gemini(self):
        assert get_context_length("gemini-2.5-pro") == 1048576

    def test_substring_match_gpt(self):
        assert get_context_length("gpt-4.1-mini") == 1047576

    def test_fallback(self):
        assert get_context_length("unknown-model-xyz") == DEFAULT_FALLBACK_CONTEXT

    def test_with_provider_prefix(self):
        assert get_context_length("local:llama-3.1-8b") == 131072

    def test_deepseek(self):
        assert get_context_length("deepseek-chat") == 128000


class TestIsLocalEndpoint:
    def test_localhost(self):
        assert is_local_endpoint("http://localhost:8080")

    def test_127(self):
        assert is_local_endpoint("http://127.0.0.1:11434")

    def test_docker_internal(self):
        assert is_local_endpoint("http://host.docker.internal:8080")

    def test_public_url(self):
        assert not is_local_endpoint("https://api.openai.com/v1")

    def test_empty(self):
        assert not is_local_endpoint("")

    def test_private_ip(self):
        assert is_local_endpoint("http://192.168.1.100:8080")


class TestParseContextLimitFromError:
    def test_anthropic_style(self):
        msg = "maximum context length is 32768 tokens"
        assert parse_context_limit_from_error(msg) == 32768

    def test_context_exceeded(self):
        msg = "context_length_exceeded: 131072"
        assert parse_context_limit_from_error(msg) == 131072

    def test_no_match(self):
        assert parse_context_limit_from_error("some random error") is None

    def test_too_small(self):
        msg = "limit is 100 tokens"
        assert parse_context_limit_from_error(msg) is None


class TestGetNextProbeTier:
    def test_from_128k(self):
        assert get_next_probe_tier(128_000) == 64_000

    def test_from_64k(self):
        assert get_next_probe_tier(64_000) == 32_000

    def test_at_minimum(self):
        assert get_next_probe_tier(8_000) is None

    def test_below_minimum(self):
        assert get_next_probe_tier(4_000) is None


class TestExtractContextLength:
    def test_simple(self):
        assert _extract_context_length({"context_length": 32768}) == 32768

    def test_nested(self):
        payload = {"model_info": {"context_length": 65536}}
        assert _extract_context_length(payload) == 65536

    def test_no_context(self):
        assert _extract_context_length({"name": "test"}) is None


class TestExtractPricing:
    def test_simple(self):
        payload = {"pricing": {"prompt": "0.001", "completion": "0.002"}}
        result = _extract_pricing(payload)
        assert result["prompt"] == "0.001"
        assert result["completion"] == "0.002"

    def test_no_pricing(self):
        assert _extract_pricing({"name": "test"}) == {}


class TestInferProvider:
    def test_openai(self):
        assert _infer_provider_from_url("https://api.openai.com/v1") == "openai"

    def test_anthropic(self):
        assert _infer_provider_from_url("https://api.anthropic.com") == "anthropic"

    def test_unknown(self):
        assert _infer_provider_from_url("https://custom.example.com") is None

    def test_empty(self):
        assert _infer_provider_from_url("") is None
