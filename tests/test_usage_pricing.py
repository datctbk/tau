"""Tests for tau.core.usage_pricing."""

from decimal import Decimal
from types import SimpleNamespace

from tau.core.usage_pricing import (
    CanonicalUsage,
    CostResult,
    estimate_usage_cost,
    format_duration_compact,
    format_token_count_compact,
    has_known_pricing,
    normalize_usage,
    resolve_billing_route,
)


class TestCanonicalUsage:
    def test_total_tokens(self):
        usage = CanonicalUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_prompt_tokens_with_cache(self):
        usage = CanonicalUsage(
            input_tokens=100, cache_read_tokens=50, cache_write_tokens=25,
        )
        assert usage.prompt_tokens == 175


class TestResolveBillingRoute:
    def test_anthropic_provider(self):
        route = resolve_billing_route("claude-sonnet-4", provider="anthropic")
        assert route.provider == "anthropic"
        assert route.model == "claude-sonnet-4"
        assert route.billing_mode == "official_docs_snapshot"

    def test_openrouter(self):
        route = resolve_billing_route("model", provider="openrouter")
        assert route.provider == "openrouter"
        assert route.billing_mode == "provider_models_api"

    def test_local(self):
        route = resolve_billing_route("llama", provider="local")
        assert route.billing_mode == "unknown"

    def test_infer_from_slash(self):
        route = resolve_billing_route("anthropic/claude-sonnet-4")
        assert route.provider == "anthropic"
        assert route.model == "claude-sonnet-4"


class TestNormalizeUsage:
    def test_anthropic_style(self):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=100,
        )
        result = normalize_usage(usage, provider="anthropic")
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.cache_read_tokens == 200
        assert result.cache_write_tokens == 100

    def test_openai_style(self):
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=200, cache_write_tokens=0),
            output_tokens_details=None,
        )
        result = normalize_usage(usage)
        assert result.input_tokens == 800  # 1000 - 200
        assert result.output_tokens == 500
        assert result.cache_read_tokens == 200

    def test_codex_style(self):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            input_tokens_details=SimpleNamespace(cached_tokens=300, cache_creation_tokens=0),
            output_tokens_details=None,
        )
        result = normalize_usage(usage, api_mode="codex_responses")
        assert result.input_tokens == 700  # 1000 - 300
        assert result.cache_read_tokens == 300

    def test_none_usage(self):
        result = normalize_usage(None)
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_reasoning_tokens(self):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=200,
            prompt_tokens_details=None,
            output_tokens_details=SimpleNamespace(reasoning_tokens=50),
        )
        result = normalize_usage(usage)
        assert result.reasoning_tokens == 50


class TestEstimateUsageCost:
    def test_known_model(self):
        usage = CanonicalUsage(input_tokens=1000, output_tokens=500)
        result = estimate_usage_cost("gpt-4o", usage, provider="openai")
        assert result.status == "estimated"
        assert result.amount_usd is not None
        assert result.amount_usd > 0

    def test_unknown_model(self):
        usage = CanonicalUsage(input_tokens=1000, output_tokens=500)
        result = estimate_usage_cost("unknown-model", usage)
        assert result.status == "unknown"
        assert result.amount_usd is None

    def test_cache_aware(self):
        usage = CanonicalUsage(
            input_tokens=500, output_tokens=200,
            cache_read_tokens=300, cache_write_tokens=100,
        )
        result = estimate_usage_cost(
            "claude-sonnet-4-20250514", usage, provider="anthropic",
        )
        assert result.status == "estimated"
        assert result.amount_usd > 0


class TestHasKnownPricing:
    def test_known(self):
        assert has_known_pricing("gpt-4o", provider="openai")

    def test_unknown(self):
        assert not has_known_pricing("random-model-xyz")


class TestFormatHelpers:
    def test_duration_compact(self):
        assert format_duration_compact(30) == "30s"
        assert format_duration_compact(90) == "2m"
        assert format_duration_compact(3660) == "1h 1m"
        assert format_duration_compact(86400) == "1.0d"

    def test_token_count_compact(self):
        assert format_token_count_compact(500) == "500"
        assert format_token_count_compact(1500) == "1.5K"
        assert format_token_count_compact(1_500_000) == "1.5M"
        assert format_token_count_compact(-5000) == "-5K"
