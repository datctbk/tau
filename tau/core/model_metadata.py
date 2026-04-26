"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by context
compression and pre-flight context checks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Provider names that can appear as a "provider:" prefix before a model ID.
# Only these are stripped — Ollama-style "model:tag" colons are preserved.
_PROVIDER_PREFIXES: frozenset = frozenset({
    "openrouter", "openai", "anthropic", "gemini", "deepseek",
    "google", "google-gemini", "google-ai-studio",
    "custom", "local",
    "kimi", "moonshot",
    "xai", "x-ai", "grok",
    "alibaba", "dashscope", "qwen",
    "minimax",
})

_OLLAMA_TAG_PATTERN = re.compile(
    r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)",
    re.IGNORECASE,
)


def _strip_provider_prefix(model: str) -> str:
    """Strip a recognised provider prefix from a model string.

    ``"local:my-model"`` → ``"my-model"``
    ``"qwen3.5:27b"``   → ``"qwen3.5:27b"``  (unchanged — Ollama model:tag)
    """
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    prefix_lower = prefix.strip().lower()
    if prefix_lower in _PROVIDER_PREFIXES:
        # Don't strip if suffix looks like an Ollama tag
        if _OLLAMA_TAG_PATTERN.match(suffix.strip()):
            return model
        return suffix
    return model


# ── Context Length Defaults ─────────────────────────────────────────────

# Thin fallback defaults — only broad model family patterns.
# For provider-specific context lengths, query the /models endpoint.
DEFAULT_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude
    "claude-opus-4.6": 1000000,
    "claude-sonnet-4.6": 1000000,
    "claude": 200000,
    # OpenAI GPT-5 family
    "gpt-5.4": 1050000,
    "gpt-5": 400000,
    "gpt-4.1": 1047576,
    "gpt-4": 128000,
    # Google
    "gemini": 1048576,
    "gemma-4": 256000,
    "gemma-3": 131072,
    "gemma": 8192,
    # DeepSeek
    "deepseek": 128000,
    # Meta
    "llama": 131072,
    # Qwen
    "qwen3-coder-plus": 1000000,
    "qwen3-coder": 262144,
    "qwen": 131072,
    # MiniMax
    "minimax": 204800,
    # xAI Grok
    "grok-4-1-fast": 2000000,
    "grok-4-fast": 2000000,
    "grok-4.20": 2000000,
    "grok-4": 256000,
    "grok-3": 131072,
    "grok-2": 131072,
    "grok": 131072,
    # Kimi
    "kimi": 262144,
}

# Default context length when no detection method succeeds.
DEFAULT_FALLBACK_CONTEXT = 128_000

# Minimum context length required to run the agent effectively.
MINIMUM_CONTEXT_LENGTH = 64_000

# Descending tiers for context length probing.
CONTEXT_PROBE_TIERS = [
    128_000,
    64_000,
    32_000,
    16_000,
    8_000,
]

# Keys commonly used by providers in /models API responses
_CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
)

_MAX_COMPLETION_KEYS = (
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
)

# Local server hostnames
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
_CONTAINER_LOCAL_SUFFIXES = (
    ".docker.internal",
    ".containers.internal",
    ".lima.internal",
)

# URL-to-provider mapping for context length inference
_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.moonshot.ai": "kimi",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "gemini",
    "api.deepseek.com": "deepseek",
    "api.x.ai": "xai",
    "dashscope.aliyuncs.com": "alibaba",
}


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """Infer a provider name from a base URL."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def is_local_endpoint(base_url: str) -> bool:
    """Return True if base_url points to a local machine (localhost / RFC-1918)."""
    import ipaddress as _ipaddress

    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    if host in _LOCAL_HOSTS:
        return True
    if any(host.endswith(suffix) for suffix in _CONTAINER_LOCAL_SUFFIXES):
        return True
    try:
        addr = _ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        pass
    return False


def _coerce_reasonable_int(
    value: Any, minimum: int = 1024, maximum: int = 10_000_000,
) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        result = int(value)
    except (TypeError, ValueError):
        return None
    if minimum <= result <= maximum:
        return result
    return None


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _extract_first_int(payload: Dict[str, Any], keys: tuple) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() not in keyset:
                continue
            coerced = _coerce_reasonable_int(value)
            if coerced is not None:
                return coerced
    return None


def _extract_context_length(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _CONTEXT_LENGTH_KEYS)


def _extract_max_completion_tokens(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _MAX_COMPLETION_KEYS)


def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pricing information from a model metadata dict."""
    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token"),
        "completion": ("completion", "output", "output_cost_per_token"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        if not any(
            any(alias in normalized for alias in aliases)
            for aliases in alias_map.values()
        ):
            continue
        pricing: Dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in (None, ""):
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            return pricing
    return {}


def get_context_length(
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> int:
    """Resolve context length for a model, using multi-tier fallback.

    Resolution order:
      1. Exact match in DEFAULT_CONTEXT_LENGTHS
      2. Substring match (longest-first) in DEFAULT_CONTEXT_LENGTHS
      3. DEFAULT_FALLBACK_CONTEXT
    """
    bare_model = _strip_provider_prefix(model).lower()

    # 1. Exact match
    if bare_model in DEFAULT_CONTEXT_LENGTHS:
        return DEFAULT_CONTEXT_LENGTHS[bare_model]

    # 2. Substring match — longest key first to prefer specific over general
    sorted_keys = sorted(DEFAULT_CONTEXT_LENGTHS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in bare_model:
            return DEFAULT_CONTEXT_LENGTHS[key]

    return DEFAULT_FALLBACK_CONTEXT


def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Return the next lower probe tier, or None if already at minimum."""
    for tier in CONTEXT_PROBE_TIERS:
        if tier < current_length:
            return tier
    return None


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Try to extract the actual context limit from an API error message.

    Many providers include the limit in their error text, e.g.:
      - "maximum context length is 32768 tokens"
      - "context_length_exceeded: 131072"
    """
    error_lower = error_msg.lower()
    patterns = [
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context[_\s]*(?:length|size|window)[_\s]*(?:exceeded|error)?[:\s]+(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            limit = int(match.group(1))
            if 1024 <= limit <= 10_000_000:
                return limit
    return None
