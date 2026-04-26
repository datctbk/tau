"""Optional capability provider wrappers for tau core.

This module centralizes optional infrastructure features (routing, pricing,
prompt caching, rate-limit parsing) so the core agent loop can stay minimal
and avoid hard imports unless explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tau.core.types import AgentConfig, Message


@dataclass
class AgentCapabilities:
    """Runtime capability hooks used by the core agent loop."""

    apply_prompt_caching: Callable[[list[Message], str, str], list[Message]]
    parse_rate_limits: Callable[[Any, str], Any | None]
    resolve_route: Callable[[str, dict, dict], dict | None]
    estimate_usage_cost: Callable[[str, dict[str, int], str], float]


def _enabled(config: AgentConfig, name: str, default: bool = True) -> bool:
    caps = getattr(config, "capabilities", None) or {}
    return bool(caps.get(name, default))


def build_capabilities(config: AgentConfig) -> AgentCapabilities:
    """Build capability hooks using lazy imports for enabled features only."""

    def _no_prompt_cache(messages: list[Message], _model: str, _provider_name: str) -> list[Message]:
        return messages

    def _no_rate_limit(_headers: Any, _provider_name: str) -> Any | None:
        return None

    def _no_route(_user_input: str, _routing_cfg: dict, _primary: dict) -> dict | None:
        return None

    def _no_pricing(_model: str, _usage_dict: dict[str, int], _provider_name: str) -> float:
        return 0.0

    apply_prompt_caching = _no_prompt_cache
    if _enabled(config, "prompt_caching", True):
        from tau.core.prompt_caching import apply_anthropic_cache_control

        def _prompt_cache(messages: list[Message], model: str, provider_name: str) -> list[Message]:
            if provider_name != "openai" or "claude" not in (model or "").lower():
                return messages
            cached = apply_anthropic_cache_control(
                [m.to_dict() for m in messages],
                cache_ttl="5m",
            )
            return [Message.from_dict(m) for m in cached]

        apply_prompt_caching = _prompt_cache

    parse_rate_limits = _no_rate_limit
    if _enabled(config, "rate_limit_tracking", True):
        from tau.core.rate_limit_tracker import parse_rate_limit_headers

        def _parse_rate(headers: Any, provider_name: str) -> Any | None:
            return parse_rate_limit_headers(headers, provider=provider_name)

        parse_rate_limits = _parse_rate

    resolve_route = _no_route
    if _enabled(config, "smart_routing", True):
        from tau.core.smart_routing import resolve_turn_route

        def _route(user_input: str, routing_cfg: dict, primary: dict) -> dict | None:
            return resolve_turn_route(user_input, routing_config=routing_cfg, primary=primary)

        resolve_route = _route

    estimate_usage_cost = _no_pricing
    if _enabled(config, "usage_pricing", True):
        from tau.core.usage_pricing import CanonicalUsage, estimate_usage_cost as _estimate_usage_cost

        def _estimate(model: str, usage_dict: dict[str, int], provider_name: str) -> float:
            usage = CanonicalUsage(**(usage_dict or {}))
            result = _estimate_usage_cost(model, usage, provider=provider_name)
            return float(result.amount_usd or 0.0)

        estimate_usage_cost = _estimate

    return AgentCapabilities(
        apply_prompt_caching=apply_prompt_caching,
        parse_rate_limits=parse_rate_limits,
        resolve_route=resolve_route,
        estimate_usage_cost=estimate_usage_cost,
    )

