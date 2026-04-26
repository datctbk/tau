"""Smart model routing — route simple queries to cheap models.

Conservative by design: if the message has signs of code, tool usage,
debugging, or long-form work, keeps the primary (expensive) model.
Only short, simple, conversational messages get routed to the cheap model.

Configuration (in config.yaml):
  smart_model_routing:
    enabled: true
    max_simple_chars: 160
    max_simple_words: 28
    cheap_model:
      provider: "openai"
      model: "gpt-4o-mini"
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Keywords that indicate a complex task requiring the primary model
_COMPLEX_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce a value to bool, handling string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_cheap_model_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Returns None if the message appears complex or routing is disabled.
    Returns a dict with provider/model/routing_reason keys if cheap is appropriate.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    # Length checks
    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None

    # Multi-line → likely code or structured content
    if text.count("\n") > 1:
        return None

    # Code fences or inline code
    if "```" in text or "`" in text:
        return None

    # URLs → likely research/analysis
    if _URL_RE.search(text):
        return None

    # Complex keyword detection
    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def resolve_turn_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Args:
        user_message: The user's input text.
        routing_config: Smart routing configuration dict.
        primary: The default model/runtime configuration.

    Returns:
        Dict with model, runtime, label, and signature fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)

    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
            ),
        }

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": route.get("api_key") or primary.get("api_key"),
            "base_url": route.get("base_url") or primary.get("base_url"),
            "provider": route.get("provider"),
        },
        "label": f"smart route → {route.get('model')} ({route.get('provider')})",
        "signature": (
            route.get("model"),
            route.get("provider"),
            route.get("base_url") or primary.get("base_url"),
        ),
    }
