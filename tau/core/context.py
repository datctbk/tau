"""Context manager — maintains the message window for an agent turn."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from tau.core.types import AgentConfig, Message, Role

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimator — tiktoken when available, char/4 fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None  # type: ignore[assignment]


def _estimate_tokens(text: str) -> int:
    if _enc is not None:
        return max(1, len(_enc.encode(text, disallowed_special=())))
    return max(1, len(text) // 4)


def _messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += _estimate_tokens(m.content or "")
        if m.tool_calls:
            for tc in m.tool_calls:
                total += _estimate_tokens(str(tc.arguments))
    return total


# ---------------------------------------------------------------------------
# Module-level summarise config (set by CLI at startup via configure_context)
# ---------------------------------------------------------------------------

_summarise_config: dict[str, Any] = {
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen3.5:9b",
}


def configure_context(ollama_base_url: str, ollama_model: str) -> None:
    _summarise_config["ollama_base_url"] = ollama_base_url
    _summarise_config["ollama_model"] = ollama_model


# ---------------------------------------------------------------------------
# Trim strategies
# ---------------------------------------------------------------------------

class TrimStrategy(Protocol):
    def trim(self, messages: list[Message], budget: int) -> list[Message]: ...


class SlidingWindowStrategy:
    """Drop the oldest non-system messages until under budget."""

    def trim(self, messages: list[Message], budget: int) -> list[Message]:
        if _messages_tokens(messages) <= budget:
            return messages

        system = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        while non_system and _messages_tokens(system + non_system) > budget:
            dropped = non_system.pop(0)
            logger.debug("SlidingWindow: dropped message role=%s", dropped.role)

        return system + non_system


class SummariseStrategy:
    """Summarise old messages via a lightweight Ollama/OpenAI call when over budget."""

    # Keep the most recent N messages untouched — only summarise older ones
    _KEEP_RECENT = 6

    def __init__(self) -> None:
        self._fallback = SlidingWindowStrategy()

    def trim(self, messages: list[Message], budget: int) -> list[Message]:
        if _messages_tokens(messages) <= budget:
            return messages

        system = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= self._KEEP_RECENT:
            return self._fallback.trim(messages, budget)

        to_summarise = non_system[: -self._KEEP_RECENT]
        keep_recent = non_system[-self._KEEP_RECENT:]

        try:
            summary_text = self._call_summary(to_summarise)
        except Exception as exc:
            logger.warning("SummariseStrategy: summary call raised (%s), falling back", exc)
            return self._fallback.trim(messages, budget)

        summary_msg = Message(
            role="user",
            content=f"[Earlier conversation summary]\n{summary_text}",
        )
        trimmed = system + [summary_msg] + keep_recent

        if _messages_tokens(trimmed) > budget:
            logger.warning("SummariseStrategy: summary still over budget, applying sliding window")
            trimmed = self._fallback.trim(trimmed, budget)

        logger.debug("SummariseStrategy: summarised %d messages", len(to_summarise))
        return trimmed

    def _call_summary(self, messages: list[Message]) -> str:
        """Call a local Ollama model to produce a concise summary.
        Falls back to a plain concatenation if the call fails."""
        transcript = "\n".join(
            f"{m.role.upper()}: {(m.content or '')[:500]}"
            for m in messages
        )
        prompt = (
            "Summarise the following conversation history in 3-5 sentences, "
            "preserving all important facts, decisions and code changes:\n\n"
            + transcript
        )
        try:
            import httpx
            base_url = _summarise_config["ollama_base_url"].rstrip("/")
            model = _summarise_config["ollama_model"]
            resp = httpx.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("response", transcript)
        except Exception as exc:
            logger.warning("SummariseStrategy: summary call failed (%s), using transcript", exc)
            return transcript


_STRATEGIES: dict[str, TrimStrategy] = {
    "sliding_window": SlidingWindowStrategy(),
    "summarise": SummariseStrategy(),
}


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._messages: list[Message] = []
        self._strategy: TrimStrategy = _STRATEGIES.get(
            config.trim_strategy, SlidingWindowStrategy()
        )
        # Inject system prompt
        if config.system_prompt:
            self._messages.append(Message(role="system", content=config.system_prompt))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, msg: Message) -> None:
        self._messages.append(msg)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def token_count(self) -> int:
        return _messages_tokens(self._messages)

    def trim(self) -> None:
        # Reserve 20% of the window for the response, capped to half the budget
        headroom = min(self._config.max_tokens // 2, max(64, self._config.max_tokens // 5))
        budget = self._config.max_tokens - headroom
        if budget <= 0:
            return
        self._messages = self._strategy.trim(self._messages, budget)

    def snapshot(self) -> list[dict]:
        """Return a JSON-serialisable copy of the current message list."""
        return [m.to_dict() for m in self._messages]

    def restore(self, raw_messages: list[dict]) -> None:
        """Restore messages from a persisted snapshot (skips system prompt)."""
        self._messages = [
            m for m in self._messages if m.role == "system"
        ] + [
            Message.from_dict(d)
            for d in raw_messages
            if d.get("role") != "system"
        ]

    def inject_prompt_fragment(self, fragment: str) -> None:
        """Append extra context to the system message (used by skills)."""
        for m in self._messages:
            if m.role == "system":
                m.content = m.content.rstrip() + "\n\n" + fragment
                return
        # No system message yet — create one
        self._messages.insert(0, Message(role="system", content=fragment))
