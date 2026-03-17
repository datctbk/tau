"""Context manager — maintains the message window for an agent turn."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from tau.core.types import AgentConfig, Message, Role

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rough token estimator (no tiktoken dependency required at runtime)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """~4 chars per token — good enough for budget enforcement."""
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
    """Placeholder — falls back to sliding window until a provider is wired in."""

    def __init__(self) -> None:
        self._fallback = SlidingWindowStrategy()

    def trim(self, messages: list[Message], budget: int) -> list[Message]:
        # TODO(P4): call provider to summarise old messages
        logger.debug("SummariseStrategy: falling back to sliding window")
        return self._fallback.trim(messages, budget)


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
