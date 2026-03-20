"""Context manager — maintains the message window for an agent turn."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from tau.core.types import AgentConfig, CompactionEntry, Message, Role

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
# Compactor — threshold-based auto-compaction using the active provider
# ---------------------------------------------------------------------------
_COMPACTION_SYSTEM = (
    "You are a conversation compactor. "
    "Produce a dense, structured summary of the conversation below. "
    "Preserve ALL key facts, decisions, file paths, code snippets, errors, "
    "and any unresolved tasks. Be thorough — this summary replaces the original history."
)

_COMPACTION_PROMPT = (
    "Summarise the following conversation history. "
    "Keep every important technical detail, file name, code change, "
    "decision, and pending task:\n\n{transcript}"
)

# Minimum non-system messages required before compaction is attempted
_MIN_MESSAGES_TO_COMPACT = 4
# How many recent messages to always keep verbatim after compaction
_KEEP_RECENT_AFTER_COMPACT = 4


class Compactor:
    """Checks whether context has hit the threshold and compacts it if so."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._overflow_recovery_attempted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compact(self, messages: list[Message]) -> bool:
        """Return True when current token count exceeds the threshold."""
        if not self._config.compaction_enabled:
            return False
        current = _messages_tokens(messages)
        threshold = int(self._config.max_tokens * self._config.compaction_threshold)
        return current >= threshold

    def is_overflow_error(self, error_message: str) -> bool:
        """Return True when the provider error looks like a context-length overflow."""
        markers = (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
            "reduce the length",
            "reduce your prompt",
            "prompt is too long",
        )
        lower = error_message.lower()
        return any(m in lower for m in markers)

    def compact(
        self,
        messages: list[Message],
        provider: Any,          # BaseProvider — avoid circular import
        tokens_before: int,
    ) -> tuple[list[Message], CompactionEntry]:
        """
        Compact *messages* using *provider*.

        Returns:
            (new_messages, CompactionEntry)
        """
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # Split: everything except the most recent _KEEP_RECENT_AFTER_COMPACT messages
        to_summarise = non_system[:-_KEEP_RECENT_AFTER_COMPACT] if len(non_system) > _KEEP_RECENT_AFTER_COMPACT else non_system
        keep_recent = non_system[-_KEEP_RECENT_AFTER_COMPACT:] if len(non_system) > _KEEP_RECENT_AFTER_COMPACT else []

        if len(to_summarise) < _MIN_MESSAGES_TO_COMPACT:
            # Not enough history to summarise — return unchanged
            raise ValueError("Not enough messages to compact (need at least %d non-system messages)" % (_MIN_MESSAGES_TO_COMPACT + _KEEP_RECENT_AFTER_COMPACT))

        transcript = self._build_transcript(to_summarise)
        summary = self._call_summary(transcript, provider)

        summary_msg = Message(
            role="user",
            content=f"[Compacted conversation summary — earlier history replaced]\n{summary}",
        )
        new_messages = system_msgs + [summary_msg] + keep_recent
        tokens_after = _messages_tokens(new_messages)

        entry = CompactionEntry(
            summary=summary,
            tokens_before=tokens_before,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.debug(
            "Compactor: %d → %d tokens (summarised %d messages, kept %d)",
            tokens_before, tokens_after, len(to_summarise), len(keep_recent),
        )
        return new_messages, entry

    # ------------------------------------------------------------------
    # Overflow recovery helpers
    # ------------------------------------------------------------------

    def mark_overflow_recovery_attempted(self) -> None:
        self._overflow_recovery_attempted = True

    def overflow_recovery_attempted(self) -> bool:
        return self._overflow_recovery_attempted

    def reset_overflow_flag(self) -> None:
        self._overflow_recovery_attempted = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_transcript(self, messages: list[Message]) -> str:
        parts: list[str] = []
        for m in messages:
            role_label = m.role.upper()
            content = (m.content or "").strip()
            if m.tool_calls:
                calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
                content = f"{content}\n[tool calls: {calls}]".strip()
            parts.append(f"{role_label}: {content}")
        return "\n\n".join(parts)

    def _call_summary(self, transcript: str, provider: Any) -> str:
        """Ask the active provider for a compaction summary."""
        prompt = _COMPACTION_PROMPT.format(transcript=transcript)
        compaction_messages = [
            Message(role="system", content=_COMPACTION_SYSTEM),
            Message(role="user", content=prompt),
        ]
        try:
            raw = provider.chat(messages=compaction_messages, tools=[])
            # Handle both streaming and non-streaming responses
            from tau.core.types import ProviderResponse, TextDelta
            if isinstance(raw, ProviderResponse):
                return (raw.content or "").strip() or transcript
            # Streaming — collect all deltas
            text_parts: list[str] = []
            for item in raw:
                if isinstance(item, TextDelta):
                    text_parts.append(item.text)
                elif isinstance(item, ProviderResponse):
                    if item.content:
                        return item.content.strip()
            result = "".join(text_parts).strip()
            return result or transcript
        except Exception as exc:
            logger.warning("Compactor: summary call failed (%s), using transcript excerpt", exc)
            # Truncate transcript as fallback
            return transcript[:2000] + ("\n…[truncated]" if len(transcript) > 2000 else "")


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
        self.compactor = Compactor(config)

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
