"""Agent loop — the heart of tau."""

from __future__ import annotations

import logging
import time
from typing import Generator, Iterator, TYPE_CHECKING

from tau.core.types import (
    AgentConfig,
    CompactionEvent,
    ErrorEvent,
    Event,
    Message,
    ProviderResponse,
    RetryEvent,
    SteerEvent,
    TextChunk,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    TurnComplete,
)

if TYPE_CHECKING:
    from tau.core.context import ContextManager
    from tau.core.session import Session, SessionManager
    from tau.core.steering import SteeringChannel
    from tau.core.tool_registry import ToolRegistry
    from tau.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Error substrings that are safe to retry (transient / server-side).
# Context-overflow errors are NOT included — those are handled by compaction.
_RETRYABLE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "overloaded",
    "overload",
    "503",
    "502",
    "500",
    "service unavailable",
    "server error",
    "internal error",
    "bad gateway",
    "timeout",
    "timed out",
    "connection error",
    "connection refused",
    "network error",
    "fetch failed",
    "socket",
    "reset",
    "temporarily unavailable",
    "retry",
)

# Sentinels returned by _call_with_retry
_OVERFLOW_RETRY_SENTINEL = object()   # compact succeeded → retry the turn
_OVERFLOW_FAILED_SENTINEL = object()  # compact failed → error already emitted, stop

# Sentinel returned by _stream() when a steer interrupted the response
_STEER_SENTINEL = object()


class MaxTurnsReachedError(Exception):
    pass


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        provider: "BaseProvider",
        registry: "ToolRegistry",
        context: "ContextManager",
        session: "Session",
        session_manager: "SessionManager",
        steering: "SteeringChannel | None" = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._registry = registry
        self._context = context
        self._session = session
        self._session_manager = session_manager
        self._steering = steering

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> Generator[Event, None, None]:
        """Process one user message; yield Events for the CLI to render."""
        self._context.add_message(Message(role="user", content=user_input))
        self._context.compactor.reset_overflow_flag()

        turns = 0
        while turns < self._config.max_turns:
            turns += 1
            logger.debug("Agent turn %d", turns)
            self._context.trim()

            # Provider call — with auto-retry and overflow handling
            raw, error_event = yield from self._call_with_retry()
            if error_event is not None:
                yield error_event
                return
            # Overflow was compacted — restart the turn
            if raw is _OVERFLOW_RETRY_SENTINEL:
                continue
            # Overflow already emitted its error — stop cleanly
            if raw is _OVERFLOW_FAILED_SENTINEL:
                return
            if raw is None:
                yield ErrorEvent(message="Provider returned no response.")
                return

            # Stream deltas live; collect the final ProviderResponse
            response, had_deltas, steer_msg = yield from self._stream(raw)

            # Mid-stream steer: discard current response, start a new turn
            if steer_msg is not None:
                discarded = len(had_deltas) if isinstance(had_deltas, list) else 0
                yield SteerEvent(new_input=steer_msg, discarded_tokens=discarded)
                self._context.add_message(Message(role="user", content=steer_msg))
                self._context.compactor.reset_overflow_flag()
                turns = 0   # reset turn counter for the new steering request
                continue

            if response is None:
                yield ErrorEvent(message="Provider returned no response.")
                return

            # Add completed assistant message to context
            self._context.add_message(Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls or None,
            ))

            # Blocking mode (no streaming) — emit full text now
            if response.content and not had_deltas:
                yield TextChunk(text=response.content)

            # No tool calls → done; check compaction then check follow-up queue
            if not response.tool_calls:
                yield TurnComplete(usage=response.usage)
                yield from self._maybe_compact()
                self._persist()

                # Drain one follow-up from the queue and loop back
                next_input = self._steering.dequeue() if self._steering else None
                if next_input is not None:
                    logger.debug("Consuming queued follow-up: %r", next_input[:60])
                    self._context.add_message(Message(role="user", content=next_input))
                    self._context.compactor.reset_overflow_flag()
                    turns = 0
                    continue
                return

            # Dispatch each tool call
            for call in response.tool_calls:
                yield ToolCallEvent(call=call)
                result = self._registry.dispatch(call)
                yield ToolResultEvent(result=result)
                self._context.add_message(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    name=call.name,
                ))

            # Check compaction after tool round-trip (before next LLM call)
            yield from self._maybe_compact()

        yield ErrorEvent(message=f"Reached max_turns limit ({self._config.max_turns}).")
        self._persist()

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _is_retryable_error(self, message: str) -> bool:
        """Return True for transient errors that should be retried."""
        # Never retry context-overflow — that's handled by compaction
        if self._context.compactor.is_overflow_error(message):
            return False
        lower = message.lower()
        return any(p in lower for p in _RETRYABLE_PATTERNS)

    def _call_with_retry(
        self,
    ) -> Generator[Event, None, tuple[object | None, ErrorEvent | None]]:
        """
        Call the provider with exponential-backoff retry for transient errors.

        Yields RetryEvent before each retry sleep.
        Returns (raw_response, None) on success or (None, ErrorEvent) on failure.
        Also handles context-overflow by delegating to _try_compact_on_overflow;
        returns ("retry_turn", None) to signal the caller should loop again —
        but since we restart from within the while loop in run(), we handle this
        by re-raising as a special sentinel and catching in run() via a flag.
        """
        cfg = self._config
        max_attempts = cfg.retry_max_attempts if cfg.retry_enabled else 1
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            try:
                raw = self._provider.chat(
                    messages=self._context.get_messages(),
                    tools=self._registry.all_definitions(),
                )
                return raw, None

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("Provider error (attempt %d/%d): %s", attempt, max_attempts, last_error)

                # Context overflow → compact & signal a full turn retry
                if self._context.compactor.is_overflow_error(last_error):
                    compact_result = yield from self._try_compact_on_overflow(last_error)
                    if compact_result == "retry":
                        return _OVERFLOW_RETRY_SENTINEL, None
                    # "failed" — error already emitted, signal run() to stop cleanly
                    return _OVERFLOW_FAILED_SENTINEL, None

                # Not retryable or last attempt → stop
                if not self._is_retryable_error(last_error) or attempt == max_attempts:
                    return None, ErrorEvent(message=f"Provider error: {exc}")

                # Compute exponential backoff delay
                delay = cfg.retry_base_delay * (2 ** (attempt - 1))
                yield RetryEvent(
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    delay=delay,
                    error=last_error,
                )
                logger.info("Retrying in %.1fs (attempt %d/%d)…", delay, attempt + 1, max_attempts)
                time.sleep(delay)

        # Should not be reached, but be safe
        return None, ErrorEvent(message=f"Provider error after {max_attempts} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Compaction helpers
    # ------------------------------------------------------------------

    def _maybe_compact(self) -> Generator[Event, None, None]:
        """Compact if the context has crossed the threshold; emit CompactionEvents."""
        messages = self._context.get_messages()
        if not self._context.compactor.should_compact(messages):
            return

        tokens_before = self._context.token_count()
        yield CompactionEvent(stage="start", tokens_before=tokens_before)

        try:
            new_messages, entry = self._context.compactor.compact(
                messages, self._provider, tokens_before
            )
        except ValueError as exc:
            logger.debug("Compaction skipped: %s", exc)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-compaction failed: %s", exc)
            yield CompactionEvent(stage="end", tokens_before=tokens_before, error=str(exc))
            return

        self._context.restore([m.to_dict() for m in new_messages if m.role != "system"])
        tokens_after = self._context.token_count()
        self._session_manager.append_compaction(self._session, entry)

        yield CompactionEvent(
            stage="end",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary=entry.summary,
        )

    def _try_compact_on_overflow(self, error_msg: str) -> Generator[Event, None, str]:
        """
        Called when the provider returns a context-overflow error.
        Tries to compact once and signals 'retry' or 'failed'.
        """
        compactor = self._context.compactor
        if compactor.overflow_recovery_attempted():
            yield ErrorEvent(
                message="Context overflow: compaction already attempted once. "
                        "Try reducing context or switching to a larger-context model."
            )
            return "failed"

        compactor.mark_overflow_recovery_attempted()
        messages = self._context.get_messages()
        tokens_before = self._context.token_count()

        yield CompactionEvent(stage="start", tokens_before=tokens_before)

        try:
            new_messages, entry = compactor.compact(messages, self._provider, tokens_before)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Overflow compaction failed: %s", exc)
            yield CompactionEvent(stage="end", tokens_before=tokens_before, error=str(exc))
            yield ErrorEvent(message=f"Context overflow and compaction failed: {exc}")
            return "failed"

        self._context.restore([m.to_dict() for m in new_messages if m.role != "system"])
        tokens_after = self._context.token_count()
        self._session_manager.append_compaction(self._session, entry)

        yield CompactionEvent(
            stage="end",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary=entry.summary,
        )
        return "retry"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream(
        self, raw
    ) -> Generator[Event, None, tuple[ProviderResponse | None, bool, str | None]]:
        """
        Yield TextDelta events live; return (ProviderResponse, had_deltas, steer_msg).

        steer_msg is non-None when a mid-stream steer interrupted the response.
        had_deltas is True when at least one TextDelta was emitted.
        """
        if isinstance(raw, ProviderResponse):
            return raw, False, None

        had_deltas = False
        response: ProviderResponse | None = None
        delta_count = 0

        try:
            for item in raw:
                # Check for a steer before processing each item
                steer_msg = (
                    self._steering.consume_steer() if self._steering else None
                )
                if steer_msg is not None:
                    logger.debug(
                        "Steer received after %d deltas — discarding stream", delta_count
                    )
                    # Drain the remaining generator so the provider connection
                    # is released cleanly (best-effort)
                    try:
                        for _ in raw:
                            pass
                    except Exception:  # noqa: BLE001
                        pass
                    return None, had_deltas, steer_msg

                if isinstance(item, TextDelta):
                    had_deltas = True
                    delta_count += 1
                    yield item
                elif isinstance(item, ProviderResponse):
                    response = item
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error consuming provider stream")
            yield ErrorEvent(message=f"Stream error: {exc}")
            return None, had_deltas, None

        return response, had_deltas, None

    def _persist(self) -> None:
        try:
            self._session_manager.save(
                self._session,
                messages=self._context.snapshot(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist session: %s", exc)
