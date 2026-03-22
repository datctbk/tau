"""Agent loop — the heart of tau."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ToolCallEvent,
    ToolResultEvent,
    TurnComplete,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    ToolResult,
)

if TYPE_CHECKING:
    from tau.core.context import ContextManager
    from tau.core.extension import ExtensionRegistry
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
        ext_registry: "ExtensionRegistry | None" = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._registry = registry
        self._ext_registry = ext_registry
        self._context = context
        self._session = session
        self._session_manager = session_manager
        self._steering = steering

    # ------------------------------------------------------------------
    # Tool dispatch (parallel or sequential)
    # ------------------------------------------------------------------
    def _dispatch_tools(
        self, calls: list[ToolCall]
    ) -> list[tuple[ToolCall, ToolResult]]:
        """
        Dispatch a batch of tool calls.
        When parallel_tools is True (default) all calls run concurrently in a
        thread pool; results are returned in the **original request order**.
        Sequential fallback is used when the config disables parallelism or
        only one call is present.
        """
        if not self._config.parallel_tools or len(calls) <= 1:
            return [(call, self._registry.dispatch(call)) for call in calls]

        max_workers = min(self._config.parallel_tools_max_workers, len(calls))
        results: dict[int, ToolResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._registry.dispatch, call): idx
                for idx, call in enumerate(calls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    call = calls[idx]
                    logger.exception("Parallel dispatch error for tool %r", call.name)
                    results[idx] = ToolResult(
                        tool_call_id=call.id,
                        content=f"Error in tool {call.name!r}: {exc}",
                        is_error=True,
                    )

        # Reconstruct in original order
        return [(calls[i], results[i]) for i in range(len(calls))]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str, images: list[str] | None = None) -> Generator[Event, None, None]:
        """Process one user message; yield Events for the CLI to render."""
        self._context.add_message(Message(role="user", content=user_input, images=images))
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

            # Accumulate usage
            cu = getattr(self._session, "cumulative_usage", None)
            if cu is not None:
                cu["input_tokens"] += response.usage.input_tokens
                cu["output_tokens"] += response.usage.output_tokens
                cu["cache_read_tokens"] += response.usage.cache_read_tokens
                cu["cache_write_tokens"] += response.usage.cache_write_tokens

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

            # ── Before-hooks (sequential, main thread) ────────────────
            calls = response.tool_calls
            blocked: dict[int, ToolResult] = {}   # idx → blocked result
            if self._ext_registry:
                for idx, call in enumerate(calls):
                    yield ToolCallEvent(call=call)
                    ctx = BeforeToolCallContext(tool_call=call, agent=self)
                    res = self._ext_registry.fire_before_tool_call(ctx)
                    if res and res.block:
                        err_msg = res.reason or f"Blocked by extension: {call.name}"
                        blocked[idx] = ToolResult(
                            tool_call_id=call.id, content=err_msg, is_error=True
                        )
            else:
                for call in calls:
                    yield ToolCallEvent(call=call)

            # ── Parallel dispatch (only non-blocked calls) ─────────────
            runnable_idx = [i for i in range(len(calls)) if i not in blocked]
            runnable_calls = [calls[i] for i in runnable_idx]

            if runnable_calls:
                pairs = self._dispatch_tools(runnable_calls)
                dispatched: dict[int, ToolResult] = {
                    runnable_idx[j]: result for j, (_, result) in enumerate(pairs)
                }
            else:
                dispatched = {}

            # ── After-hooks + emit results (sequential, original order) ─
            for idx, call in enumerate(calls):
                result = blocked.get(idx) or dispatched.get(idx)
                if result is None:
                    continue  # should not happen
                if self._ext_registry and idx not in blocked:
                    ctx_after = AfterToolCallContext(tool_call=call, result=result, agent=self)
                    self._ext_registry.fire_after_tool_call(ctx_after)
                yield ToolResultEvent(result=result)
                self._context.add_message(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    name=call.name,
                ))

            parallel_count = len(runnable_calls)
            if parallel_count > 1:
                logger.debug("Ran %d tool calls in parallel", parallel_count)

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
