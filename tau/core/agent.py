"""Agent loop — the heart of tau."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator, Iterator, TYPE_CHECKING

from tau.core.types import (
    AgentConfig,
    CompactionEvent,
    CostLimitExceeded,
    PolicyDecisionEvent,
    ErrorEvent,
    Event,
    Message,
    ProviderResponse,
    RetryEvent,
    SteerEvent,
    ToolDefinition,
    TextChunk,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    TurnComplete,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    ToolResult,
)
from tau.core import trace as _trace
from tau.core.audit import append_audit_record
from tau.core.assistant_events import append_assistant_event, make_assistant_event
from tau.core.policy import DefaultToolPolicyHook
from tau.core.context import _messages_tokens

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

_EMPTY_RESPONSE_NUDGE = "Please continue and provide at least a short textual response."
_EMPTY_RESPONSE_MAX_RETRIES = 1


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
        cost_calculator: "Callable[[str, object], float] | None" = None,
        policy_approval_hook: "Callable[[str], bool] | None" = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._registry = registry
        self._ext_registry = ext_registry
        self._context = context
        self._session = session
        self._session_manager = session_manager
        self._steering = steering
        self._cost_calculator = cost_calculator
        self._policy_approval_hook = policy_approval_hook
        self._policy_hook = (
            DefaultToolPolicyHook(profile=config.policy_profile)
            if config.policy_enabled
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {x for x in re.findall(r"[a-z0-9_]+", (text or "").lower()) if x}

    def _trim_messages_for_prompt_budget(self, messages: list[Message], budget: int) -> list[Message]:
        if _messages_tokens(messages) <= budget:
            return messages
        system = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        if not non_system:
            return system

        # Keep a contiguous recent window from the tail.
        kept_rev: list[Message] = []
        for msg in reversed(non_system):
            trial_rev = kept_rev + [msg]
            trial = system + list(reversed(trial_rev))
            if _messages_tokens(trial) <= budget or not kept_rev:
                kept_rev = trial_rev
            else:
                break
        return system + list(reversed(kept_rev))

    def _select_tools_for_prompt_budget(
        self,
        tools: list[ToolDefinition],
        *,
        query_text: str,
        max_tools: int,
    ) -> list[ToolDefinition]:
        if max_tools <= 0:
            return []
        if len(tools) <= max_tools:
            return tools

        query_tokens = self._tokenize(query_text)
        baseline_tools = {
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "search_files",
            "grep",
            "find",
            "ls",
            "run_bash",
        }
        scored: list[tuple[float, int, ToolDefinition]] = []
        for idx, tool in enumerate(tools):
            corpus = f"{tool.name} {tool.description}"
            overlap = len(self._tokenize(corpus) & query_tokens)
            score = float(overlap)
            if tool.name in baseline_tools:
                score += 0.25
            scored.append((score, idx, tool))
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = [row[2] for row in scored[:max_tools]]
        return selected

    def _prepare_request_payload(self) -> tuple[list[Message], list[ToolDefinition]]:
        messages = self._context.get_messages()
        tools = self._registry.all_definitions()
        if not self._config.prompt_budget_enabled:
            return messages, tools

        max_input = max(512, int(self._config.prompt_budget_max_input_tokens))
        reserve = max(0, int(self._config.prompt_budget_output_reserve))
        input_budget = max(256, max_input - reserve)
        budgeted_messages = self._trim_messages_for_prompt_budget(messages, input_budget)

        latest_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user = msg.content or ""
                break
        max_tools = max(1, int(self._config.prompt_budget_max_tools_total))
        budgeted_tools = self._select_tools_for_prompt_budget(
            tools,
            query_text=latest_user,
            max_tools=max_tools,
        )

        return budgeted_messages, budgeted_tools

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

            # Some providers occasionally end a turn with no text/tool calls.
            # Retry with a short nudge, but always under a hard cap.
            empty_retry_attempts = 0
            while self._is_empty_end_turn_response(response, had_deltas):
                if empty_retry_attempts >= _EMPTY_RESPONSE_MAX_RETRIES:
                    yield ErrorEvent(
                        message=(
                            "Provider returned empty end_turn response repeatedly; "
                            "aborting empty-response retries."
                        )
                    )
                    return

                empty_retry_attempts += 1
                logger.warning(
                    "Provider returned empty end_turn response; retrying with nudge "
                    "(%d/%d)",
                    empty_retry_attempts,
                    _EMPTY_RESPONSE_MAX_RETRIES,
                )
                raw_retry, retry_error = self._call_with_empty_response_nudge()
                if retry_error is not None:
                    yield retry_error
                    return
                if raw_retry is None:
                    yield ErrorEvent(message="Provider returned no response after empty-response retry.")
                    return

                response, had_deltas, steer_msg = yield from self._stream(raw_retry)

                if steer_msg is not None:
                    discarded = len(had_deltas) if isinstance(had_deltas, list) else 0
                    yield SteerEvent(new_input=steer_msg, discarded_tokens=discarded)
                    self._context.add_message(Message(role="user", content=steer_msg))
                    self._context.compactor.reset_overflow_flag()
                    turns = 0
                    continue

                if response is None:
                    yield ErrorEvent(message="Provider returned no response after empty-response retry.")
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

            # Budget guard: check if session cost exceeds --max-cost
            if self._config.max_cost > 0 and cu is not None and self._cost_calculator:
                class _U:
                    pass
                _u = _U()
                _u.input_tokens = cu["input_tokens"]        # type: ignore[attr-defined]
                _u.output_tokens = cu["output_tokens"]      # type: ignore[attr-defined]
                _u.cache_read_tokens = cu["cache_read_tokens"]  # type: ignore[attr-defined]
                _u.cache_write_tokens = cu["cache_write_tokens"]  # type: ignore[attr-defined]
                session_cost = self._cost_calculator(self._config.model, _u)
                if session_cost >= self._config.max_cost:
                    yield TurnComplete(usage=response.usage)
                    yield CostLimitExceeded(session_cost=session_cost, max_cost=self._config.max_cost)
                    self._persist()
                    return

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
            for idx, call in enumerate(calls):
                yield ToolCallEvent(call=call)

                # Phase A policy scaffold: allow future policy engines to block tool calls.
                if self._policy_hook is not None:
                    decision = self._policy_hook.before_tool_call(agent=self, call=call)
                    if decision.requires_approval:
                        reason = decision.reason or f"Approval required by policy ({decision.risk}): {call.name}"
                        if self._policy_approval_hook is None:
                            # Non-interactive or legacy contexts without approval hook:
                            # allow execution to preserve backward compatibility.
                            pass
                        else:
                            yield PolicyDecisionEvent(
                                action=call.name,
                                decision="confirm",
                                risk=decision.risk,
                                reason=reason,
                            )
                            append_assistant_event(
                                self._config.workspace_root,
                                make_assistant_event(
                                    family="policy",
                                    name="approval_requested",
                                    payload={
                                        "tool": call.name,
                                        "tool_call_id": call.id,
                                        "risk": decision.risk,
                                        "reason": reason,
                                    },
                                    session_id=getattr(self._session, "id", ""),
                                    severity="warning",
                                ),
                            )
                            append_audit_record(
                                self._config.workspace_root,
                                "policy.approval_requested",
                                {
                                    "tool": call.name,
                                    "tool_call_id": call.id,
                                    "risk": decision.risk,
                                    "reason": reason,
                                },
                            )

                            approved = bool(self._policy_approval_hook(reason))
                            if approved:
                                yield PolicyDecisionEvent(
                                    action=call.name,
                                    decision="approved",
                                    risk=decision.risk,
                                    reason=reason,
                                )
                                append_assistant_event(
                                    self._config.workspace_root,
                                    make_assistant_event(
                                        family="policy",
                                        name="approved",
                                        payload={
                                            "tool": call.name,
                                            "tool_call_id": call.id,
                                            "risk": decision.risk,
                                        },
                                        session_id=getattr(self._session, "id", ""),
                                        severity="info",
                                    ),
                                )
                                append_audit_record(
                                    self._config.workspace_root,
                                    "policy.approved",
                                    {
                                        "tool": call.name,
                                        "tool_call_id": call.id,
                                        "risk": decision.risk,
                                    },
                                )
                            else:
                                deny_reason = f"Approval denied: {reason}"
                                yield PolicyDecisionEvent(
                                    action=call.name,
                                    decision="denied",
                                    risk=decision.risk,
                                    reason=deny_reason,
                                )
                                append_assistant_event(
                                    self._config.workspace_root,
                                    make_assistant_event(
                                        family="policy",
                                        name="denied",
                                        payload={
                                            "tool": call.name,
                                            "tool_call_id": call.id,
                                            "risk": decision.risk,
                                            "reason": deny_reason,
                                        },
                                        session_id=getattr(self._session, "id", ""),
                                        severity="warning",
                                    ),
                                )
                                append_audit_record(
                                    self._config.workspace_root,
                                    "policy.denied",
                                    {
                                        "tool": call.name,
                                        "tool_call_id": call.id,
                                        "risk": decision.risk,
                                        "reason": deny_reason,
                                    },
                                )
                                blocked[idx] = ToolResult(
                                    tool_call_id=call.id,
                                    content=deny_reason,
                                    is_error=True,
                                )
                                continue

                    if not decision.allow:
                        reason = decision.reason or f"Blocked by policy ({decision.risk}): {call.name}"
                        yield PolicyDecisionEvent(
                            action=call.name,
                            decision="block",
                            risk=decision.risk,
                            reason=reason,
                        )
                        append_assistant_event(
                            self._config.workspace_root,
                            make_assistant_event(
                                family="policy",
                                name="blocked",
                                payload={
                                    "tool": call.name,
                                    "tool_call_id": call.id,
                                    "risk": decision.risk,
                                    "reason": reason,
                                },
                                session_id=getattr(self._session, "id", ""),
                                severity="warning",
                            ),
                        )
                        append_audit_record(
                            self._config.workspace_root,
                            "policy.blocked",
                            {
                                "tool": call.name,
                                "tool_call_id": call.id,
                                "risk": decision.risk,
                                "reason": reason,
                            },
                        )
                        blocked[idx] = ToolResult(
                            tool_call_id=call.id,
                            content=reason,
                            is_error=True,
                        )
                        continue

                if self._ext_registry:
                    ctx = BeforeToolCallContext(tool_call=call, agent=self)
                    res = self._ext_registry.fire_before_tool_call(ctx)
                    if res and res.block:
                        err_msg = res.reason or f"Blocked by extension: {call.name}"
                        blocked[idx] = ToolResult(
                            tool_call_id=call.id, content=err_msg, is_error=True
                        )

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

            # Fast-path: when any `agent` tool call in this batch spawns a
            # background task, tool outputs are already user-ready. Skip the
            # extra LLM round-trip that usually adds redundant narration.
            spawned_background = False
            for idx, call in enumerate(calls):
                result = blocked.get(idx) or dispatched.get(idx)
                if result is None or result.is_error:
                    continue
                if (
                    call.name == "agent"
                    and isinstance(result.content, str)
                    and result.content.startswith("Agent spawned in background.")
                ):
                    spawned_background = True
                    break

            if spawned_background:
                yield TurnComplete(usage=response.usage)
                yield from self._maybe_compact()
                self._persist()
                return

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
                messages, tools = self._prepare_request_payload()
                _trace.log_request(messages, tools)
                raw = self._provider.chat(
                    messages=messages,
                    tools=tools,
                )
                return raw, None

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                _trace.log_error(last_error)
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
            _trace.log_response(raw)
            return raw, False, None

        had_deltas = False
        response: ProviderResponse | None = None
        delta_count = 0
        thinking_parts: list[str] = []

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
                    if thinking_parts:
                        _trace.log_thinking("".join(thinking_parts))
                    return None, had_deltas, steer_msg

                if isinstance(item, TextDelta):
                    had_deltas = True
                    delta_count += 1
                    if item.is_thinking:
                        thinking_parts.append(item.text)
                    yield item
                elif isinstance(item, ProviderResponse):
                    response = item
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error consuming provider stream")
            _trace.log_error(str(exc))
            if thinking_parts:
                _trace.log_thinking("".join(thinking_parts))
            yield ErrorEvent(message=f"Stream error: {exc}")
            return None, had_deltas, None

        if thinking_parts:
            _trace.log_thinking("".join(thinking_parts))
        if response is not None:
            _trace.log_response(response)
        return response, had_deltas, None

    @staticmethod
    def _is_empty_end_turn_response(response: ProviderResponse, had_deltas: bool) -> bool:
        return (
            response.stop_reason == "end_turn"
            and not response.tool_calls
            and not (response.content or "").strip()
            and not had_deltas
        )

    def _call_with_empty_response_nudge(self) -> tuple[object | None, ErrorEvent | None]:
        try:
            messages, tools = self._prepare_request_payload()
            retry_messages = list(messages) + [Message(role="user", content=_EMPTY_RESPONSE_NUDGE)]
            _trace.log_request(retry_messages, tools)
            raw = self._provider.chat(messages=retry_messages, tools=tools)
            return raw, None
        except Exception as exc:  # noqa: BLE001
            _trace.log_error(str(exc))
            logger.warning("Empty-response retry failed: %s", exc)
            return None, ErrorEvent(message=f"Provider empty-response retry failed: {exc}")

    def _persist(self) -> None:
        try:
            self._session_manager.save(
                self._session,
                messages=self._context.snapshot(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist session: %s", exc)
