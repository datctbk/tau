"""Agent loop — the heart of tau."""

from __future__ import annotations

import logging
from typing import Generator, Iterator, TYPE_CHECKING

from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    Event,
    Message,
    ProviderResponse,
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
    from tau.core.tool_registry import ToolRegistry
    from tau.providers.base import BaseProvider

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._config = config
        self._provider = provider
        self._registry = registry
        self._context = context
        self._session = session
        self._session_manager = session_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> Generator[Event, None, None]:
        """Process one user message; yield Events for the CLI to render."""
        self._context.add_message(Message(role="user", content=user_input))

        turns = 0
        while turns < self._config.max_turns:
            turns += 1
            logger.debug("Agent turn %d", turns)

            self._context.trim()

            try:
                raw = self._provider.chat(
                    messages=self._context.get_messages(),
                    tools=self._registry.all_definitions(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Provider error on turn %d", turns)
                yield ErrorEvent(message=f"Provider error: {exc}")
                return

            # Stream deltas live; collect the final ProviderResponse
            response, had_deltas = yield from self._stream(raw)
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

            # No tool calls → done
            if not response.tool_calls:
                yield TurnComplete(usage=response.usage)
                self._persist()
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
            # Always loop back — the model must see tool results before finishing
            continue

        yield ErrorEvent(message=f"Reached max_turns limit ({self._config.max_turns}).")
        self._persist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream(self, raw) -> Generator[Event, None, tuple[ProviderResponse | None, bool]]:
        """Yield TextDelta events live; return (ProviderResponse, had_deltas)."""
        if isinstance(raw, ProviderResponse):
            return raw, False

        had_deltas = False
        response: ProviderResponse | None = None
        try:
            for item in raw:
                if isinstance(item, TextDelta):
                    had_deltas = True
                    yield item          # ← emitted immediately to the CLI
                elif isinstance(item, ProviderResponse):
                    response = item
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error consuming provider stream")
            yield ErrorEvent(message=f"Stream error: {exc}")
            return None, had_deltas

        return response, had_deltas

    def _persist(self) -> None:
        try:
            self._session_manager.save(
                self._session,
                messages=self._context.snapshot(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist session: %s", exc)
