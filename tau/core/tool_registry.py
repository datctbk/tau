"""Tool registry and dispatcher."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from tau.core.types import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            logger.warning("Tool %r already registered — overwriting.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_many(self, tools: list[ToolDefinition]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not registered)."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"No tool named {name!r} is registered.")

    def all_definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=str(exc),
                is_error=True,
            )

        logger.debug("Dispatching tool %r with args %s", call.name, call.arguments)
        try:
            sig = inspect.signature(tool.handler)
            params = sig.parameters
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_var_keyword:
                filtered = call.arguments
            else:
                filtered = {k: v for k, v in call.arguments.items() if k in params}
                dropped = set(call.arguments) - set(filtered)
                if dropped:
                    logger.warning(
                        "Tool %r: ignoring unexpected argument(s): %s", call.name, dropped
                    )
            raw: Any = tool.handler(**filtered)
            content = raw if isinstance(raw, str) else str(raw)
            return ToolResult(tool_call_id=call.id, content=content, is_error=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %r raised an error.", call.name)
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error in tool {call.name!r}: {exc}",
                is_error=True,
            )
