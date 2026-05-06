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
    def __init__(self, max_result_chars: int = 0) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._owners: dict[str, str] = {}
        self._max_result_chars = max_result_chars  # 0 = unlimited

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition, owner: str | None = None) -> None:
        owner_name = owner or "core"
        if tool.name in self._tools:
            previous_owner = self._owners.get(tool.name, "unknown")
            logger.warning(
                "Tool %r already registered by %r — overwriting with %r.",
                tool.name,
                previous_owner,
                owner_name,
            )
        self._tools[tool.name] = tool
        self._owners[tool.name] = owner_name
        logger.debug("Registered tool: %s", tool.name)

    def register_many(self, tools: list[ToolDefinition], owner: str | None = None) -> None:
        for tool in tools:
            self.register(tool, owner=owner)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not registered)."""
        self._tools.pop(name, None)
        self._owners.pop(name, None)

    def keep_only(self, names: list[str]) -> None:
        """Keep only the named tools, remove everything else."""
        self._tools = {k: v for k, v in self._tools.items() if k in names}
        self._owners = {k: v for k, v in self._owners.items() if k in names}

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

    def owner_of(self, name: str) -> str | None:
        return self._owners.get(name)

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
            if self._max_result_chars > 0 and len(content) > self._max_result_chars:
                content = content[:self._max_result_chars] + (
                    f"\n\n... (truncated — {len(content):,} chars total, "
                    f"showing first {self._max_result_chars:,})"
                )
            return ToolResult(tool_call_id=call.id, content=content, is_error=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %r raised an error.", call.name)
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error in tool {call.name!r}: {exc}",
                is_error=True,
            )
