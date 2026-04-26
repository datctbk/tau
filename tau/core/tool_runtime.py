"""Tool runtime contract used by the core agent loop."""

from __future__ import annotations

from typing import Protocol

from tau.core.types import ToolCall, ToolDefinition, ToolResult


class ToolRuntime(Protocol):
    def all_definitions(self) -> list[ToolDefinition]: ...
    def dispatch(self, call: ToolCall) -> ToolResult: ...

