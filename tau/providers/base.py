"""Base provider protocol."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from tau.core.types import Message, ProviderResponse, ToolDefinition


@runtime_checkable
class BaseProvider(Protocol):
    @property
    def name(self) -> str: ...

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = False,
    ) -> ProviderResponse: ...
