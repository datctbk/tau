"""Base provider protocol."""
from __future__ import annotations

from collections.abc import Generator
from typing import Protocol, runtime_checkable

from tau.core.types import Message, ProviderResponse, ToolDefinition


@runtime_checkable
class BaseProvider(Protocol):
    @property
    def name(self) -> str: ...

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse | Generator: ...
