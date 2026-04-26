"""Prompt-layer contribution contracts.

Core context keeps prompt assembly minimal. Optional layers (workspace context,
extension guidance, etc.) can be injected through this narrow interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tau.core.context import ContextManager


@dataclass(frozen=True)
class PromptLayer:
    name: str
    content: str
    priority: int = 50


class PromptLayerContributor(Protocol):
    def prompt_layers(self) -> list[PromptLayer]:
        """Return prompt layers to inject into the system prompt."""


def apply_prompt_layers(context: "ContextManager", layers: list[PromptLayer]) -> None:
    """Inject prompt layers in descending priority order."""
    for layer in sorted(layers, key=lambda x: x.priority, reverse=True):
        if layer.content.strip():
            context.inject_prompt_fragment(layer.content, name=layer.name, priority=layer.priority)
