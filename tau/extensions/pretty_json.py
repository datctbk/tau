"""pretty_json — built-in reference extension.

Adds a `pretty_print_json` tool, a /json slash command, and an event hook
that logs TurnComplete token usage to a module-level counter (useful as a
hook example).
"""
from __future__ import annotations

import json

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import (
    ExtensionManifest,
    SlashCommand,
    ToolDefinition,
    ToolParameter,
    TurnComplete,
)


def _pretty_json(json_string: str, indent: int = 2) -> str:
    try:
        parsed = json.loads(json_string)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON — {exc}"


class PrettyJsonExtension(Extension):
    manifest = ExtensionManifest(
        name="pretty_json",
        version="0.1.0",
        description="Pretty-prints JSON. Adds a /json slash command and a token-usage hook.",
        author="tau",
    )

    def __init__(self) -> None:
        self.total_tokens: int = 0   # accumulated across all turns (hook demo)

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="pretty_print_json",
                description="Pretty-print a JSON string with configurable indentation.",
                parameters={
                    "json_string": ToolParameter(
                        type="string",
                        description="The raw JSON string to format.",
                    ),
                    "indent": ToolParameter(
                        type="integer",
                        description="Number of spaces for indentation (default 2).",
                        required=False,
                    ),
                },
                handler=_pretty_json,
            )
        ]

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="json",
                description="Pretty-print inline JSON.",
                usage='/json {"key": "value"}',
            )
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        if command != "json":
            return False
        if not args:
            context.print("[dim]  Usage: /json <json_string>[/dim]")
            return True
        result = _pretty_json(args)
        context.print(f"[green]{result}[/green]")
        return True

    def event_hook(self, event) -> None:
        if isinstance(event, TurnComplete):
            self.total_tokens += event.usage.total


EXTENSION = PrettyJsonExtension()
