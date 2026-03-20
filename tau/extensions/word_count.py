"""word_count — built-in reference extension.

Adds a `word_count` tool and a /wc slash command.
"""
from __future__ import annotations

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand, ToolDefinition, ToolParameter


def _word_count(text: str) -> str:
    words = len(text.split())
    lines = len(text.splitlines())
    chars = len(text)
    return f"words={words}, lines={lines}, chars={chars}"


class WordCountExtension(Extension):
    manifest = ExtensionManifest(
        name="word_count",
        version="0.1.0",
        description="Counts words/lines/chars in a string. Adds a /wc slash command.",
        author="tau",
    )

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="word_count",
                description="Count the number of words, lines, and characters in a text string.",
                parameters={
                    "text": ToolParameter(
                        type="string",
                        description="The text to count.",
                    ),
                },
                handler=_word_count,
            )
        ]

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="wc",
                description="Count words/lines/chars of inline text.",
                usage="/wc <text>",
            )
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        if command != "wc":
            return False
        if not args:
            context.print("[dim]  Usage: /wc <text>[/dim]")
            return True
        result = _word_count(args)
        context.print(f"[cyan]  {result}[/cyan]")
        return True


EXTENSION = WordCountExtension()
