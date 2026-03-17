"""Example skill tools."""

from tau.core.types import ToolDefinition, ToolParameter


def say_hello(name: str) -> str:
    return f"Hello, {name}! This is an example skill tool."


TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="say_hello",
        description="A demo tool that greets a person by name.",
        parameters={
            "name": ToolParameter(type="string", description="The name to greet."),
        },
        handler=say_hello,
    ),
]
