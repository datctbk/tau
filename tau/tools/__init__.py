"""Built-in tools for tau."""

from tau.core.tool_registry import ToolRegistry
from tau.tools.fs import FS_TOOLS
from tau.tools.shell import SHELL_TOOLS


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register_many(FS_TOOLS)
    registry.register_many(SHELL_TOOLS)
