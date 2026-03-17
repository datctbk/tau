"""Tests for tau.core.tool_registry."""

import pytest
from tau.core.tool_registry import ToolRegistry, ToolNotFoundError
from tau.core.types import ToolCall, ToolDefinition, ToolParameter


def _make_tool(name: str, fn=None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        parameters={"x": ToolParameter(type="string", description="input")},
        handler=fn or (lambda x: f"ok:{x}"),
    )


def test_register_and_get():
    reg = ToolRegistry()
    tool = _make_tool("greet")
    reg.register(tool)
    assert reg.get("greet") is tool


def test_get_missing_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.get("nope")


def test_dispatch_success():
    reg = ToolRegistry()
    reg.register(_make_tool("echo", lambda x: f"echo:{x}"))
    result = reg.dispatch(ToolCall(id="1", name="echo", arguments={"x": "hello"}))
    assert result.content == "echo:hello"
    assert not result.is_error


def test_dispatch_unknown_tool_returns_error():
    reg = ToolRegistry()
    result = reg.dispatch(ToolCall(id="1", name="unknown", arguments={}))
    assert result.is_error


def test_dispatch_handler_exception_returns_error():
    def boom(x: str) -> str:
        raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(_make_tool("boom", boom))
    result = reg.dispatch(ToolCall(id="1", name="boom", arguments={"x": "y"}))
    assert result.is_error
    assert "kaboom" in result.content


def test_all_definitions():
    reg = ToolRegistry()
    reg.register(_make_tool("a"))
    reg.register(_make_tool("b"))
    assert {t.name for t in reg.all_definitions()} == {"a", "b"}
