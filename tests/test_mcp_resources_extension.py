from __future__ import annotations

import json
from pathlib import Path

from tau.core.context import ContextManager
from tau.core.extension import ExtensionContext
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig, ToolCall
from tau.extensions.mcp_resources import MCPResourcesExtension


def _cfg(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        provider="openai",
        model="gpt-4o",
        workspace_root=str(tmp_path),
        compaction_enabled=False,
        retry_enabled=False,
    )


def _ctx(tmp_path: Path):
    reg = ToolRegistry()
    ctx = ContextManager(_cfg(tmp_path))
    steering = SteeringChannel()
    prints: list[str] = []
    ext_ctx = ExtensionContext(
        registry=reg,
        context=ctx,
        steering=steering,
        console_print=prints.append,
        agent_config=_cfg(tmp_path),
    )
    return reg, ext_ctx, prints


def test_mcp_registers_tools_and_slash(tmp_path: Path):
    reg, ext_ctx, _ = _ctx(tmp_path)
    ext = MCPResourcesExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    assert "mcp_list_resources" in reg.names()
    assert "mcp_read_resource" in reg.names()
    assert "mcp_list_tools" in reg.names()
    assert "mcp_call_tool" in reg.names()
    slash = {c.name for c in ext.slash_commands()}
    assert "mcp-resources" in slash
    assert "mcp-tools" in slash


def test_mcp_list_and_read_from_catalog(tmp_path: Path):
    reg, ext_ctx, _ = _ctx(tmp_path)
    ext = MCPResourcesExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    (tmp_path / ".tau" / "mcp").mkdir(parents=True, exist_ok=True)
    note = tmp_path / "docs.txt"
    note.write_text("hello mcp", encoding="utf-8")
    catalog = [
        {
            "uri": "mcp://local/docs",
            "server": "local",
            "name": "Docs",
            "path": "docs.txt",
        }
    ]
    (tmp_path / ".tau" / "mcp" / "resources.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )

    listed = reg.dispatch(ToolCall(id="1", name="mcp_list_resources", arguments={}))
    assert "mcp://local/docs" in listed.content

    content = reg.dispatch(
        ToolCall(id="2", name="mcp_read_resource", arguments={"uri": "mcp://local/docs"})
    )
    assert "hello mcp" in content.content


def test_mcp_file_uri_is_workspace_scoped(tmp_path: Path):
    reg, ext_ctx, _ = _ctx(tmp_path)
    ext = MCPResourcesExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    p = tmp_path / "a.txt"
    p.write_text("abc", encoding="utf-8")
    ok = reg.dispatch(
        ToolCall(id="1", name="mcp_read_resource", arguments={"uri": f"file://{p}"})
    )
    assert ok.content == "abc"

    bad = reg.dispatch(
        ToolCall(id="2", name="mcp_read_resource", arguments={"uri": "file:///etc/passwd"})
    )
    assert "outside workspace" in bad.content


def test_mcp_list_tools_and_call_tool(tmp_path: Path):
    reg, ext_ctx, _ = _ctx(tmp_path)
    ext = MCPResourcesExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    (tmp_path / ".tau" / "mcp").mkdir(parents=True, exist_ok=True)
    tools_catalog = [
        {
            "server": "local",
            "name": "echo_text",
            "description": "Echo a provided text",
            "argv": ["echo", "{text}"],
            "cwd": ".",
            "timeout_sec": 5,
        }
    ]
    (tmp_path / ".tau" / "mcp" / "tools.json").write_text(
        json.dumps(tools_catalog), encoding="utf-8"
    )

    listed = reg.dispatch(ToolCall(id="1", name="mcp_list_tools", arguments={}))
    assert "echo_text" in listed.content

    called = reg.dispatch(
        ToolCall(
            id="2",
            name="mcp_call_tool",
            arguments={
                "server": "local",
                "tool": "echo_text",
                "arguments_json": json.dumps({"text": "hello"}),
            },
        )
    )
    assert "[exit 0]" in called.content
    assert "hello" in called.content


def test_mcp_call_tool_missing_argument(tmp_path: Path):
    reg, ext_ctx, _ = _ctx(tmp_path)
    ext = MCPResourcesExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    (tmp_path / ".tau" / "mcp").mkdir(parents=True, exist_ok=True)
    tools_catalog = [
        {"server": "local", "name": "echo_text", "argv": ["echo", "{text}"]}
    ]
    (tmp_path / ".tau" / "mcp" / "tools.json").write_text(
        json.dumps(tools_catalog), encoding="utf-8"
    )
    called = reg.dispatch(
        ToolCall(
            id="1",
            name="mcp_call_tool",
            arguments={"server": "local", "tool": "echo_text", "arguments_json": "{}"},
        )
    )
    assert "missing argument: text" in called.content
