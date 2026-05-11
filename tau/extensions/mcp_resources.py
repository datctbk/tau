"""mcp_resources — minimal MCP-style resource layer for tau.

Step-1 MCP support:
  - mcp_list_resources
  - mcp_read_resource
Step-2 MCP support:
  - mcp_list_tools
  - mcp_call_tool

Catalogs are declared in:
  - `.tau/mcp/resources.json`
  - `.tau/mcp/tools.json`
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand, ToolDefinition, ToolParameter


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


class MCPResourcesExtension(Extension):
    manifest = ExtensionManifest(
        name="mcp_resources",
        version="0.2.0",
        description="Minimal MCP resource and tool bridge from a local workspace catalog.",
        author="tau",
        system_prompt_fragment=(
            "When external project context is needed, use mcp_list_resources "
            "and mcp_read_resource before broad file scans. "
            "When an MCP tool exists for the task, prefer mcp_call_tool."
        ),
    )

    def __init__(self) -> None:
        self._workspace_root = Path(".").resolve()
        self._catalog_path = self._workspace_root / ".tau" / "mcp" / "resources.json"
        self._tools_catalog_path = self._workspace_root / ".tau" / "mcp" / "tools.json"

    def on_load(self, context: ExtensionContext) -> None:
        workspace = "."
        if hasattr(context, "_agent_config") and context._agent_config:
            workspace = getattr(context._agent_config, "workspace_root", ".") or "."
        self._workspace_root = Path(workspace).resolve()
        self._catalog_path = self._workspace_root / ".tau" / "mcp" / "resources.json"
        self._tools_catalog_path = self._workspace_root / ".tau" / "mcp" / "tools.json"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="mcp_list_resources",
                description="List MCP-style resources declared in workspace catalog.",
                parameters={
                    "server": ToolParameter(
                        type="string",
                        description="Optional server filter.",
                        required=False,
                    ),
                },
                handler=self._handle_list_resources,
            ),
            ToolDefinition(
                name="mcp_read_resource",
                description="Read MCP-style resource by URI (catalog URI or file:// URI).",
                parameters={
                    "uri": ToolParameter(
                        type="string",
                        description="Resource URI to read.",
                    ),
                },
                handler=self._handle_read_resource,
            ),
            ToolDefinition(
                name="mcp_list_tools",
                description="List MCP-style tools declared in workspace catalog.",
                parameters={
                    "server": ToolParameter(
                        type="string",
                        description="Optional server filter.",
                        required=False,
                    ),
                },
                handler=self._handle_list_tools,
            ),
            ToolDefinition(
                name="mcp_call_tool",
                description=(
                    "Invoke a cataloged MCP-style tool by server+name with JSON arguments. "
                    "Executes argv without shell."
                ),
                parameters={
                    "server": ToolParameter(type="string", description="Tool server id."),
                    "tool": ToolParameter(type="string", description="Tool name."),
                    "arguments_json": ToolParameter(
                        type="string",
                        description="Optional JSON object string for tool arguments.",
                        required=False,
                    ),
                },
                handler=self._handle_call_tool,
            ),
        ]

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="mcp-resources",
                description="Show resources from .tau/mcp/resources.json",
                usage="/mcp-resources [server]",
            ),
            SlashCommand(
                name="mcp-tools",
                description="Show tools from .tau/mcp/tools.json",
                usage="/mcp-tools [server]",
            ),
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        server = args.strip() or None
        if command == "mcp-resources":
            context.print(self._handle_list_resources(server=server))
            return True
        if command == "mcp-tools":
            context.print(self._handle_list_tools(server=server))
            return True
        return False

    def _load_catalog(self) -> list[dict[str, Any]]:
        if not self._catalog_path.is_file():
            return []
        try:
            data = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("uri"), str):
                out.append(item)
        return out

    def _resolve_catalog_path(self, item: dict[str, Any]) -> Path | None:
        raw = item.get("path")
        if not isinstance(raw, str) or not raw.strip():
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = self._workspace_root / p
        p = p.resolve()
        if not _is_within(self._workspace_root, p):
            return None
        return p

    def _load_tools_catalog(self) -> list[dict[str, Any]]:
        if not self._tools_catalog_path.is_file():
            return []
        try:
            data = json.loads(self._tools_catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("server"), str):
                continue
            if not isinstance(item.get("name"), str):
                continue
            argv = item.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                continue
            out.append(item)
        return out

    def _resolve_workdir(self, raw: str | None) -> Path | None:
        if not raw:
            return self._workspace_root
        p = Path(raw)
        if not p.is_absolute():
            p = self._workspace_root / p
        p = p.resolve()
        if not _is_within(self._workspace_root, p):
            return None
        return p

    def _handle_list_resources(self, server: str | None = None) -> str:
        items = self._load_catalog()
        if server:
            items = [it for it in items if it.get("server") == server]
        if not items:
            return (
                f"No MCP resources found. Add entries to {self._catalog_path} "
                "(JSON array with uri/server/path/name)."
            )
        lines = [f"MCP resources ({len(items)}):"]
        for it in items:
            uri = str(it.get("uri", ""))
            srv = str(it.get("server", "local"))
            name = str(it.get("name", uri))
            lines.append(f"- [{srv}] {name} — {uri}")
        return "\n".join(lines)

    def _handle_list_tools(self, server: str | None = None) -> str:
        items = self._load_tools_catalog()
        if server:
            items = [it for it in items if it.get("server") == server]
        if not items:
            return (
                f"No MCP tools found. Add entries to {self._tools_catalog_path} "
                "(JSON array with server/name/argv)."
            )
        lines = [f"MCP tools ({len(items)}):"]
        for it in items:
            srv = str(it.get("server", "local"))
            name = str(it.get("name", ""))
            desc = str(it.get("description", "")).strip()
            if desc:
                lines.append(f"- [{srv}] {name} — {desc}")
            else:
                lines.append(f"- [{srv}] {name}")
        return "\n".join(lines)

    def _read_text_file(self, path: Path, max_chars: int = 200_000) -> str:
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) <= max_chars:
            return data
        return data[:max_chars] + "\n\n...[truncated]..."

    def _handle_read_resource(self, uri: str) -> str:
        uri = (uri or "").strip()
        if not uri:
            return "Error: uri is required."

        # Direct file URI support for local workflows.
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            p = Path(parsed.path).resolve()
            if not _is_within(self._workspace_root, p):
                return "Error: file URI is outside workspace."
            if not p.is_file():
                return f"Error: file not found: {p}"
            return self._read_text_file(p)

        items = self._load_catalog()
        match = next((it for it in items if it.get("uri") == uri), None)
        if match is None:
            return f"Error: resource not found: {uri}"

        p = self._resolve_catalog_path(match)
        if p is None:
            return f"Error: resource path is invalid or outside workspace for uri: {uri}"
        if not p.is_file():
            return f"Error: resource file not found: {p}"
        return self._read_text_file(p)

    def _handle_call_tool(self, server: str, tool: str, arguments_json: str = "") -> str:
        server = (server or "").strip()
        tool = (tool or "").strip()
        if not server or not tool:
            return "Error: server and tool are required."

        args_map: dict[str, Any] = {}
        if arguments_json.strip():
            try:
                parsed = json.loads(arguments_json)
            except Exception as exc:
                return f"Error: invalid arguments_json: {exc}"
            if not isinstance(parsed, dict):
                return "Error: arguments_json must decode to an object."
            args_map = parsed

        items = self._load_tools_catalog()
        match = next((it for it in items if it.get("server") == server and it.get("name") == tool), None)
        if match is None:
            return f"Error: MCP tool not found: [{server}] {tool}"

        argv_raw = list(match.get("argv", []))
        argv: list[str] = []
        for tok in argv_raw:
            if tok.startswith("{") and tok.endswith("}") and len(tok) > 2:
                key = tok[1:-1]
                if key not in args_map:
                    return f"Error: missing argument: {key}"
                argv.append(str(args_map[key]))
            else:
                argv.append(tok)

        workdir = self._resolve_workdir(match.get("cwd"))
        if workdir is None:
            return "Error: tool cwd is outside workspace."

        timeout = int(match.get("timeout_sec", 30))
        if timeout < 1 or timeout > 600:
            timeout = 30

        try:
            res = subprocess.run(
                argv,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return f"Error: executable not found: {argv[0]}"
        except subprocess.TimeoutExpired:
            return f"Error: tool timed out after {timeout}s"
        except Exception as exc:
            return f"Error: tool execution failed: {exc}"

        out = (res.stdout or "").rstrip()
        err = (res.stderr or "").rstrip()
        parts = [f"[exit {res.returncode}]"]
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        return "\n".join(parts)


EXTENSION = MCPResourcesExtension()
