"""Trace logger — logs full LLM requests and responses to a file for debugging."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau.core.types import Message, ProviderResponse, ToolDefinition

logger = logging.getLogger(__name__)

_trace_path: Path | None = None
_turn_counter: int = 0


def configure_trace(path: str | None) -> None:
    """Enable trace logging to the given file path. None disables it."""
    global _trace_path, _turn_counter
    if path is None:
        _trace_path = None
        return
    _trace_path = Path(path).resolve()
    _trace_path.parent.mkdir(parents=True, exist_ok=True)
    _turn_counter = 0
    # Write header
    with _trace_path.open("w", encoding="utf-8") as f:
        f.write(f"# tau trace log — {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# {'=' * 70}\n\n")


def is_enabled() -> bool:
    return _trace_path is not None


def _write(text: str) -> None:
    if _trace_path is None:
        return
    with _trace_path.open("a", encoding="utf-8") as f:
        f.write(text)


def _fmt_message(m: Message) -> str:
    """Format a single message for the trace log."""
    parts: list[str] = []
    role_tag = m.role.upper()
    parts.append(f"  [{role_tag}]")
    if m.name:
        parts[0] += f" (name={m.name})"
    if m.tool_call_id:
        parts[0] += f" (tool_call_id={m.tool_call_id})"

    content = m.content or ""
    if len(content) > 2000:
        content = content[:2000] + f"\n  ... ({len(content)} chars total, truncated)"
    if content:
        for line in content.splitlines():
            parts.append(f"    {line}")

    if m.tool_calls:
        for tc in m.tool_calls:
            args_json = json.dumps(tc.arguments, indent=2, ensure_ascii=False)
            parts.append(f"    [TOOL_CALL] {tc.name} (id={tc.id})")
            for aline in args_json.splitlines():
                parts.append(f"      {aline}")

    if m.images:
        parts.append(f"    [IMAGES] {', '.join(m.images)}")

    return "\n".join(parts)


def _fmt_tools(tools: list[ToolDefinition]) -> str:
    """Format tool definitions as a compact list."""
    if not tools:
        return "  (none)"
    lines: list[str] = []
    for t in tools:
        params = ", ".join(
            f"{k}: {p.type}" + (" (optional)" if not p.required else "")
            for k, p in t.parameters.items()
        )
        lines.append(f"  • {t.name}({params})")
    return "\n".join(lines)


def log_request(messages: list[Message], tools: list[ToolDefinition]) -> None:
    """Log the full request being sent to the LLM."""
    if _trace_path is None:
        return
    global _turn_counter
    _turn_counter += 1

    lines: list[str] = []
    lines.append(f"{'─' * 72}")
    lines.append(f"▶ REQUEST  turn={_turn_counter}  time={datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    lines.append(f"{'─' * 72}")
    lines.append("")
    lines.append(f"Messages ({len(messages)}):")
    for i, m in enumerate(messages):
        lines.append(f"  --- message {i} ---")
        lines.append(_fmt_message(m))
    lines.append("")
    lines.append(f"Tools ({len(tools)}):")
    lines.append(_fmt_tools(tools))
    lines.append("")
    _write("\n".join(lines) + "\n")


def log_response(response: ProviderResponse) -> None:
    """Log the full response received from the LLM."""
    if _trace_path is None:
        return

    lines: list[str] = []
    lines.append(f"{'─' * 72}")
    lines.append(f"◀ RESPONSE  turn={_turn_counter}  time={datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    lines.append(f"{'─' * 72}")
    lines.append("")
    lines.append(f"  stop_reason: {response.stop_reason}")
    lines.append(f"  usage: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    if response.content:
        content = response.content
        if len(content) > 3000:
            content = content[:3000] + f"\n  ... ({len(response.content)} chars total, truncated)"
        lines.append("")
        lines.append("  Content:")
        for line in content.splitlines():
            lines.append(f"    {line}")

    if response.tool_calls:
        lines.append("")
        lines.append(f"  Tool calls ({len(response.tool_calls)}):")
        for tc in response.tool_calls:
            args_json = json.dumps(tc.arguments, indent=2, ensure_ascii=False)
            lines.append(f"    [{tc.name}] (id={tc.id})")
            for aline in args_json.splitlines():
                lines.append(f"      {aline}")

    lines.append("")
    lines.append("")
    _write("\n".join(lines) + "\n")


def log_error(error: str) -> None:
    """Log a provider error."""
    if _trace_path is None:
        return
    lines: list[str] = []
    lines.append(f"{'─' * 72}")
    lines.append(f"✗ ERROR  turn={_turn_counter}  time={datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    lines.append(f"{'─' * 72}")
    lines.append(f"  {error}")
    lines.append("")
    lines.append("")
    _write("\n".join(lines) + "\n")
