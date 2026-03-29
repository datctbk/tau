"""Filesystem tools — read, write, edit, list, search."""

from __future__ import annotations

import os
import re
from pathlib import Path

from tau.core.types import ToolDefinition, ToolParameter

# ---------------------------------------------------------------------------
# Module-level workspace root (set by CLI at startup via configure_fs)
# ---------------------------------------------------------------------------

_workspace_root: str = "."


def configure_fs(workspace_root: str) -> None:
    global _workspace_root
    _workspace_root = workspace_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str, workspace_root: str | None = None) -> Path:
    root = workspace_root if workspace_root is not None else _workspace_root
    p = Path(path)
    if not p.is_absolute():
        p = Path(root).resolve() / p
    resolved = p.resolve()
    root_resolved = Path(root).resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise PermissionError(f"Path {path!r} is outside the workspace root.")
    return resolved


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def read_file(path: str, start_line: int = 0, end_line: int = -1) -> str:
    p = _resolve(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    chunk = lines[start_line:] if end_line == -1 else lines[start_line:end_line + 1]
    numbered = [f"{i:4d} | {line}" for i, line in enumerate(chunk, start=start_line + 1)]
    return "".join(numbered) if numbered else "(empty)"


def write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path!r} does not exist. Use write_file to create it first."
        )
    original = p.read_text(encoding="utf-8")
    if old_str not in original:
        raise ValueError(
            f"old_str not found in {path!r}. Make sure it matches exactly (including whitespace)."
        )
    count = original.count(old_str)
    if count > 1:
        raise ValueError(f"old_str appears {count} times in {path!r} — be more specific.")
    p.write_text(original.replace(old_str, new_str, 1), encoding="utf-8")
    return f"Edited {path}: replaced 1 occurrence."


def list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.is_dir():
        raise NotADirectoryError(f"{path!r} is not a directory.")
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    lines = [f"{e.name}{'/' if e.is_dir() else ''}" for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"


def search_files(pattern: str, path: str = ".", use_regex: bool = False) -> str:
    root = _resolve(path)
    matches: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                hit = bool(re.search(pattern, line)) if use_regex else (pattern in line)
                if hit:
                    matches.append(f"{fpath.relative_to(root)}:{i}: {line.strip()}")
    return "\n".join(matches[:200]) if matches else "No matches found."


# ---------------------------------------------------------------------------
# ToolDefinition list
# ---------------------------------------------------------------------------

FS_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read the contents of a file. Optionally slice by line range (0-based). "
            "Returns lines prefixed with line numbers."
        ),
        parameters={
            "path": ToolParameter(type="string", description="Path to the file (relative to workspace root)."),
            "start_line": ToolParameter(type="integer", description="First line to read (0-based, inclusive).", required=False),
            "end_line": ToolParameter(type="integer", description="Last line to read (0-based, inclusive). -1 = end of file.", required=False),
        },
        handler=read_file,
    ),
    ToolDefinition(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={
            "path": ToolParameter(type="string", description="Destination path (relative to workspace root)."),
            "content": ToolParameter(type="string", description="Full file content to write."),
        },
        handler=write_file,
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Replace an exact string in a file with a new string. "
            "old_str must appear exactly once in the file."
        ),
        parameters={
            "path": ToolParameter(type="string", description="Path to the file."),
            "old_str": ToolParameter(type="string", description="Exact string to find and replace."),
            "new_str": ToolParameter(type="string", description="Replacement string."),
        },
        handler=edit_file,
    ),
    ToolDefinition(
        name="list_dir",
        description="List the contents of a directory.",
        parameters={
            "path": ToolParameter(type="string", description="Directory path (relative to workspace root). Defaults to '.'.", required=False),
        },
        handler=list_dir,
    ),
    ToolDefinition(
        name="search_files",
        description="Search for a string or regex pattern across files in a directory.",
        parameters={
            "pattern": ToolParameter(type="string", description="The string or regex pattern to search for."),
            "path": ToolParameter(type="string", description="Root directory to search in. Defaults to '.'.", required=False),
            "use_regex": ToolParameter(type="boolean", description="If true, treat pattern as a regex.", required=False),
        },
        handler=search_files,
    ),
]
