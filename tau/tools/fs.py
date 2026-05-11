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


def _get_changed_rel_paths(workspace_root: Path) -> set[str]:
    try:
        from tau.core.code_index import detect_workspace_changes

        changes, _new_manifest, _old_manifest = detect_workspace_changes(workspace_root)
        return set(changes.added + changes.modified)
    except Exception:
        return set()


def _filter_targets_changed_only(root: Path, targets: list[Path]) -> list[Path]:
    changed = _get_changed_rel_paths(Path(_workspace_root).resolve())
    if not changed:
        return targets
    ws_root = Path(_workspace_root).resolve()
    filtered: list[Path] = []
    for t in targets:
        try:
            rel = t.resolve().relative_to(ws_root).as_posix()
        except Exception:
            continue
        if rel in changed:
            filtered.append(t)
    return filtered


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


def search_files(
    pattern: str,
    path: str = ".",
    use_regex: bool = False,
    changed_only: bool = False,
) -> str:
    root = _resolve(path)
    if use_regex:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Invalid regex: {exc}"
    matches: list[str] = []
    targets = [f for f in root.rglob("*") if f.is_file()] if root.is_dir() else ([root] if root.is_file() else [])
    if changed_only:
        targets = _filter_targets_changed_only(root, targets)
    for fpath in targets:
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                hit = bool(compiled.search(line)) if use_regex else (pattern in line)
                if hit:
                    matches.append(f"{fpath.relative_to(root)}:{i}: {line.strip()}")
    return "\n".join(matches[:200]) if matches else "No matches found."


def grep(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    case_insensitive: bool = False,
    include: str = "",
    max_results: int = 200,
    changed_only: bool = False,
) -> str:
    """Search for a regex pattern across file contents."""
    root = _resolve(path)
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        return f"Invalid regex: {exc}"

    matches: list[str] = []
    try:
        include_pat = re.compile(include) if include else None
    except re.error as exc:
        return f"Invalid include regex: {exc}"

    targets: list[Path] = []
    if root.is_file():
        targets = [root]
    elif recursive:
        targets = [f for f in root.rglob("*") if f.is_file()]
    else:
        targets = [f for f in root.iterdir() if f.is_file()]
    if changed_only:
        targets = _filter_targets_changed_only(root, targets)

    for fpath in sorted(targets):
        if include_pat and not include_pat.search(fpath.name):
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                try:
                    rel = fpath.relative_to(Path(_workspace_root).resolve())
                except ValueError:
                    rel = fpath
                matches.append(f"{rel}:{i}: {line.rstrip()}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n(truncated at {max_results} results)"
    return "\n".join(matches) if matches else "No matches found."


def find(
    path: str = ".",
    name: str = "",
    type: str = "",
    max_depth: int = -1,
    max_results: int = 200,
    changed_only: bool = False,
) -> str:
    """Find files or directories by name pattern and/or type."""
    root = _resolve(path)
    try:
        name_pat = re.compile(name) if name else None
    except re.error as exc:
        return f"Invalid name regex: {exc}"
    results: list[str] = []

    changed = _get_changed_rel_paths(Path(_workspace_root).resolve()) if changed_only else set()

    def _walk(cur: Path, depth: int) -> None:
        if max_depth >= 0 and depth > max_depth:
            return
        try:
            entries = sorted(cur.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if len(results) >= max_results:
                return
            # type filter
            if type == "f" and not entry.is_file():
                pass
            elif type == "d" and not entry.is_dir():
                pass
            else:
                match_name = (not name_pat) or bool(name_pat.search(entry.name))
                match_type = (
                    (not type)
                    or (type == "f" and entry.is_file())
                    or (type == "d" and entry.is_dir())
                )
                if match_name and match_type:
                    try:
                        rel = entry.relative_to(Path(_workspace_root).resolve())
                    except ValueError:
                        rel = entry
                    rel_str = str(rel)
                    if changed_only:
                        rel_posix = Path(rel_str).as_posix()
                        if entry.is_file():
                            if rel_posix in changed:
                                results.append(rel_str)
                        else:
                            # Include directory only if it contains any changed file.
                            prefix = rel_posix.rstrip("/") + "/"
                            if any(c.startswith(prefix) for c in changed):
                                results.append(rel_str)
                    else:
                        results.append(rel_str)
            if entry.is_dir():
                _walk(entry, depth + 1)

    _walk(root, 0)
    suffix = f"\n(truncated at {max_results} results)" if len(results) >= max_results else ""
    return ("\n".join(results) + suffix) if results else "No matches found."


def ls(path: str = ".", all: bool = False, long: bool = False) -> str:
    """List directory contents, optionally with hidden files and metadata."""
    import stat as _stat
    p = _resolve(path)
    if not p.is_dir():
        raise NotADirectoryError(f"{path!r} is not a directory.")
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    if not all:
        entries = [e for e in entries if not e.name.startswith(".")]
    if not entries:
        return "(empty)"
    if not long:
        lines = [f"{e.name}{'/' if e.is_dir() else ''}" for e in entries]
        return "\n".join(lines)
    # long format
    import datetime as _dt
    lines: list[str] = []
    for e in entries:
        try:
            st = e.stat()
            mode = _stat.filemode(st.st_mode)
            size = st.st_size
            mtime = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%b %d %H:%M")
        except OSError:
            mode, size, mtime = "?---------", 0, "?"
        name = f"{e.name}{'/' if e.is_dir() else ''}"
        lines.append(f"{mode}  {size:>10}  {mtime}  {name}")
    return "\n".join(lines)


def code_index_get_changed(path: str = ".", persist: bool = False, max_paths: int = 200) -> str:
    """Return changed files according to code index Merkle manifests."""
    root = _resolve(path)
    workspace_root = Path(_workspace_root).resolve()
    try:
        from tau.core.code_index import detect_workspace_changes, refresh_code_index

        if persist:
            stats = refresh_code_index(workspace_root)
            changes = stats["changes"]
            added = changes.added
            modified = changes.modified
            deleted = changes.deleted
            unchanged_count = changes.unchanged_count
            file_count = int(stats.get("file_count", 0))
            duration_ms = int(stats.get("duration_ms", 0))
        else:
            changes, new_manifest, _old_manifest = detect_workspace_changes(workspace_root)
            added = changes.added
            modified = changes.modified
            deleted = changes.deleted
            unchanged_count = changes.unchanged_count
            file_count = len(new_manifest.get("files", {}))
            duration_ms = 0
    except Exception as exc:
        return f"Error: code index unavailable: {exc}"

    # If path is a subdirectory, filter relative changed paths to that subtree.
    if root != workspace_root:
        try:
            sub = root.relative_to(workspace_root).as_posix().rstrip("/") + "/"
            added = [p for p in added if p.startswith(sub)]
            modified = [p for p in modified if p.startswith(sub)]
            deleted = [p for p in deleted if p.startswith(sub)]
        except Exception:
            pass

    lines = [
        f"added={len(added)} modified={len(modified)} deleted={len(deleted)} unchanged={unchanged_count} files={file_count}"
    ]
    if duration_ms:
        lines.append(f"duration_ms={duration_ms}")
    combined = [("A", p) for p in added] + [("M", p) for p in modified] + [("D", p) for p in deleted]
    for kind, rel in combined[: max(1, max_paths)]:
        lines.append(f"{kind} {rel}")
    if len(combined) > max_paths:
        lines.append(f"... and {len(combined) - max_paths} more")
    return "\n".join(lines)


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
            "changed_only": ToolParameter(type="boolean", description="If true, search only added/modified files from code index.", required=False),
        },
        handler=search_files,
    ),
    ToolDefinition(
        name="grep",
        description=(
            "Search file contents for a regex pattern. "
            "Returns matching lines with file path and line number. "
            "Supports recursive search, case-insensitive mode, and filename filtering."
        ),
        parameters={
            "pattern": ToolParameter(type="string", description="Regular expression to search for."),
            "path": ToolParameter(type="string", description="File or directory to search. Defaults to workspace root.", required=False),
            "recursive": ToolParameter(type="boolean", description="Search subdirectories recursively (default true).", required=False),
            "case_insensitive": ToolParameter(type="boolean", description="Case-insensitive matching (default false).", required=False),
            "include": ToolParameter(type="string", description="Regex filter on filename (e.g. '\\.py$'). Empty = all files.", required=False),
            "max_results": ToolParameter(type="integer", description="Maximum number of matching lines to return (default 200).", required=False),
            "changed_only": ToolParameter(type="boolean", description="If true, search only added/modified files from code index.", required=False),
        },
        handler=grep,
    ),
    ToolDefinition(
        name="find",
        description=(
            "Find files or directories matching a name pattern and/or type under a path."
        ),
        parameters={
            "path": ToolParameter(type="string", description="Directory to search from. Defaults to workspace root.", required=False),
            "name": ToolParameter(type="string", description="Regex pattern to match against entry names. Empty = all.", required=False),
            "type": ToolParameter(type="string", description="Filter by type: 'f' = files only, 'd' = directories only. Empty = both.", required=False),
            "max_depth": ToolParameter(type="integer", description="Maximum directory depth (-1 = unlimited).", required=False),
            "max_results": ToolParameter(type="integer", description="Maximum number of results (default 200).", required=False),
            "changed_only": ToolParameter(type="boolean", description="If true, return only paths affected in added/modified set.", required=False),
        },
        handler=find,
    ),
    ToolDefinition(
        name="code_index_get_changed",
        description=(
            "Get changed files from Merkle code index (added/modified/deleted). "
            "Use persist=true to refresh and save index stats first."
        ),
        parameters={
            "path": ToolParameter(type="string", description="Workspace subpath scope. Defaults to '.'.", required=False),
            "persist": ToolParameter(type="boolean", description="Refresh and persist manifest/stats before reporting.", required=False),
            "max_paths": ToolParameter(type="integer", description="Maximum changed paths to show.", required=False),
        },
        handler=code_index_get_changed,
    ),
    ToolDefinition(
        name="ls",
        description="List directory contents with optional hidden-file and long-format display.",
        parameters={
            "path": ToolParameter(type="string", description="Directory path (relative to workspace root). Defaults to '.'.", required=False),
            "all": ToolParameter(type="boolean", description="Include hidden files (names starting with '.'). Default false.", required=False),
            "long": ToolParameter(type="boolean", description="Long format: permissions, size, modification time. Default false.", required=False),
        },
        handler=ls,
    ),
]
