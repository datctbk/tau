"""Incremental code index primitives (Merkle manifest + changed-file detector).

This module is intentionally dependency-light and reusable by core or extensions.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".tau",
}

MANIFEST_VERSION = 1
INDEX_STATS_NAME = "stats.json"


@dataclass
class ChangedFiles:
    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged_count: int

    @property
    def changed(self) -> list[str]:
        return sorted(self.added + self.modified)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_gitignore(workspace_root: Path) -> list[str]:
    p = workspace_root / ".gitignore"
    if not p.is_file():
        return []
    patterns: list[str] = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line)
    return patterns


def _matches_ignore(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.endswith("/"):
            if rel_posix == pat[:-1] or rel_posix.startswith(pat):
                return True
        # plain segment pattern (e.g. *.log, temp/*)
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        # support basename matching for simple patterns
        if "/" not in pat and fnmatch.fnmatch(Path(rel_posix).name, pat):
            return True
    return False


def scan_workspace_files(
    workspace_root: str | Path,
    *,
    extra_ignore_globs: list[str] | None = None,
) -> list[Path]:
    root = Path(workspace_root).resolve()
    ignore_patterns = _parse_gitignore(root) + list(extra_ignore_globs or [])
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in DEFAULT_IGNORE_DIRS:
                continue
            rel = p.relative_to(root).as_posix()
            if _matches_ignore(rel + "/", ignore_patterns):
                continue
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in DEFAULT_IGNORE_DIRS for part in p.parts):
            continue
        if _matches_ignore(rel, ignore_patterns):
            continue
        files.append(p)
    files.sort()
    return files


def build_file_map(
    workspace_root: str | Path,
    *,
    extra_ignore_globs: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    root = Path(workspace_root).resolve()
    result: dict[str, dict[str, Any]] = {}
    for p in scan_workspace_files(root, extra_ignore_globs=extra_ignore_globs):
        st = p.stat()
        rel = p.relative_to(root).as_posix()
        result[rel] = {
            "hash": _hash_file(p),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    return result


def compute_root_hash(files_map: dict[str, dict[str, Any]]) -> str:
    if not files_map:
        return _sha256_bytes(b"")
    lines = [f"{k}:{files_map[k].get('hash','')}" for k in sorted(files_map.keys())]
    return _sha256_bytes("\n".join(lines).encode("utf-8"))


def build_manifest(
    workspace_root: str | Path,
    *,
    extra_ignore_globs: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    files = build_file_map(root, extra_ignore_globs=extra_ignore_globs)
    return {
        "version": MANIFEST_VERSION,
        "generated_at": _utc_now_iso(),
        "workspace_root": str(root),
        "root_hash": compute_root_hash(files),
        "files": files,
    }


def default_manifest_path(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    return root / ".tau" / "index" / "merkle.json"


def default_stats_path(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    return root / ".tau" / "index" / INDEX_STATS_NAME


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("files"), dict):
        return None
    return data


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(p)


def diff_manifests(old_manifest: dict[str, Any] | None, new_manifest: dict[str, Any]) -> ChangedFiles:
    old_files = (old_manifest or {}).get("files", {}) or {}
    new_files = new_manifest.get("files", {}) or {}

    old_paths = set(old_files.keys())
    new_paths = set(new_files.keys())
    added = sorted(new_paths - old_paths)
    deleted = sorted(old_paths - new_paths)

    modified: list[str] = []
    unchanged_count = 0
    for rel in sorted(old_paths & new_paths):
        old_hash = (old_files.get(rel) or {}).get("hash", "")
        new_hash = (new_files.get(rel) or {}).get("hash", "")
        if old_hash != new_hash:
            modified.append(rel)
        else:
            unchanged_count += 1
    return ChangedFiles(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged_count=unchanged_count,
    )


def detect_workspace_changes(
    workspace_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    extra_ignore_globs: list[str] | None = None,
) -> tuple[ChangedFiles, dict[str, Any], dict[str, Any] | None]:
    mpath = Path(manifest_path) if manifest_path else default_manifest_path(workspace_root)
    old_manifest = load_manifest(mpath)
    new_manifest = build_manifest(workspace_root, extra_ignore_globs=extra_ignore_globs)
    changes = diff_manifests(old_manifest, new_manifest)
    return changes, new_manifest, old_manifest


def refresh_code_index(
    workspace_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    extra_ignore_globs: list[str] | None = None,
) -> dict[str, Any]:
    """Rebuild manifest, persist it, persist stats, and return refresh summary."""
    start = time.perf_counter()
    mpath = Path(manifest_path) if manifest_path else default_manifest_path(workspace_root)
    spath = Path(stats_path) if stats_path else default_stats_path(workspace_root)
    changes, new_manifest, old_manifest = detect_workspace_changes(
        workspace_root,
        manifest_path=mpath,
        extra_ignore_globs=extra_ignore_globs,
    )
    save_manifest(mpath, new_manifest)
    duration_ms = int((time.perf_counter() - start) * 1000)
    stats = {
        "version": MANIFEST_VERSION,
        "generated_at": _utc_now_iso(),
        "workspace_root": str(Path(workspace_root).resolve()),
        "manifest_path": str(mpath),
        "root_hash": new_manifest.get("root_hash", ""),
        "file_count": len(new_manifest.get("files", {})),
        "added_count": len(changes.added),
        "modified_count": len(changes.modified),
        "deleted_count": len(changes.deleted),
        "changed_count": len(changes.added) + len(changes.modified) + len(changes.deleted),
        "unchanged_count": changes.unchanged_count,
        "duration_ms": duration_ms,
        "had_previous_manifest": old_manifest is not None,
    }
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats["changes"] = changes
    return stats


def load_index_stats(workspace_root: str | Path, *, stats_path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(stats_path) if stats_path else default_stats_path(workspace_root)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
