"""Incremental code index primitives (Merkle manifest + changed-file detector).

This module is intentionally dependency-light and reusable by core or extensions.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau.core.chunker import chunk_file
from tau.core.embedding_cache import EmbeddingCache
from tau.core.retrieval_mode import retrieval_mode_label, semantic_retrieval_enabled
from tau.core.semantic_pipeline import ingest_workspace_changes

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

MANIFEST_VERSION = 2
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


@dataclass
class MerkleNode:
    """Hierarchical Merkle node for workspace tree hashing."""
    kind: str  # "dir" | "file"
    hash: str
    children: dict[str, "MerkleNode"] | None = None


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


def _compute_dir_hash(children: dict[str, MerkleNode]) -> str:
    if not children:
        return _sha256_bytes(b"")
    lines = [f"{name}:{children[name].hash}" for name in sorted(children.keys())]
    return _sha256_bytes("\n".join(lines).encode("utf-8"))


def build_merkle_tree(files_map: dict[str, dict[str, Any]]) -> MerkleNode:
    """Build a directory/file Merkle tree from flat file map."""
    root = MerkleNode(kind="dir", hash="", children={})
    # 1) materialize paths
    for rel in sorted(files_map.keys()):
        parts = rel.split("/")
        cur = root
        assert cur.children is not None
        for part in parts[:-1]:
            nxt = cur.children.get(part)
            if nxt is None:
                nxt = MerkleNode(kind="dir", hash="", children={})
                cur.children[part] = nxt
            cur = nxt
            if cur.children is None:
                cur.children = {}
        file_hash = str((files_map.get(rel) or {}).get("hash", ""))
        cur.children[parts[-1]] = MerkleNode(kind="file", hash=file_hash, children=None)

    # 2) post-order hash fill
    def _finalize(node: MerkleNode) -> str:
        if node.kind == "file":
            return node.hash
        kids = node.children or {}
        for k in sorted(kids.keys()):
            _finalize(kids[k])
        node.hash = _compute_dir_hash(kids)
        return node.hash

    _finalize(root)
    return root


def _node_to_dict(node: MerkleNode) -> dict[str, Any]:
    if node.kind == "file":
        return {"kind": "file", "hash": node.hash}
    return {
        "kind": "dir",
        "hash": node.hash,
        "children": {k: _node_to_dict(v) for k, v in sorted((node.children or {}).items())},
    }


def _node_from_dict(data: dict[str, Any] | None) -> MerkleNode | None:
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind", ""))
    h = str(data.get("hash", ""))
    if kind == "file":
        return MerkleNode(kind="file", hash=h, children=None)
    if kind != "dir":
        return None
    raw_children = data.get("children", {})
    children: dict[str, MerkleNode] = {}
    if isinstance(raw_children, dict):
        for name, child in raw_children.items():
            parsed = _node_from_dict(child if isinstance(child, dict) else None)
            if parsed is not None:
                children[str(name)] = parsed
    return MerkleNode(kind="dir", hash=h, children=children)


def build_manifest(
    workspace_root: str | Path,
    *,
    extra_ignore_globs: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    files = build_file_map(root, extra_ignore_globs=extra_ignore_globs)
    tree = build_merkle_tree(files)
    return {
        "version": MANIFEST_VERSION,
        "generated_at": _utc_now_iso(),
        "workspace_root": str(root),
        "root_hash": tree.hash or compute_root_hash(files),
        "files": files,
        "tree": _node_to_dict(tree),
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

    old_tree = _node_from_dict((old_manifest or {}).get("tree"))
    new_tree = _node_from_dict(new_manifest.get("tree"))

    # Backward-compatible fallback for v1 manifests without tree.
    if old_tree is None or new_tree is None:
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

    added: list[str] = []
    deleted: list[str] = []
    modified: list[str] = []
    unchanged_count = 0

    def _collect_all_files(node: MerkleNode, prefix: str, out: list[str]) -> None:
        if node.kind == "file":
            out.append(prefix)
            return
        for name, child in sorted((node.children or {}).items()):
            p = f"{prefix}/{name}" if prefix else name
            _collect_all_files(child, p, out)

    def _walk(old_node: MerkleNode | None, new_node: MerkleNode | None, prefix: str = "") -> None:
        nonlocal unchanged_count
        if old_node is None and new_node is None:
            return
        if old_node is None and new_node is not None:
            _collect_all_files(new_node, prefix, added)
            return
        if new_node is None and old_node is not None:
            _collect_all_files(old_node, prefix, deleted)
            return
        assert old_node is not None and new_node is not None
        if old_node.kind == "file" and new_node.kind == "file":
            if old_node.hash == new_node.hash:
                unchanged_count += 1
            else:
                modified.append(prefix)
            return
        if old_node.kind != new_node.kind:
            _collect_all_files(old_node, prefix, deleted)
            _collect_all_files(new_node, prefix, added)
            return
        if old_node.hash == new_node.hash:
            # Entire directory unchanged; count leaves without per-file hash compares.
            tmp: list[str] = []
            _collect_all_files(new_node, prefix, tmp)
            unchanged_count += len(tmp)
            return
        old_children = old_node.children or {}
        new_children = new_node.children or {}
        names = sorted(set(old_children.keys()) | set(new_children.keys()))
        for name in names:
            child_old = old_children.get(name)
            child_new = new_children.get(name)
            child_prefix = f"{prefix}/{name}" if prefix else name
            _walk(child_old, child_new, child_prefix)

    _walk(old_tree, new_tree, "")
    added.sort()
    deleted.sort()
    modified.sort()
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
        "retrieval_mode": retrieval_mode_label(),
    }
    if semantic_retrieval_enabled() and _flag_enabled(os.getenv("TAU_EMBEDDING_CACHE_ENABLED")):
        model = os.getenv("TAU_EMBEDDING_CACHE_MODEL", "default")
        cache_db_path = os.getenv("TAU_EMBEDDING_CACHE_DB_PATH")
        stats["embedding_cache"] = _collect_embedding_cache_stats(
            workspace_root,
            changes,
            model=model,
            cache_db_path=cache_db_path,
        )
    if semantic_retrieval_enabled() and _flag_enabled(os.getenv("TAU_SEMANTIC_STORE_ENABLED", "1")):
        stats["semantic_store"] = ingest_workspace_changes(
            workspace_root,
            changes,
            model=os.getenv("TAU_SEMANTIC_MODEL", "local-hash-v1"),
            db_path=os.getenv("TAU_SEMANTIC_STORE_DB_PATH"),
        )
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


def _flag_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _collect_embedding_cache_stats(
    workspace_root: str | Path,
    changes: ChangedFiles,
    *,
    model: str,
    cache_db_path: str | None,
) -> dict[str, int | str]:
    root = Path(workspace_root).resolve()
    chunk_count = 0
    cache_hit = 0
    cache_miss = 0
    cache = EmbeddingCache(db_path=Path(cache_db_path) if cache_db_path else None)
    try:
        for rel in changes.changed:
            p = root / rel
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunks = chunk_file(rel, text)
            chunk_count += len(chunks)
            for c in chunks:
                got = cache.get(c.content_hash, model)
                if got is None:
                    cache_miss += 1
                else:
                    cache_hit += 1
    finally:
        cache.close()
    return {
        "model": model,
        "chunk_count": chunk_count,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "recomputed_chunks": cache_miss,
    }
