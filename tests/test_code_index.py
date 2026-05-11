from __future__ import annotations

from pathlib import Path

from tau.core.chunker import chunk_file
from tau.core.code_index import (
    build_manifest,
    default_stats_path,
    default_manifest_path,
    detect_workspace_changes,
    diff_manifests,
    load_index_stats,
    load_manifest,
    refresh_code_index,
    save_manifest,
)
from tau.core.embedding_cache import EmbeddingCache


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_manifest_basic(tmp_path: Path):
    _write(tmp_path / "a.py", "print('a')\n")
    _write(tmp_path / "pkg" / "b.py", "print('b')\n")
    m = build_manifest(tmp_path)
    assert m["version"] == 2
    assert "root_hash" in m
    assert "tree" in m
    assert "a.py" in m["files"]
    assert "pkg/b.py" in m["files"]


def test_gitignore_applied(tmp_path: Path):
    _write(tmp_path / ".gitignore", "*.log\nbuild/\n")
    _write(tmp_path / "keep.py", "ok\n")
    _write(tmp_path / "debug.log", "x\n")
    _write(tmp_path / "build" / "x.py", "nope\n")
    m = build_manifest(tmp_path)
    assert "keep.py" in m["files"]
    assert "debug.log" not in m["files"]
    assert "build/x.py" not in m["files"]


def test_detect_changes_added_modified_deleted(tmp_path: Path):
    _write(tmp_path / "a.py", "v1\n")
    _write(tmp_path / "b.py", "v1\n")

    first = build_manifest(tmp_path)
    mp = default_manifest_path(tmp_path)
    save_manifest(mp, first)

    # mutate workspace
    _write(tmp_path / "a.py", "v2\n")  # modified
    (tmp_path / "b.py").unlink()  # deleted
    _write(tmp_path / "c.py", "new\n")  # added

    changes, new_manifest, old_manifest = detect_workspace_changes(tmp_path, manifest_path=mp)
    assert old_manifest is not None
    assert "a.py" in changes.modified
    assert "b.py" in changes.deleted
    assert "c.py" in changes.added
    assert new_manifest["files"]["a.py"]["hash"] != old_manifest["files"]["a.py"]["hash"]


def test_diff_manifests_unchanged_count(tmp_path: Path):
    _write(tmp_path / "a.py", "same\n")
    old = build_manifest(tmp_path)
    new = build_manifest(tmp_path)
    d = diff_manifests(old, new)
    assert d.added == []
    assert d.modified == []
    assert d.deleted == []
    assert d.unchanged_count == 1


def test_manifest_save_load_roundtrip(tmp_path: Path):
    _write(tmp_path / "x.py", "1\n")
    m = build_manifest(tmp_path)
    p = default_manifest_path(tmp_path)
    save_manifest(p, m)
    got = load_manifest(p)
    assert got is not None
    assert got["root_hash"] == m["root_hash"]


def test_refresh_persists_stats_and_manifest(tmp_path: Path):
    _write(tmp_path / "k.py", "1\n")
    stats = refresh_code_index(tmp_path)
    assert stats["file_count"] >= 1
    assert default_manifest_path(tmp_path).is_file()
    assert default_stats_path(tmp_path).is_file()
    loaded = load_index_stats(tmp_path)
    assert loaded is not None
    assert loaded.get("file_count", 0) >= 1
    assert loaded.get("retrieval_mode") == "lexical"


def test_refresh_embedding_cache_stats_when_enabled(tmp_path: Path, monkeypatch):
    f = tmp_path / "a.py"
    _write(f, "def add(a, b):\n    return a + b\n")
    text = f.read_text(encoding="utf-8")
    chunks = chunk_file("a.py", text)
    assert chunks

    db = tmp_path / "emb.db"
    cache = EmbeddingCache(db_path=db)
    try:
        # Preload first chunk as cache-hit, leave rest as miss.
        cache.set(chunks[0].content_hash, "test-embed-v1", [0.1, 0.2])
    finally:
        cache.close()

    monkeypatch.setenv("TAU_SEMANTIC_RETRIEVAL", "1")
    monkeypatch.setenv("TAU_EMBEDDING_CACHE_ENABLED", "1")
    monkeypatch.setenv("TAU_EMBEDDING_CACHE_MODEL", "test-embed-v1")
    monkeypatch.setenv("TAU_EMBEDDING_CACHE_DB_PATH", str(db))

    stats = refresh_code_index(tmp_path)
    emb = stats.get("embedding_cache")
    assert isinstance(emb, dict)
    assert emb.get("model") == "test-embed-v1"
    assert int(emb.get("chunk_count", 0)) >= 1
    assert int(emb.get("cache_hit", 0)) >= 1
    assert int(emb.get("cache_miss", 0)) >= 0


def test_refresh_embedding_cache_ignored_when_semantic_off(tmp_path: Path, monkeypatch):
    _write(tmp_path / "a.py", "def f():\n    return 1\n")
    monkeypatch.delenv("TAU_SEMANTIC_RETRIEVAL", raising=False)
    monkeypatch.setenv("TAU_EMBEDDING_CACHE_ENABLED", "1")
    stats = refresh_code_index(tmp_path)
    assert stats.get("retrieval_mode") == "lexical"
    assert "embedding_cache" not in stats
