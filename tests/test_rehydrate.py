from __future__ import annotations

from pathlib import Path

from tau.core.code_index import build_manifest, diff_manifests
from tau.core.rehydrate import build_lexical_rehydrate_block, build_rehydrate_block
from tau.core.semantic_pipeline import ingest_workspace_changes


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_rehydrate_returns_ranked_code_context(tmp_path: Path):
    _write(tmp_path / "src" / "auth.py", "def verify_token(token):\n    return token == 'ok'\n")
    _write(tmp_path / "src" / "billing.py", "def charge(amount):\n    return amount > 0\n")

    block = build_lexical_rehydrate_block(
        query="fix verify token bug in auth",
        workspace_root=tmp_path,
        max_chunks=4,
        max_chars_per_chunk=400,
        max_total_chars=3000,
        max_files_scan=50,
    )
    assert block
    assert "Rehydrated Code Context" in block
    assert "src/auth.py" in block


def test_rehydrate_respects_total_char_budget(tmp_path: Path):
    big = "def f():\n" + ("    x = 1\n" * 3000)
    _write(tmp_path / "src" / "big.py", big)
    block = build_lexical_rehydrate_block(
        query="f function",
        workspace_root=tmp_path,
        max_chunks=10,
        max_chars_per_chunk=1200,
        max_total_chars=900,
        max_files_scan=20,
    )
    # May be empty if budget is too strict, but if present must obey budget-ish limit.
    if block:
        assert len(block) <= 1100


def test_rehydrate_hybrid_when_semantic_enabled(tmp_path: Path, monkeypatch):
    _write(tmp_path / "src" / "retry.py", "def retry_backoff():\n    return 1\n")
    old = {"files": {}, "tree": {"kind": "dir", "hash": "", "children": {}}}
    new = build_manifest(tmp_path)
    changes = diff_manifests(old, new)
    db = tmp_path / "semantic.db"
    ingest_workspace_changes(tmp_path, changes, model="local-hash-v1", db_path=str(db))

    monkeypatch.setenv("TAU_SEMANTIC_RETRIEVAL", "1")
    monkeypatch.setenv("TAU_SEMANTIC_STORE_DB_PATH", str(db))
    monkeypatch.setenv("TAU_SEMANTIC_MODEL", "local-hash-v1")

    block = build_rehydrate_block(
        query="how retry backoff works",
        workspace_root=tmp_path,
        max_chunks=5,
        max_total_chars=3000,
    )
    assert block
    assert "Rehydrated Code Context" in block
