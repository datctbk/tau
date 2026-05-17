from __future__ import annotations

from pathlib import Path

from tau.core.code_index import build_manifest, diff_manifests
from tau.core.semantic_pipeline import embed_text_local_hash, ingest_workspace_changes
from tau.core.semantic_store import SemanticStore


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_embed_text_local_hash_is_deterministic():
    v1 = embed_text_local_hash("retry backoff logic")
    v2 = embed_text_local_hash("retry backoff logic")
    assert len(v1) == len(v2) == 256
    assert v1 == v2


def test_ingest_workspace_changes_populates_semantic_store(tmp_path: Path):
    _write(tmp_path / "src" / "auth.py", "def verify_token(token):\n    return token == 'ok'\n")
    old = {"files": {}, "tree": {"kind": "dir", "hash": "", "children": {}}}
    new = build_manifest(tmp_path)
    changes = diff_manifests(old, new)

    db = tmp_path / "semantic.db"
    stats = ingest_workspace_changes(tmp_path, changes, model="local-hash-v1", db_path=str(db))
    assert int(stats["ingested_chunks"]) >= 1

    store = SemanticStore(db_path=db)
    try:
        hits = store.hybrid_search(
            query="verify token",
            query_vector=embed_text_local_hash("verify token"),
            model="local-hash-v1",
            limit=5,
        )
        assert hits
        assert any("auth.py" in h.path for h in hits)
    finally:
        store.close()

