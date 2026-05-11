from __future__ import annotations

from tau.core.semantic_store import SemanticStore


def test_upsert_chunk_and_lexical_search(tmp_path):
    db = SemanticStore(db_path=tmp_path / "semantic.db")
    try:
        db.upsert_chunk(
            chunk_id="c1",
            path="src/a.py",
            language="python",
            kind="function",
            name="retry_backoff",
            start_line=1,
            end_line=5,
            content="def retry_backoff():\n    return 1\n",
            content_hash="h1",
        )
        hits = db.lexical_search("retry backoff", limit=5)
        assert hits
        assert hits[0].chunk_id == "c1"
    finally:
        db.close()


def test_semantic_search_and_hybrid_merge(tmp_path):
    db = SemanticStore(db_path=tmp_path / "semantic.db")
    try:
        db.upsert_chunk(
            chunk_id="c1",
            path="src/a.py",
            language="python",
            kind="function",
            name="f1",
            start_line=1,
            end_line=2,
            content="alpha beta gamma",
            content_hash="h1",
        )
        db.upsert_chunk(
            chunk_id="c2",
            path="src/b.py",
            language="python",
            kind="function",
            name="f2",
            start_line=1,
            end_line=2,
            content="delta epsilon zeta",
            content_hash="h2",
        )
        db.upsert_vector(chunk_id="c1", model="m1", vector=[1.0, 0.0, 0.0])
        db.upsert_vector(chunk_id="c2", model="m1", vector=[0.0, 1.0, 0.0])

        sem = db.semantic_search([1.0, 0.0, 0.0], model="m1", limit=5)
        assert sem
        assert sem[0].chunk_id == "c1"

        hybrid = db.hybrid_search(
            query="delta",
            query_vector=[1.0, 0.0, 0.0],
            model="m1",
            limit=5,
            lexical_weight=0.5,
            semantic_weight=0.5,
        )
        assert hybrid
        ids = [h.chunk_id for h in hybrid]
        assert "c1" in ids
        assert "c2" in ids
    finally:
        db.close()


def test_delete_path_cascades_vectors(tmp_path):
    db = SemanticStore(db_path=tmp_path / "semantic.db")
    try:
        db.upsert_chunk(
            chunk_id="c1",
            path="src/a.py",
            language="python",
            kind="function",
            name="f1",
            start_line=1,
            end_line=1,
            content="x",
            content_hash="h1",
        )
        db.upsert_vector(chunk_id="c1", model="m1", vector=[1.0])
        deleted = db.delete_path("src/a.py")
        assert deleted == 1
        assert db.lexical_search("x", limit=3) == []
    finally:
        db.close()

