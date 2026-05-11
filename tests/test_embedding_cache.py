from __future__ import annotations

from tau.core.embedding_cache import EmbeddingCache, hash_chunk_content


def test_hash_chunk_content_deterministic():
    a = hash_chunk_content("hello")
    b = hash_chunk_content("hello")
    c = hash_chunk_content("hello!")
    assert a == b
    assert a != c


def test_cache_set_get_and_count(tmp_path):
    cache = EmbeddingCache(db_path=tmp_path / "embeddings.db")
    try:
        key = hash_chunk_content("def add(a, b): return a + b")
        model = "test-embed-v1"
        vec = [0.1, 0.2, 0.3]
        rec = cache.set(key, model, vec)
        assert rec.chunk_hash == key
        assert rec.model == model
        assert rec.dim == 3
        assert cache.count() == 1

        got = cache.get(key, model)
        assert got is not None
        assert got.vector == vec
        assert got.dim == 3
    finally:
        cache.close()


def test_cache_upsert_replaces_vector(tmp_path):
    cache = EmbeddingCache(db_path=tmp_path / "embeddings.db")
    try:
        key = hash_chunk_content("class A: pass")
        model = "test-embed-v1"
        cache.set(key, model, [1.0, 2.0])
        cache.set(key, model, [3.0, 4.0])
        assert cache.count() == 1
        got = cache.get(key, model)
        assert got is not None
        assert got.vector == [3.0, 4.0]
    finally:
        cache.close()


def test_cache_delete_and_clear(tmp_path):
    cache = EmbeddingCache(db_path=tmp_path / "embeddings.db")
    try:
        k1 = hash_chunk_content("x")
        k2 = hash_chunk_content("y")
        model = "test-embed-v1"
        cache.set(k1, model, [0.5])
        cache.set(k2, model, [0.7])
        assert cache.count() == 2
        assert cache.delete(k1, model) == 1
        assert cache.get(k1, model) is None
        assert cache.count() == 1
        cache.clear()
        assert cache.count() == 0
    finally:
        cache.close()

