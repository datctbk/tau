from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


TAU_HOME = Path.home() / ".tau"
DEFAULT_DB_PATH = TAU_HOME / "embedding_cache.db"


def hash_chunk_content(content: str) -> str:
    """Return deterministic SHA-256 hash for chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_hash: str
    model: str
    dim: int
    vector: list[float]
    updated_at: float


class EmbeddingCache:
    """SQLite-backed embedding cache keyed by (chunk_hash, model)."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    chunk_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chunk_hash, model)
                );

                CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at
                ON embedding_cache(updated_at DESC);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(self, chunk_hash: str, model: str) -> EmbeddingRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT chunk_hash, model, dim, vector_json, updated_at
                FROM embedding_cache
                WHERE chunk_hash = ? AND model = ?
                """,
                (chunk_hash, model),
            ).fetchone()
        if row is None:
            return None
        vector = json.loads(row["vector_json"])
        return EmbeddingRecord(
            chunk_hash=row["chunk_hash"],
            model=row["model"],
            dim=row["dim"],
            vector=[float(v) for v in vector],
            updated_at=float(row["updated_at"]),
        )

    def set(self, chunk_hash: str, model: str, vector: list[float]) -> EmbeddingRecord:
        if not vector:
            raise ValueError("vector must not be empty")
        now = time.time()
        dim = len(vector)
        vector_json = json.dumps(vector, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embedding_cache(chunk_hash, model, dim, vector_json, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chunk_hash, model) DO UPDATE SET
                    dim = excluded.dim,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (chunk_hash, model, dim, vector_json, now),
            )
        return EmbeddingRecord(
            chunk_hash=chunk_hash,
            model=model,
            dim=dim,
            vector=[float(v) for v in vector],
            updated_at=now,
        )

    def delete(self, chunk_hash: str, model: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM embedding_cache WHERE chunk_hash = ? AND model = ?",
                (chunk_hash, model),
            )
        return int(cur.rowcount or 0)

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM embedding_cache").fetchone()
        return int(row["c"]) if row is not None else 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embedding_cache")

