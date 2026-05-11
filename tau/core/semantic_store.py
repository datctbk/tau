"""Local-first semantic store (SQLite + FTS5 + vector table).

Phase-4 prep: keeps lexical default path intact while enabling optional
hybrid retrieval when semantic mode is enabled.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

TAU_HOME = Path.home() / ".tau"
DEFAULT_DB_PATH = TAU_HOME / "semantic_store.db"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    path: str
    language: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    updated_at: float


@dataclass(frozen=True)
class HybridMatch:
    chunk_id: str
    lexical_score: float
    semantic_score: float
    score: float
    path: str
    start_line: int
    end_line: int
    snippet: str


class SemanticStore:
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
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_updated ON chunks(updated_at DESC);

                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chunk_id, model)
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_vectors_model ON chunk_vectors(model);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    path,
                    name,
                    content=chunks,
                    content_rowid=rowid
                );

                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content, path, name)
                    VALUES (new.rowid, new.content, new.path, new.name);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, path, name)
                    VALUES('delete', old.rowid, old.content, old.path, old.name);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content, path, name)
                    VALUES('delete', old.rowid, old.content, old.path, old.name);
                    INSERT INTO chunks_fts(rowid, content, path, name)
                    VALUES (new.rowid, new.content, new.path, new.name);
                END;
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        path: str,
        language: str,
        kind: str,
        name: str,
        start_line: int,
        end_line: int,
        content: str,
        content_hash: str,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunks(
                    chunk_id, path, language, kind, name, start_line, end_line, content, content_hash, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    path=excluded.path,
                    language=excluded.language,
                    kind=excluded.kind,
                    name=excluded.name,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    content=excluded.content,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (chunk_id, path, language, kind, name, start_line, end_line, content, content_hash, now),
            )

    def upsert_vector(self, *, chunk_id: str, model: str, vector: list[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunk_vectors(chunk_id, model, dim, vector_json, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, model) DO UPDATE SET
                    dim=excluded.dim,
                    vector_json=excluded.vector_json,
                    updated_at=excluded.updated_at
                """,
                (chunk_id, model, len(vector), json.dumps(vector, separators=(",", ":")), now),
            )

    def delete_path(self, path: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        return int(cur.rowcount or 0)

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        q = query.strip()
        if not q:
            return ""
        # lightweight sanitize, similar spirit to SessionDB
        cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", ".", " ", "\"", "*"} else " " for ch in q)
        cleaned = " ".join(cleaned.split())
        return cleaned

    def lexical_search(self, query: str, *, limit: int = 20) -> list[HybridMatch]:
        q = self._sanitize_fts5_query(query)
        if not q:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.path, c.start_line, c.end_line,
                       snippet(chunks_fts, 0, '', '', ' … ', 32) AS snippet,
                       bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (q, int(limit)),
            ).fetchall()
        out: list[HybridMatch] = []
        for r in rows:
            # bm25 lower is better; convert into "higher better" score
            lex = 1.0 / (1.0 + max(0.0, float(r["rank"])))
            out.append(
                HybridMatch(
                    chunk_id=r["chunk_id"],
                    lexical_score=lex,
                    semantic_score=0.0,
                    score=lex,
                    path=r["path"],
                    start_line=int(r["start_line"]),
                    end_line=int(r["end_line"]),
                    snippet=r["snippet"] or "",
                )
            )
        return out

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def semantic_search(self, query_vector: list[float], *, model: str, limit: int = 20) -> list[HybridMatch]:
        if not query_vector:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.path, c.start_line, c.end_line, c.content, v.vector_json
                FROM chunk_vectors v
                JOIN chunks c ON c.chunk_id = v.chunk_id
                WHERE v.model = ?
                """,
                (model,),
            ).fetchall()
        scored: list[HybridMatch] = []
        for r in rows:
            vec = [float(x) for x in json.loads(r["vector_json"])]
            sim = self._cosine(query_vector, vec)
            if sim <= 0.0:
                continue
            content = r["content"] or ""
            snippet = content[:220].replace("\n", " ")
            scored.append(
                HybridMatch(
                    chunk_id=r["chunk_id"],
                    lexical_score=0.0,
                    semantic_score=sim,
                    score=sim,
                    path=r["path"],
                    start_line=int(r["start_line"]),
                    end_line=int(r["end_line"]),
                    snippet=snippet,
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def hybrid_search(
        self,
        *,
        query: str,
        query_vector: list[float] | None,
        model: str,
        limit: int = 20,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> list[HybridMatch]:
        lex_hits = self.lexical_search(query, limit=max(limit * 2, 20))
        sem_hits = self.semantic_search(query_vector or [], model=model, limit=max(limit * 2, 20))
        merged: dict[str, HybridMatch] = {}

        for h in lex_hits:
            merged[h.chunk_id] = h
        for h in sem_hits:
            cur = merged.get(h.chunk_id)
            if cur is None:
                merged[h.chunk_id] = h
            else:
                merged[h.chunk_id] = HybridMatch(
                    chunk_id=cur.chunk_id,
                    lexical_score=cur.lexical_score,
                    semantic_score=h.semantic_score,
                    score=0.0,
                    path=cur.path,
                    start_line=cur.start_line,
                    end_line=cur.end_line,
                    snippet=cur.snippet or h.snippet,
                )

        out: list[HybridMatch] = []
        for v in merged.values():
            score = lexical_weight * v.lexical_score + semantic_weight * v.semantic_score
            out.append(
                HybridMatch(
                    chunk_id=v.chunk_id,
                    lexical_score=v.lexical_score,
                    semantic_score=v.semantic_score,
                    score=score,
                    path=v.path,
                    start_line=v.start_line,
                    end_line=v.end_line,
                    snippet=v.snippet,
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]

