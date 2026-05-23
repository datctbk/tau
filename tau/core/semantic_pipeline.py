from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from tau.core.chunker import chunk_file
from tau.core.semantic_store import SemanticStore

if TYPE_CHECKING:
    from tau.core.code_index import ChangedFiles


def _tokenize(text: str) -> list[str]:
    return [x for x in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(x) >= 2]


def semantic_model_name() -> str:
    return os.getenv("TAU_SEMANTIC_MODEL", "local-hash-v1")


def embed_text_local_hash(text: str, dim: int = 256) -> list[float]:
    """Deterministic local embedding (no external API).

    This is a lightweight semantic baseline used for opt-in local retrieval.
    """
    vec = [0.0] * dim
    toks = _tokenize(text)
    if not toks:
        return vec
    for t in toks:
        h = hash(t)
        idx = abs(h) % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


def ingest_workspace_changes(
    workspace_root: str | Path,
    changes: "ChangedFiles",
    *,
    model: str | None = None,
    db_path: str | None = None,
) -> dict[str, int | str]:
    root = Path(workspace_root).resolve()
    model_name = model or semantic_model_name()
    store = SemanticStore(db_path=Path(db_path) if db_path else None)
    ingested_chunks = 0
    deleted_files = 0
    try:
        for rel in changes.deleted:
            deleted_files += store.delete_path(rel)
        for rel in changes.changed:
            p = root / rel
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for c in chunk_file(rel, text):
                store.upsert_chunk(
                    chunk_id=c.id,
                    path=c.path,
                    language=c.language,
                    kind=c.kind,
                    name=c.name,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content,
                    content_hash=c.content_hash,
                )
                vec = embed_text_local_hash(c.content)
                store.upsert_vector(chunk_id=c.id, model=model_name, vector=vec)
                ingested_chunks += 1
    finally:
        store.close()
    return {
        "model": model_name,
        "ingested_chunks": ingested_chunks,
        "deleted_files": deleted_files,
    }
