from __future__ import annotations

import os
from dataclasses import dataclass

from tau.core.retrieval_mode import semantic_retrieval_enabled

def semantic_vector_index_enabled() -> bool:
    """Feature gate for Phase 4 vector search pipeline."""
    if not semantic_retrieval_enabled():
        return False
    return os.getenv("TAU_SEMANTIC_VECTOR_INDEX_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class VectorMatch:
    chunk_id: str
    score: float


class VectorIndex:
    """Phase 4 placeholder. Explicitly disabled unless feature flag is enabled."""

    def __init__(self):
        if not semantic_vector_index_enabled():
            raise RuntimeError(
                "Vector index is disabled. "
                "Set TAU_SEMANTIC_VECTOR_INDEX_ENABLED=1 to enable."
            )

    def add(self, chunk_id: str, vector: list[float]) -> None:
        raise NotImplementedError("VectorIndex implementation is planned for Phase 4.")

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        raise NotImplementedError("VectorIndex implementation is planned for Phase 4.")
